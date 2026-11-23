"""Forecast what each caching option would cost on a logged workload, and recommend one."""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .diagnose import Break, find_breaks, summarise
from .logs import by_app, conversation_key
from .messages import (FIVE_MIN, ONE_HOUR, BreakPlan, Template, canonicalize_request, cc,
                       est_tokens, learn, learn_breakpoints, place_breakpoints, prefix_blocks,
                       strip_cache_control, system_text)
from .models import TOOL_USE_OVERHEAD, min_cacheable
from .sim import CacheSim, Usage

OPTIONS = {     # key -> (label, changes the prompt text)
    "off": ("Caching off", False),
    "auto_5m": ("Automatic caching, 5-minute", False),
    "auto_1h": ("Automatic caching, 1-hour", False),
    "plan_5m": ("Planned breakpoints, 5-minute", False),
    "plan_1h": ("Planned breakpoints, 1-hour", False),
    "fix_5m": ("Prompt fix + breakpoints, 5-minute", True),
    "fix_1h": ("Prompt fix + breakpoints, 1-hour", True),
    "logged": ("As logged (current settings)", False),
}
SIMPLICITY = ["off", "auto_5m", "auto_1h", "plan_5m", "plan_1h"]


@dataclass
class Knowledge:
    """What was learned from a log: one prompt template and one breakpoint plan per application."""
    templates: dict[str, Template | None] = field(default_factory=dict)
    plans: dict[str, BreakPlan] = field(default_factory=dict)


def learn_from(rows: list[dict]) -> Knowledge:
    k = Knowledge()
    for app, rs in by_app(rows).items():
        k.templates[app] = learn([system_text(r["request"]) for r in rs], [conversation_key(r) for r in rs])
        k.plans[app] = learn_breakpoints([r["request"] for r in rs])
    return k


def build(option: str, row: dict, k: Knowledge) -> dict:
    """The request as it would be sent under one option."""
    req, app = row["request"], row["app"]
    ttl = ONE_HOUR if option.endswith("_1h") else FIVE_MIN
    if option == "logged":
        return {f: req[f] for f in ("tools", "system", "messages", "cache_control") if req.get(f)}
    if option == "off":
        return strip_cache_control(req)
    if option.startswith("auto"):
        return dict(strip_cache_control(req), cache_control=cc(ttl))
    plan = k.plans.get(app) or BreakPlan(head=False, offsets=())
    if option.startswith("plan"):
        return place_breakpoints(req, plan=plan, ttl=ttl)
    return canonicalize_request(req, k.templates.get(app), plan, ttl=ttl)


def has_fix(rows: list[dict], k: Knowledge) -> bool:
    tpl = k.templates.get(rows[0]["app"])
    return bool(tpl and any(s.kind == "volatile" or s.distinct > 1 for s in tpl.slots()))


def row_minimum(row: dict, min_tokens: int | None = None, model: str | None = None) -> int:
    """Minimum cacheable length for one logged request: an explicit value wins, then an
    explicit model, then the model named in the request itself."""
    if min_tokens:
        return min_tokens
    return min_cacheable(model or row["request"].get("model"))[0]


def simulate(rows: list[dict], k: Knowledge, options: list[str], min_tokens: int | None = None,
             model: str | None = None) -> dict:
    """{app: {option: Usage}} from the local cache model."""
    out: dict[str, dict[str, Usage]] = defaultdict(dict)
    floors = [row_minimum(r, min_tokens, model) for r in rows]
    for opt in options:
        sim = CacheSim()
        for row, floor in zip(rows, floors):
            out[row["app"]].setdefault(opt, Usage()).add(sim.request(build(opt, row, k), row["ts"], floor))
    return dict(out)


def reusable_share(rows: list[dict], ttl: float = FIVE_MIN) -> float:
    """Share of input tokens that repeat an earlier request's prefix within the lifetime:
    the most that caching could read from these prompts exactly as written."""
    seen, reuse, total = [], 0, 0
    for r in rows:
        blocks = [t for t, _ in prefix_blocks(strip_cache_control(r["request"]))]
        best = 0
        for ts, prev in seen[-40:]:
            if r["ts"] - ts > ttl:
                continue
            n = 0
            while n < min(len(prev), len(blocks)) and prev[n] == blocks[n]:
                n += 1
            best = max(best, sum(est_tokens(b) for b in blocks[:n]))
        reuse += best
        total += sum(est_tokens(b) for b in blocks)
        seen.append((r["ts"], blocks))
    return reuse / total if total else 0.0


@dataclass
class Verdict:
    app: str
    requests: int
    tokens: int
    layout: str
    reusable: float
    median_gap: float
    gaps_over_5m: float
    usage: dict[str, Usage]
    recommended: str
    reason: str
    fix_option: str | None            # best option that changes the prompt, if clearly better
    plan: BreakPlan
    breaks: list[Break]
    in_sample: bool = True
    minimum: int = 1024               # minimum cacheable length used for this application
    minimum_known: bool = True        # False when the model was not recognised
    typical_tokens: int = 0           # median prompt size (estimate)

    def cost(self, opt: str) -> float:
        return self.usage[opt].cost_units() / max(1, self.usage["off"].total)

    def read(self, opt: str) -> float:
        u = self.usage[opt]
        return u.cache_read_input_tokens / max(1, u.total)


def _recommend(cost: dict[str, float]) -> tuple[str, str]:
    plain = [o for o in SIMPLICITY if o in cost]
    best = min(plain, key=lambda o: cost[o])
    if "logged" in cost and cost["logged"] < 0.97 and cost["logged"] <= cost[best] + 0.02:
        return "logged", ("your current settings are the cheapest option" if cost["logged"] <= cost[best]
                          else "your current settings are within 2 points of the best option")
    if cost[best] >= 0.97:
        return "off", "no option saves at least 3%, and cache writes cost extra"
    pick = next(o for o in plain if cost[o] <= cost[best] + 0.02)       # simplest within 2 points
    if pick == best:
        return pick, "lowest forecast cost"
    return pick, f"within 2 points of the best option and simpler to set up"


def _layout(reusable: float, cost: dict[str, float], breaks: list[Break], fixable: bool) -> str:
    head = [b for b in breaks if b.path.startswith(("system", "tools"))]
    if breaks and len(head) * 2 >= len(breaks) and (fixable or reusable < 0.15):
        return "changing content at the top of the prompt"
    if reusable < 0.15:
        return "little repeats between requests"
    if cost["auto_5m"] <= cost["plan_5m"] + 0.03 and cost["auto_5m"] < 0.97:
        return "append-only conversation"
    if any(b.kind == "inserted" for b in breaks) or cost["plan_5m"] < cost["auto_5m"] - 0.03:
        return "content inserted or changed mid-prompt"
    return "mixed"


def examine(rows: list[dict], k: Knowledge | None = None, min_tokens: int | None = None,
            model: str | None = None) -> list[Verdict]:
    """One verdict per application. If k is None the plan is learned from the same rows."""
    in_sample = k is None
    k = k or learn_from(rows)
    apps = by_app(rows)
    options = [o for o in OPTIONS if o != "logged"]
    if any(any(ttl for _, ttl in prefix_blocks(r["request"])) for r in rows):
        options.append("logged")
    usage = simulate(rows, k, options, min_tokens, model)
    out = []
    for app, rs in apps.items():
        u = dict(usage[app])
        fixable = has_fix(rs, k)
        if not fixable:
            u.pop("fix_5m", None), u.pop("fix_1h", None)
        base = max(1, u["off"].total)
        cost = {o: x.cost_units() / base for o, x in u.items()}
        pick, reason = _recommend(cost)
        fix = None
        if fixable:
            best_fix = min(("fix_5m", "fix_1h"), key=lambda o: cost[o])
            if cost[best_fix] < cost[pick] - 0.05:
                fix = best_fix
        gaps = [b["ts"] - a["ts"] for a, b in zip(rs, rs[1:])]
        breaks = find_breaks(app, rs)
        out.append(Verdict(app, len(rs), u["off"].total, _layout(reusable_share(rs), cost, breaks, fixable),
                           reusable_share(rs), statistics.median(gaps) if gaps else 0.0,
                           sum(g > FIVE_MIN for g in gaps) / len(gaps) if gaps else 0.0,
                           u, pick, reason, fix, k.plans.get(app) or BreakPlan(head=False, offsets=()),
                           breaks, in_sample, row_minimum(rs[0], min_tokens, model),
                           bool(min_tokens) or min_cacheable(model or rs[0]["request"].get("model"))[1],
                           sorted(sum(est_tokens(t) for t, _ in prefix_blocks(strip_cache_control(r["request"])))
                                  + (TOOL_USE_OVERHEAD if r["request"].get("tools") else 0)
                                  for r in rs)[len(rs) // 2]))
    return sorted(out, key=lambda v: -v.tokens)
