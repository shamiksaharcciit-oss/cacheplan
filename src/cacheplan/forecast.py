"""Forecast what each caching option would cost on a logged workload, and recommend one."""
from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .diagnose import Break, find_breaks
from .logs import by_app, conversation_key, real_usage
from .messages import (FIVE_MIN, ONE_HOUR, BreakPlan, Template, canonicalize_request, cc,
                       est_tokens, learn, learn_breakpoints, place_breakpoints, prefix_blocks,
                       strip_cache_control, system_text)
from .providers import Rules, rules as provider_rules
from .sim import CacheSim, ImplicitSim, Usage

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
NO_PLAN = BreakPlan(head=False, offsets=())
SEARCH_ROWS = 300            # requests per application used when searching for a plan


@dataclass
class Calibration:
    """How estimated tokens are turned into real ones for one application."""
    scale: float = 1.0           # multiplier on the character-based estimate
    overhead: int = 300          # tokens the API adds to requests that define tools
    responses: int = 0           # recorded responses the calibration was fitted on (0 = defaults)


@dataclass
class Knowledge:
    """What was learned from a log, per application."""
    templates: dict[str, Template | None] = field(default_factory=dict)
    plans: dict[str, dict[int, BreakPlan]] = field(default_factory=dict)      # app -> lifetime -> plan
    calibration: dict[str, Calibration] = field(default_factory=dict)
    specs: dict[str, Rules] = field(default_factory=dict)
    floors: dict[str, int] = field(default_factory=dict)                      # minimum cacheable tokens


def _est(req: dict) -> int:
    return sum(est_tokens(t) for t, _ in prefix_blocks(strip_cache_control(req)))


def calibrate(rows: list[dict], default_overhead: int) -> Calibration:
    """Fit real = scale * estimate + overhead * (has tools) on the responses in the log."""
    pts = [(_est(r["request"]), 1.0 if r["request"].get("tools") else 0.0, u.total)
           for r in rows if (u := real_usage(r)) and u.total > 0]
    if len(pts) < 5:
        return Calibration(1.0, default_overhead, 0)
    sxx = sum(x * x for x, _, _ in pts)
    sxt = sum(x * t for x, t, _ in pts)
    stt = sum(t * t for _, t, _ in pts)
    sxy = sum(x * y for x, _, y in pts)
    sty = sum(t * y for _, t, y in pts)
    det = sxx * stt - sxt * sxt
    spread = max(x for x, _, _ in pts) > 1.5 * min(x for x, _, _ in pts)
    if stt and spread and abs(det) > 1e-6 * sxx * stt:
        scale, overhead = (sxy * stt - sxt * sty) / det, (sxx * sty - sxt * sxy) / det
        if 0.5 <= scale <= 2.0 and 0 <= overhead <= 1000:
            return Calibration(scale, round(overhead), len(pts))
    # not enough variation to separate the two: keep the documented overhead, fit the scale
    num = sum(x * (y - default_overhead * t) for x, t, y in pts)
    scale = min(2.0, max(0.5, num / sxx)) if sxx else 1.0
    return Calibration(scale, default_overhead, len(pts))


def _model_of(rows: list[dict], model: str | None) -> str | None:
    return model or next((r["request"].get("model") for r in rows if r["request"].get("model")), None)


def _provider_of(rows: list[dict]) -> str:
    return rows[0].get("provider") or "anthropic"


def _units(u: Usage, r: Rules) -> float:
    return u.cost_units(r.read_x, r.write_short, r.write_long)


def _cost(rows, build_fn, floor, cal, r: Rules) -> float:
    sim = CacheSim(ttl_short=r.ttl_short, ttl_long=r.ttl_long)
    return sum(_units(sim.request(build_fn(row), row["ts"], floor, cal.overhead, cal.scale), r) for row in rows)


def choose_plan(rows: list[dict], ttl: int, floor: int, cal: Calibration, r: Rules) -> BreakPlan:
    """Pick breakpoint positions by replaying the log: candidates come from where prefixes
    were seen to repeat, and the cheapest combination under this lifetime and this log's
    real timing wins. Fewer breakpoints win ties."""
    rows = rows[:SEARCH_ROWS]
    seed = learn_breakpoints([r["request"] for r in rows])
    offsets = sorted(int(k[4:]) for k in seed.reuse if k.startswith("end-"))
    if r.mode != "breakpoints" or (ttl == ONE_HOUR and r.write_long is None):
        return NO_PLAN
    best, best_cost = NO_PLAN, _cost(rows, lambda row: strip_cache_control(row["request"]), floor, cal, r)
    heads = [False, True]
    consts = sorted({0, seed.const_prefix})
    for n in range(0, 3):
        for combo in itertools.combinations(offsets, n):
            for head in heads:
                for const in consts:
                    if not (head or const or combo):
                        continue
                    plan = BreakPlan(head, const, tuple(combo), seed.reuse)
                    c = _cost(rows, lambda row: place_breakpoints(row["request"], plan=plan, ttl=ttl), floor, cal, r)
                    if c < best_cost * 0.995:           # must beat the simpler plan by a real margin
                        best, best_cost = plan, c
    return best


def learn_from(rows: list[dict], min_tokens: int | None = None, model: str | None = None) -> Knowledge:
    k = Knowledge()
    for app, rs in by_app(rows).items():
        s = provider_rules(_provider_of(rs), _model_of(rs, model))
        k.specs[app] = s
        k.floors[app] = min_tokens or s.minimum
        k.calibration[app] = calibrate(rs, s.tool_overhead)
        k.templates[app] = learn([system_text(r["request"]) for r in rs], [conversation_key(r) for r in rs])
        k.plans[app] = {ttl: choose_plan(rs, ttl, k.floors[app], k.calibration[app], s)
                        for ttl in (FIVE_MIN, ONE_HOUR)}
    return k


def _fill(k: Knowledge, rows: list[dict], min_tokens: int | None, model: str | None) -> None:
    """Applications that were not in the learning log still need a model spec and a floor."""
    for app, rs in by_app(rows).items():
        if app not in k.specs:
            s = provider_rules(_provider_of(rs), _model_of(rs, model))
            k.specs[app], k.floors[app] = s, min_tokens or s.minimum
            k.calibration[app] = calibrate(rs, s.tool_overhead)
            k.templates.setdefault(app, None)
            k.plans.setdefault(app, {FIVE_MIN: NO_PLAN, ONE_HOUR: NO_PLAN})


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
    plan = (k.plans.get(app) or {}).get(ttl, NO_PLAN)
    if option.startswith("plan"):
        return place_breakpoints(req, plan=plan, ttl=ttl)
    return canonicalize_request(req, k.templates.get(app), plan, ttl=ttl)


def has_fix(rows: list[dict], k: Knowledge) -> bool:
    tpl = k.templates.get(rows[0]["app"])
    return bool(tpl and any(s.kind == "volatile" or s.distinct > 1 for s in tpl.slots()))


def simulate(rows: list[dict], k: Knowledge, options: list[str]) -> dict:
    """{app: {option: Usage}} from the local cache model. Each application is simulated
    under its own provider's rules."""
    out: dict[str, dict[str, Usage]] = defaultdict(dict)
    for app, rs in by_app(rows).items():
        r = k.specs.get(app) or Rules()
        cal = k.calibration.get(app) or Calibration()
        floor = k.floors.get(app, r.minimum)
        for opt in options:
            if opt.endswith("_1h") and r.write_long is None:
                continue
            if r.mode == "implicit" and opt.startswith("plan"):
                continue
            if not r.automatic and opt.startswith("auto"):
                continue
            total = Usage()
            if r.mode == "implicit":
                if opt == "off":                              # a reference point: these models always cache
                    for row in rs:
                        total.add(Usage(input_tokens=ImplicitSim(0).request(
                            strip_cache_control(row["request"]), row["ts"], floor, cal.overhead, cal.scale).total))
                else:
                    sim = ImplicitSim(r.ttl_long if opt.endswith("_1h") else r.ttl_short, r.interval)
                    for row in rs:
                        req = strip_cache_control(build(opt, row, k)) if opt.startswith("fix") else strip_cache_control(row["request"])
                        total.add(sim.request(req, row["ts"], floor, cal.overhead, cal.scale))
            else:
                sim = CacheSim(ttl_short=r.ttl_short, ttl_long=r.ttl_long)
                for row in rs:
                    total.add(sim.request(build(opt, row, k), row["ts"], floor, cal.overhead, cal.scale))
            out[app][opt] = total
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
    plans: dict[int, BreakPlan]
    breaks: list[Break]
    in_sample: bool = True
    minimum: int = 1024               # minimum cacheable length used for this application
    minimum_known: bool = True        # False when the model was not recognised
    typical_tokens: int = 0           # median prompt size
    read_x: float = 0.10              # cache read price relative to an input token
    price: float | None = None        # USD per million input tokens, if known
    calibration: Calibration = field(default_factory=Calibration)
    measured: Usage | None = None     # the provider's own counts for this log, if recorded
    rules: Rules = field(default_factory=Rules)   # the provider's caching rules for this model

    @property
    def plan(self) -> BreakPlan:
        return self.plans.get(ONE_HOUR if self.recommended.endswith("_1h") else FIVE_MIN, NO_PLAN)

    def cost(self, opt: str) -> float:
        return _units(self.usage[opt], self.rules) / max(1, self.usage["off"].total)

    def label(self, opt: str) -> str:
        """Option names in this provider's own terms."""
        text = OPTIONS[opt][0].replace("5-minute", self.rules.name_short).replace("1-hour", self.rules.name_long)
        if self.rules.mode == "implicit":
            text = text.replace("Caching off", "Without caching (reference only)").replace("Prompt fix + breakpoints", "Prompt fix")
        return text

    def read(self, opt: str) -> float:
        u = self.usage[opt]
        return u.cache_read_input_tokens / max(1, u.total)

    def dollars(self, opt: str) -> float | None:
        return None if self.price is None else _units(self.usage[opt], self.rules) * self.price / 1e6

    @property
    def as_sent(self) -> str:
        """The option that models the requests as they were logged."""
        if "logged" in self.usage:
            return "logged"
        return "auto_5m" if self.rules.mode == "implicit" else "off"

    @property
    def measured_cost(self) -> float | None:
        m = self.measured
        return None if m is None or not m.total else _units(m, self.rules) / m.total


def _recommend(cost: dict[str, float], r: Rules | None = None) -> tuple[str, str]:
    if r is not None and r.mode == "implicit":
        if "auto_1h" in cost and cost["auto_1h"] < cost["auto_5m"] - 0.02:
            return "auto_1h", "longer retention has no write fee and more of the prompt is still cached after pauses"
        return "auto_5m", "caching is automatic on this model; nothing to configure"
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
    return pick, "within 2 points of the best option and simpler to set up"


def _layout(reusable: float, cost: dict[str, float], breaks: list[Break], fixable: bool) -> str:
    head = [b for b in breaks if b.path.startswith(("system", "tools"))]
    if breaks and len(head) * 2 >= len(breaks) and (fixable or reusable < 0.15):
        return "changing content at the top of the prompt"
    if reusable < 0.15:
        return "little repeats between requests"
    auto = cost.get("auto_5m", cost.get("plan_5m", 1.0))
    plan = cost.get("plan_5m", auto)
    if auto <= plan + 0.03 and auto < 0.97:
        return "append-only conversation"
    if any(b.kind == "inserted" for b in breaks) or plan < auto - 0.03:
        return "content inserted or changed mid-prompt"
    return "mixed"


def examine(rows: list[dict], k: Knowledge | None = None, min_tokens: int | None = None,
            model: str | None = None, price: float | None = None) -> list[Verdict]:
    """One verdict per application. If k is None the plan is learned from the same rows."""
    in_sample = k is None
    k = k or learn_from(rows, min_tokens, model)
    _fill(k, rows, min_tokens, model)
    apps = by_app(rows)
    options = [o for o in OPTIONS if o != "logged"]
    if any(any(ttl for _, ttl in prefix_blocks(r["request"])) for r in rows):
        options.append("logged")
    usage = simulate(rows, k, options)
    out = []
    for app, rs in apps.items():
        u = dict(usage[app])
        s, cal = k.specs[app], k.calibration[app]
        fixable = has_fix(rs, k)
        if not fixable:
            u.pop("fix_5m", None), u.pop("fix_1h", None)
        base = max(1, u["off"].total)
        cost = {o: _units(x, s) / base for o, x in u.items()}
        pick, reason = _recommend(cost, s)
        fix = None
        if fixable:
            best_fix = min((o for o in ("fix_5m", "fix_1h") if o in cost), key=lambda o: cost[o])
            if cost[best_fix] < cost[pick] - 0.05:
                fix = best_fix
        gaps = [b["ts"] - a["ts"] for a, b in zip(rs, rs[1:])]
        breaks = find_breaks(app, rs)
        reusable = reusable_share(rs, s.ttl_short)
        if s.mode == "implicit":                           # these models reuse any shared prefix, not whole blocks
            sim, seen = ImplicitSim(s.ttl_short, 1), Usage()
            for r in rs:
                seen.add(sim.request(strip_cache_control(r["request"]), r["ts"], 0, cal.overhead, cal.scale))
            reusable = seen.cache_read_input_tokens / max(1, seen.total)
        sizes = sorted(round(_est(r["request"]) * cal.scale) + (cal.overhead if r["request"].get("tools") else 0)
                       for r in rs)
        measured = None
        if all(real_usage(r) for r in rs):
            measured = Usage()
            for r in rs:
                measured.add(real_usage(r))
        out.append(Verdict(
            app, len(rs), u["off"].total, _layout(reusable, cost, breaks, fixable), reusable,
            statistics.median(gaps) if gaps else 0.0,
            sum(g > s.ttl_short for g in gaps) / len(gaps) if gaps else 0.0,
            u, pick, reason, fix, k.plans.get(app) or {}, breaks, in_sample,
            k.floors[app], bool(min_tokens) or s.known, sizes[len(sizes) // 2], s.read_x,
            price if price is not None else s.input_price, cal, measured, s))
    return sorted(out, key=lambda v: -v.tokens)
