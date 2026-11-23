"""Text and JSON output for the report."""
from __future__ import annotations

from .findings import Finding
from .forecast import OPTIONS, Verdict
from .messages import BreakPlan


def plan_words(p: BreakPlan) -> list[str]:
    out = []
    if p.head:
        out.append("the end of the system prompt (covers tools and system)")
    if p.const_prefix:
        out.append(f"message block {p.const_prefix} (the end of the part that never changes)")
    for o in p.offsets:
        out.append("the last message block" if o == 0 else
                   f"{o} block{'s' if o != 1 else ''} before the last message block")
    return out


def _advice(v: Verdict) -> list[str]:
    label = OPTIONS[v.recommended][0]
    lines = []
    if v.recommended == "off":
        lines.append(f"Recommendation: leave caching off as the prompts stand ({v.reason}).")
    elif v.recommended == "logged":
        lines.append(f"Recommendation: keep your current settings - forecast {v.cost('logged'):.0%} "
                     f"of the uncached input cost ({v.reason}).")
    else:
        lines.append(f"Recommendation: {label[0].lower() + label[1:]} "
                     f"- forecast {v.cost(v.recommended):.0%} of the uncached input cost ({v.reason}).")
        if v.recommended.startswith("auto"):
            ttl = ', "ttl": "1h"' if v.recommended.endswith("1h") else ""
            lines.append(f'  How: add  "cache_control": {{"type": "ephemeral"{ttl}}}  at the top level of each request.')
        else:
            where = plan_words(v.plan)
            ttl = " with the 1-hour lifetime" if v.recommended.endswith("1h") else ""
            lines.append(f"  How: put cache_control breakpoints{ttl} at " + "; ".join(where) + ".")
    if v.fix_option:
        lines.append(f"With the prompt fix in the findings: {v.cost(v.fix_option):.0%} "
                     f"({OPTIONS[v.fix_option][0].lower()}). It changes what the model reads, "
                     "so compare answer quality before adopting it.")
    return lines


def text(v: Verdict, fs: list[Finding], name_width: int = 36) -> str:
    out = [f"{v.app}",
           f"  {v.requests} requests, ~{v.tokens:,} input tokens (estimate)",
           f"  Layout     {v.layout}",
           f"  Reusable   {v.reusable:.0%} of input tokens repeat an earlier prefix as written",
           f"  Pace       median {v.median_gap:.0f} s between requests; {v.gaps_over_5m:.0%} of pauses over 5 minutes",
           "",
           f"  {'Option':<{name_width}} {'Read':>5}  {'Cost vs no caching':>18}"]
    for opt in OPTIONS:
        if opt in v.usage:
            mark = "  <- recommended" if opt == v.recommended else ""
            out.append(f"  {OPTIONS[opt][0]:<{name_width}} {v.read(opt):>5.0%}  {v.cost(opt):>18.0%}{mark}")
    out += [""] + ["  " + l for l in _advice(v)]
    if v.in_sample and v.recommended.startswith(("plan", "fix")) or (v.in_sample and v.fix_option):
        out.append("  Note: the plan was learned from this same log; use --learn-from for an out-of-sample forecast.")
    if fs:
        out += ["", "  Findings"] + ["    " + str(f) for f in fs]
    return "\n".join(out)


def as_dict(v: Verdict, fs: list[Finding]) -> dict:
    return {
        "app": v.app, "requests": v.requests, "input_tokens_estimate": v.tokens, "layout": v.layout,
        "reusable_share": round(v.reusable, 4), "median_gap_seconds": round(v.median_gap, 1),
        "gaps_over_5m": round(v.gaps_over_5m, 4),
        "options": {o: {"label": OPTIONS[o][0], "read_share": round(v.read(o), 4),
                        "cost_vs_no_caching": round(v.cost(o), 4)} for o in OPTIONS if o in v.usage},
        "recommended": v.recommended, "reason": v.reason, "prompt_fix_option": v.fix_option,
        "plan": {"head": v.plan.head, "constant_prefix_blocks": v.plan.const_prefix,
                 "offsets_from_end": list(v.plan.offsets)},
        "plan_learned_in_sample": v.in_sample,
        "findings": [{"rule": f.rule, "message": f.message, "fix": f.fix, "tokens": f.tokens} for f in fs],
    }
