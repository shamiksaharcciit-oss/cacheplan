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


def _how(v: Verdict) -> str:
    """How to apply the recommended option, in the provider's own request terms."""
    long = v.recommended.endswith("1h")
    provider = v.rules.provider
    if v.recommended.startswith("auto"):
        if provider == "openai":
            return ("How: this is the default on this model; nothing to set." if not long else
                    'How: set  "prompt_cache_options": {"mode": "implicit", "ttl": "12h"}  on each request (Responses API).')
        if provider == "bedrock":
            return ('How: pass  additionalModelRequestFields={"cache_control": {"type": "ephemeral"'
                    + (', "ttl": "1h"' if long else "") + "}}  to Converse (Claude models that support it).")
        return 'How: add  "cache_control": {"type": "ephemeral"' + (', "ttl": "1h"' if long else "") + "}  at the top level of each request."
    marker = {"openai": '"prompt_cache_options": {"mode": "explicit"' + (', "ttl": "12h"' if long else "")
                        + '} and a "prompt_cache_breakpoint": {} on the content part',
              "bedrock": '{"cachePoint": {"type": "default"' + (', "ttl": "1h"' if long else "") + "}} blocks"}.get(
                  provider, "cache_control breakpoints")
    life = f" with the {v.rules.name_long} lifetime" if long and provider == "anthropic" else ""
    return f"How: put {marker}{life} at " + "; ".join(plan_words(v.plan)) + "."


def _advice(v: Verdict) -> list[str]:
    lines = []
    if not v.rules.forecast:
        return ["No cost forecast for this provider: " + v.rules.note + "."]
    if v.rules.mode == "implicit":
        lines.append(f"Caching is automatic on this model: forecast {v.cost('auto_5m'):.0%} of the price without it"
                     + (" (an estimate under the assumptions in the note below). " if v.rules.provider == "gemini" else ". ")
                     + "There is nothing to switch on; what you control is how much of the prompt repeats.")
        if v.recommended == "auto_1h":
            lines.append(f'Recommendation: set  "prompt_cache_retention": "24h"  - forecast {v.cost("auto_1h"):.0%} ({v.reason}).')
    elif v.recommended == "off":
        lines.append(f"Recommendation: leave caching off as the prompts stand ({v.reason}).")
        if v.rules.provider == "openai" and "logged" in v.usage and v.cost("logged") > 1.03:
            lines.append(f"  Caching is on by default on this model and these requests pay {v.cost('logged'):.0%}. "
                         'Per OpenAI\'s documentation,  "prompt_cache_options": {"mode": "explicit"}  with no breakpoints '
                         "stops the automatic writes; this project has not tested that setting.")
    elif v.recommended == "logged":
        lines.append(f"Recommendation: keep your current settings - forecast {v.cost('logged'):.0%} "
                     f"of the uncached input cost ({v.reason}).")
    else:
        label = v.label(v.recommended)
        lines.append(f"Recommendation: {label[0].lower() + label[1:]} "
                     f"- forecast {v.cost(v.recommended):.0%} of the uncached input cost ({v.reason}).")
        lines.append("  " + _how(v))
    if v.fix_option:
        lines.append(f"With the prompt fix in the findings: {v.cost(v.fix_option):.0%} "
                     f"({v.label(v.fix_option).lower()}). It changes what the model reads, "
                     "so compare answer quality before adopting it.")
    if v.source != "capture":
        lines.append(f"Note: read from a {v.source} export. A tracer records the messages its integration saw, which "
                     "may differ from what the provider received (tools, cache markers, order), so treat the "
                     "costs as rougher than from a capture.")
    if v.rules.note and v.rules.provider != "anthropic":
        lines.append(f"Note ({v.rules.provider}): {v.rules.note}.")
    if v.rules.provider != "anthropic" and v.rules.forecast:
        lines.append(f"Evidence for these rules: {v.rules.evidence}.")
    return lines


def _money(x: float | None) -> str:
    return "" if x is None else (f"${x:,.2f}" if x >= 1 else f"${x:.4f}")


def _token_source(v: Verdict) -> str:
    c = v.calibration
    if c.responses:
        return f"calibrated on {c.responses} recorded responses"
    return "estimated from characters; record responses for real counts"


def _check(v: Verdict) -> str | None:
    """Compare the forecast for the logged settings with what the provider actually reported."""
    if v.measured is None or not v.rules.forecast:
        return None
    opt = v.as_sent
    if opt == "off" and v.measured.cache_creation_input_tokens + v.measured.cache_read_input_tokens:
        return None
    return (f"Forecast check: this log's own settings were forecast at {v.cost(opt):.0%} and the recorded "
            f"responses show {v.measured_cost:.0%} (read {v.measured.cache_read_input_tokens / max(1, v.measured.total):.0%})."
            + (" The first request already read from the cache, so it was warm from traffic before this log; "
               "the forecast starts cold and will read high." if v.warm_start else ""))


def text(v: Verdict, fs: list[Finding], name_width: int = 36) -> str:
    money = v.price is not None
    out = [f"{v.app}",
           f"  {v.requests} requests, ~{v.tokens:,} input tokens ({_token_source(v)})"
           + ("" if v.rules.provider == "anthropic" else f"; provider: {v.rules.provider}"),
           f"  Layout     {v.layout}",
           f"  Reusable   {v.reusable:.0%} of input tokens repeat an earlier prefix as written",
           f"  Pace       median {v.median_gap:.0f} s between requests; {v.gaps_over_5m:.0%} of pauses over {v.rules.ttl_short // 60} minutes",
           f"  Minimum    {v.minimum:,} tokens to be cacheable"
           + ("" if v.minimum_known else " (model not recognised: default used, set --model or --min-tokens)")
           + f"; typical prompt ~{v.typical_tokens:,}",
           "",
           f"  {'Option':<{name_width}} {'Read':>5}  {'Cost vs no caching':>18}" + (f"  {'This log':>10}" if money else "")]
    if not v.rules.forecast:
        out[-2:] = []                                      # no option table without a forecast
    for opt in OPTIONS:
        if opt in v.usage and v.rules.forecast:
            mark = "  <- recommended" if opt == v.recommended and (v.rules.mode != "implicit" or opt == "auto_1h") else ""
            cash = f"  {_money(v.dollars(opt)):>10}" if money else ""
            out.append(f"  {v.label(opt):<{name_width}} {v.read(opt):>5.0%}  {v.cost(opt):>18.0%}{cash}{mark}")
    if money and v.rules.forecast:
        out.append(f"  (input tokens at ${v.price:g} per million; cache reads at {v.read_x:g}x)")
    out += [""] + ["  " + l for l in _advice(v)]
    check = _check(v)
    if check:
        out.append("  " + check)
    if v.in_sample and (v.recommended.startswith(("plan", "fix")) or v.fix_option):
        out.append("  Note: the plan was chosen on this same log; use --learn-from for an out-of-sample forecast.")
    if not v.calibration.responses and 0.5 * v.minimum <= v.typical_tokens <= 2 * v.minimum:
        out.append("  Note: prompts are close to the minimum cacheable length, where token estimates decide "
                   "what is cached. Treat this forecast as rough and "
                   + ("run `cacheplan verify`." if v.rules.provider == "anthropic"
                      else "capture with usage recorded to see the forecast check."))
    if fs:
        out += ["", "  Findings"] + ["    " + str(f) for f in fs]
    return "\n".join(out)


def as_dict(v: Verdict, fs: list[Finding]) -> dict:
    return {
        "app": v.app, "requests": v.requests, "input_tokens_estimate": v.tokens, "layout": v.layout,
        "reusable_share": round(v.reusable, 4), "median_gap_seconds": round(v.median_gap, 1),
        "gaps_over_5m": round(v.gaps_over_5m, 4),
        "minimum_cacheable_tokens": v.minimum, "minimum_known": v.minimum_known,
        "typical_prompt_tokens_estimate": v.typical_tokens,
        "token_calibration": {"scale": round(v.calibration.scale, 4), "tool_overhead": v.calibration.overhead,
                              "responses": v.calibration.responses},
        "cache_read_multiplier": v.read_x, "input_price_per_million": v.price,
        "measured_cost_vs_no_caching": None if v.measured_cost is None else round(v.measured_cost, 4),
        "provider": v.rules.provider, "caching_mode": v.rules.mode, "log_source": v.source,
        "options": {o: {"label": v.label(o), "read_share": round(v.read(o), 4),
                        "cost_vs_no_caching": round(v.cost(o), 4),
                        "cost_usd": None if v.dollars(o) is None else round(v.dollars(o), 6)}
                    for o in OPTIONS if o in v.usage and v.rules.forecast},
        "forecast_available": v.rules.forecast, "rules_source": v.rules.source, "rules_evidence": v.rules.evidence, "provider_note": v.rules.note or None,
        "recommended": v.recommended, "reason": v.reason, "prompt_fix_option": v.fix_option,
        "plan": {"head": v.plan.head, "constant_prefix_blocks": v.plan.const_prefix,
                 "offsets_from_end": list(v.plan.offsets)},
        "plans": {str(ttl): {"head": p.head, "constant_prefix_blocks": p.const_prefix,
                             "offsets_from_end": list(p.offsets)} for ttl, p in v.plans.items()},
        "plan_learned_in_sample": v.in_sample,
        "findings": [{"rule": f.rule, "message": f.message, "fix": f.fix, "tokens": f.tokens} for f in fs],
    }


def html(results: list[tuple[Verdict, list[Finding]]], version: str) -> str:
    """A self-contained HTML report: no scripts, no external resources."""
    import html as h
    css = """
    :root{--bg:#fcfcfb;--fg:#0b0b0b;--sec:#52514e;--line:#e3e2dd;--bar:#2a78d6;--warn:#eb6834;--card:#fff}
    @media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--fg:#f0efec;--sec:#c3c2b7;--line:#3a3a37;--bar:#3987e5;--warn:#d95926;--card:#242422}}
    body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:920px;margin:32px auto;padding:0 16px}
    h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:0 0 6px} .sec{color:var(--sec)}
    .app{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:18px 0}
    table{border-collapse:collapse;width:100%;margin:12px 0} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:14px;vertical-align:middle}
    th{color:var(--sec);font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
    .track{position:relative;height:14px;min-width:160px;background:transparent;border-left:1px solid var(--line)}
    .fill{height:14px;border-radius:0 4px 4px 0;background:var(--bar)} .fill.over{background:var(--warn)}
    .ref{position:absolute;top:-3px;bottom:-3px;width:0;border-left:2px dashed var(--sec);left:74%}
    tr.rec td{font-weight:650} .pill{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin-left:6px;color:var(--sec)}
    .advice{border-left:4px solid var(--bar);padding:8px 12px;margin:12px 0;background:var(--bg)}
    code{font-family:ui-monospace,Consolas,monospace;font-size:13px} li{margin:6px 0}
    """
    parts = [f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>cacheplan report</title><style>{css}</style><h1>Prompt caching report</h1>"
             f"<p class='sec'>cacheplan {h.escape(version)} · forecasts come from a local model of the provider's caching "
             "rules. Cost is input cost relative to the same requests without caching; the dashed line marks 100%.</p>"]
    for v, fs in results:
        rows = []
        for opt in OPTIONS:
            if opt not in v.usage or not v.rules.forecast:
                continue
            c = v.cost(opt)
            width = min(100.0, c * 74)                      # 100% of cost sits at 74% of the track
            cash = f"<td class='n'>{_money(v.dollars(opt))}</td>" if v.price is not None else ""
            chosen = opt == v.recommended and (v.rules.mode != "implicit" or opt == "auto_1h")
            rec = " class='rec'" if chosen else ""
            pill = "<span class='pill'>recommended</span>" if chosen else ""
            rows.append(f"<tr{rec}><td>{h.escape(v.label(opt))}{pill}</td><td class='n'>{v.read(opt):.0%}</td>"
                        f"<td class='n'>{c:.0%}</td>{cash}<td><div class='track'><div class='fill{' over' if c > 1.005 else ''}' "
                        f"style='width:{width:.1f}%'></div><div class='ref'></div></div></td></tr>")
        cash_h = "<th class='n'>This log</th>" if v.price is not None else ""
        notes = [l for l in (_check(v),) if l]
        findings_html = "".join(f"<li><code>{f.rule}</code> {h.escape(f.message)}<br><span class='sec'>Fix: {h.escape(f.fix)}</span></li>"
                                for f in fs)
        parts.append(
            f"<section class='app'><h2>{h.escape(v.app)}</h2>"
            f"<div class='sec'>{v.requests} requests · ~{v.tokens:,} input tokens ({h.escape(_token_source(v))}) · "
            f"layout: {h.escape(v.layout)} · {v.reusable:.0%} reusable as written · median pause {v.median_gap:.0f} s</div>"
            + (f"<table><tr><th>Option</th><th class='n'>Read</th><th class='n'>Cost vs no caching</th>{cash_h}<th></th></tr>{''.join(rows)}</table>" if rows else "") +
            f"<div class='advice'>{'<br>'.join(h.escape(l) for l in _advice(v) + notes)}</div>"
            + (f"<h3>Findings</h3><ul>{findings_html}</ul>" if fs else "") + "</section>")
    return "".join(parts) + "</html>"
