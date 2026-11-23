"""cacheplan command line."""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from . import __version__
from .diagnose import find_breaks, summarise
from .findings import RULES, findings
from .forecast import OPTIONS, Knowledge, examine, learn_from, simulate
from .logs import by_app, load
from .render import as_dict, html, text


def _load(args) -> tuple[list[dict], Knowledge | None]:
    """Rows to analyse, and knowledge learned from another log when --learn-from is given."""
    learn_path = getattr(args, "learn_from", None)
    log = load([learn_path, *args.logs] if learn_path else args.logs)
    for why, n in log.skipped.items():
        print(f"skipped {n} line(s): {why}", file=sys.stderr)
    if not log.rows:
        raise SystemExit("No model requests found in the log.")
    if learn_path:
        train = [r for r in log.rows if r["_src"] == 0]
        rows = [r for r in log.rows if r["_src"] > 0]
        return rows, learn_from(train, getattr(args, "min_tokens", None), getattr(args, "model", None))
    return log.rows, None


def cmd_report(a) -> int:
    rows, k = _load(a)
    k = k or learn_from(rows, a.min_tokens, a.model)
    verdicts = examine(rows, k, a.min_tokens, a.model, a.price)
    for v in verdicts:
        v.in_sample = not getattr(a, "learn_from", None)
    apps = by_app(rows)
    results = [(v, findings(v, apps[v.app], k, not a.no_content)) for v in verdicts]
    if a.json:
        print(json.dumps({"cacheplan": __version__, "applications": [as_dict(v, f) for v, f in results]}, indent=1))
    else:
        print(f"cacheplan {__version__}: {len(rows)} requests, {len(results)} application(s). "
              "Forecasts come from a local model of each provider's caching rules"
              + ("; confirm with `cacheplan verify`.\n" if all(v.rules.provider == "anthropic" for v in verdicts)
                 else ". Only the Anthropic API rules have been checked against live traffic.\n"))
        shown = [(v, f) for v, f in results if v.requests >= 3 or len(results) == 1]
        print("\n\n".join(text(v, f) for v, f in shown))
        if len(shown) < len(results):
            print(f"\n{len(results) - len(shown)} application(s) with fewer than 3 requests not shown (use --json for all).")
    if a.html:
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(html(results, __version__))
        print(f"\nwrote {a.html}", file=sys.stderr)
    fail = set(filter(None, (a.fail_on or "").split(",")))
    hit = sorted({f.rule for _, fs in results for f in fs} & fail)
    if hit:
        print("\nfailing on: " + ", ".join(hit), file=sys.stderr)
        return 1
    return 0


def cmd_diagnose(a) -> int:
    rows, _ = _load(a)
    for app, rs in by_app(rows).items():
        breaks = find_breaks(app, rs)
        print(f"{app}: {len(rs)} requests, {len(breaks)} prefix break(s)")
        for g in summarise(breaks):
            cause = f" ({g['cause']})" if g["cause"] else ""
            print(f"  {g['count']:>3}x {g['kind']}{cause} at {g['where']}: ~{g['lost_tokens']:,} tokens re-sent")
        if a.verbose:
            for b in breaks:
                print(f"    request {b.request}: {b.kind} at {b.path}" + (f" +{b.offset}" if b.kind == "changed" else "")
                      + f", ~{b.lost_tokens:,} tokens behind it")
                if b.was and not a.no_content:
                    print(f"        was: {b.was}\n        now: {b.now}")
        print()
    return 0


def cmd_check(a) -> int:
    """For pipelines: fail when the reusable share of a log falls below a threshold."""
    rows, k = _load(a)
    bad = 0
    for v in examine(rows, k, a.min_tokens, a.model):
        ok = v.reusable >= a.min_reusable
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {v.app}: {v.reusable:.0%} of input tokens reusable "
              f"(threshold {a.min_reusable:.0%})")
    return 1 if bad else 0


def cmd_gaps(a) -> int:
    from .forecast import _fill
    rows, k = _load(a)
    k = k or learn_from(rows, a.min_tokens, a.model)
    _fill(k, rows, a.min_tokens, a.model)
    opts = ["off", "auto_5m", "auto_1h", "plan_5m", "plan_1h"]
    t0 = rows[0]["ts"]
    print("Forecast cost vs no caching as the time between requests grows (local model).\n"
          "The breakpoint plan is the one chosen for the pace in the log.\n")
    print("| Pauses scaled by | Median pause | " + " | ".join(OPTIONS[o][0] for o in opts[1:]) + " |")
    print("| --- | --- |" + " --- |" * (len(opts) - 1))
    for scale in [float(x) for x in a.scales.split(",")]:
        scaled = [dict(r, ts=t0 + (r["ts"] - t0) * scale) for r in rows]
        res = simulate(scaled, k, opts)
        tot = {o: sum(per[o].cost_units(k.specs[app].read_x) for app, per in res.items()) for o in opts}
        base = sum(per["off"].total for per in res.values())
        d = [y["ts"] - x["ts"] for x, y in zip(scaled, scaled[1:])] or [0]
        print(f"| {scale:g}x | {statistics.median(d):.0f} s | " + " | ".join(f"{tot[o] / base:.0%}" for o in opts[1:]) + " |")
    return 0


def _anthropic_only(rows: list[dict], command: str) -> list[dict]:
    """Live commands send requests to the Anthropic API, so only its logs can be replayed."""
    keep = [r for r in rows if r.get("provider", "anthropic") == "anthropic"]
    other = sorted({r["provider"] for r in rows} - {"anthropic"})
    if other:
        print(f"`cacheplan {command}` sends requests to the Anthropic API only; "
              f"leaving out {len(rows) - len(keep)} request(s) from: {', '.join(other)}.", file=sys.stderr)
    if not keep:
        raise SystemExit(f"Nothing to send: `cacheplan {command}` supports Anthropic API logs only. For other "
                         "providers, capture with usage recorded and read the forecast check in `cacheplan report`.")
    return keep


def cmd_verify(a) -> int:
    from . import live
    from .models import spec
    rows, k = _load(a)
    rows = _anthropic_only(rows, "verify")
    if a.limit:
        rows = rows[:a.limit]
    k = k or learn_from(rows, a.min_tokens, a.model)
    verdicts = examine(rows, k, a.min_tokens, a.model)
    wanted = ["auto_5m"] + sorted({v.recommended for v in verdicts} | {v.fix_option for v in verdicts if v.fix_option})
    if a.options:
        wanted = a.options.split(",")
    options = [o for o in dict.fromkeys(wanted) if o in OPTIONS and o != "off"]
    est = sum(v.tokens for v in verdicts)
    print(f"Live run: {len(rows)} requests x {len(options)} option(s) ({', '.join(options)}), "
          f"about {est * len(options):,} input tokens in total (estimate), max_tokens=1.")
    if not a.yes:
        print("Add --yes to send them.")
        return 2
    measured = live.run(rows, k, options, a.model)
    forecast = simulate(rows, k, ["off"] + options)
    read_x = spec(a.model).read_x
    print(f"\n{'Application':<34} {'Option':<36} {'Forecast':>9} {'Measured':>9} {'Read':>6}")
    for app, per in measured.items():
        base = per["off"].total
        fbase = max(1, forecast[app]["off"].total)
        for o in options:
            u = per[o]
            print(f"{app[:34]:<34} {OPTIONS[o][0]:<36} {forecast[app][o].cost_units(read_x) / fbase:>9.0%} "
                  f"{u.cost_units(read_x) / base:>9.0%} {u.cache_read_input_tokens / max(1, u.total):>6.0%}")
    print(f"\nCache reads are priced at {read_x:g}x an input token for this model.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({app: {o: vars(u) for o, u in per.items()} for app, per in measured.items()}, f, indent=1)
    return 0


def cmd_quality(a) -> int:
    from . import quality
    rows, k = _load(a)
    rows = _anthropic_only(rows, "quality")
    k = k or learn_from(rows, a.min_tokens, None)
    print(f"Quality check: up to {a.sample} requests per application, three calls each "
          f"(as written twice, then with the prompt fix), max_tokens={a.max_tokens}. This spends output tokens.")
    if not a.yes:
        print("Add --yes to send them.")
        return 2
    results = quality.run(rows, k, a.model, a.sample, a.max_tokens)
    print("\n" + quality.text(results))
    return 1 if any("review before adopting" in r.verdict for r in results) else 0


def cmd_baseline(a) -> int:
    """Save or compare a content-free summary of a log, for regression checks in pipelines."""
    rows, k = _load(a)
    current = {v.app: {"reusable": round(v.reusable, 4), "best_cost": round(min(v.cost(o) for o in v.usage), 4)}
               for v in examine(rows, k, a.min_tokens, a.model)}
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump({"cacheplan": __version__, "applications": current}, f, indent=1)
        print(f"saved baseline for {len(current)} application(s) to {a.save}")
        return 0
    with open(a.compare, encoding="utf-8") as f:
        base = json.load(f)["applications"]
    bad = 0
    for app, now in current.items():
        was = base.get(app)
        if was is None:
            print(f"new   {app}: {now['reusable']:.0%} reusable, best option {now['best_cost']:.0%}")
            continue
        worse = now["reusable"] < was["reusable"] - a.max_drop or now["best_cost"] > was["best_cost"] + a.max_drop
        bad += worse
        print(f"{'FAIL' if worse else 'ok  '} {app}: reusable {was['reusable']:.0%} -> {now['reusable']:.0%}, "
              f"best option {was['best_cost']:.0%} -> {now['best_cost']:.0%}")
    return 1 if bad else 0


def cmd_capture(a) -> int:
    from .proxy import BASE_URL_HINT, UPSTREAMS, serve
    if a.provider == "bedrock":
        raise SystemExit("Bedrock requests are signed for their host and cannot go through a proxy.\n"
                         "Record them inside your application instead:\n\n"
                         "    from cacheplan.record import record_bedrock\n"
                         '    client = record_bedrock(boto3.client("bedrock-runtime"), "requests.jsonl")')
    upstream = a.upstream or UPSTREAMS[a.provider]
    print(f"Capturing on http://127.0.0.1:{a.port} -> {upstream}; writing {a.out}.\n"
          f"In the shell that runs your application set {BASE_URL_HINT[a.provider].format(port=a.port)}\n"
          "Ctrl+C to stop.")
    try:
        serve(a.port, upstream, a.out, a.app).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_rules(_) -> int:
    for rule, what in RULES.items():
        print(f"{rule}  {what}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cacheplan", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, learn=True, model=True):
        p.add_argument("logs", nargs="+", help="JSONL request log(s)")
        p.add_argument("--min-tokens", type=int, default=None,
                       help="minimum cacheable prompt length; default: looked up from the model")
        if model:
            p.add_argument("--model", default=None,
                           help="model id, used for the minimum and prices when the log does not name one")
        if learn:
            p.add_argument("--learn-from", metavar="LOG",
                           help="learn the plan from this other log (out-of-sample forecast)")

    p = sub.add_parser("report", help="diagnosis, forecast and recommendation per application")
    common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-content", action="store_true", help="never print prompt text")
    p.add_argument("--fail-on", metavar="RULES", help="comma-separated rule ids that make the exit status 1")
    p.add_argument("--price", type=float, default=None,
                   help="USD per million input tokens; default: looked up from the model")
    p.add_argument("--html", metavar="FILE", help="also write a self-contained HTML report")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("diagnose", help="where and why the reusable prefix breaks")
    common(p, learn=False, model=False)
    p.add_argument("-v", "--verbose", action="store_true", help="list every break")
    p.add_argument("--no-content", action="store_true")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("verify", help="replay under the recommended options against the live API")
    common(p, model=False)
    p.add_argument("--model", required=True)
    p.add_argument("--options", help="comma-separated option keys: " + ",".join(o for o in OPTIONS if o != "off"))
    p.add_argument("--limit", type=int)
    p.add_argument("--yes", action="store_true", help="actually send the requests")
    p.add_argument("--out", help="write measured usage to this JSON file")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("gaps", help="how the forecast changes with longer pauses between requests")
    common(p)
    p.add_argument("--scales", default="1,2,5,10,20,50")
    p.set_defaults(fn=cmd_gaps)

    p = sub.add_parser("check", help="for pipelines: fail when too little of the prompt is reusable")
    common(p)
    p.add_argument("--min-reusable", type=float, default=0.5)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("quality", help="check whether the prompt fix changes the model's answers (live)")
    common(p, model=False)
    p.add_argument("--model", required=True)
    p.add_argument("--sample", type=int, default=12, help="requests per application to compare")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--yes", action="store_true", help="actually send the requests")
    p.set_defaults(fn=cmd_quality)

    p = sub.add_parser("baseline", help="save or compare a content-free summary, for regression checks")
    common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="FILE")
    g.add_argument("--compare", metavar="FILE")
    p.add_argument("--max-drop", type=float, default=0.05, help="allowed worsening, as a fraction (default 0.05)")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("capture", help="logging proxy: record the requests your application sends")
    p.add_argument("--out", required=True)
    p.add_argument("--app", default="app", help="label for the application being captured")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--provider", choices=["anthropic", "openai", "gemini", "bedrock"], default="anthropic")
    p.add_argument("--upstream", default=None, help="override the provider's API address")
    p.set_defaults(fn=cmd_capture)

    sub.add_parser("rules", help="list rule ids").set_defaults(fn=cmd_rules)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
