"""cachedoctor command line."""
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
from .messages import est_tokens, prefix_blocks
from .render import as_dict, text


def _load(args) -> tuple[list[dict], Knowledge | None]:
    """Rows to analyse, and knowledge learned from another log when --learn-from is given."""
    learn_path = getattr(args, "learn_from", None)
    log = load([learn_path, *args.logs] if learn_path else args.logs)
    for why, n in log.skipped.items():
        print(f"skipped {n} line(s): {why}", file=sys.stderr)
    if not log.rows:
        raise SystemExit("No Anthropic Messages requests found in the log.")
    if learn_path:
        train = [r for r in log.rows if r["_src"] == 0]
        rows = [r for r in log.rows if r["_src"] > 0]
        return rows, learn_from(train)
    return log.rows, None


def cmd_report(a) -> int:
    rows, k = _load(a)
    verdicts = examine(rows, k, a.min_tokens)
    k = k or learn_from(rows)
    apps = by_app(rows)
    results = [(v, findings(v, apps[v.app], k, a.min_tokens, not a.no_content)) for v in verdicts]
    if a.json:
        print(json.dumps({"cachedoctor": __version__, "applications": [as_dict(v, f) for v, f in results]}, indent=1))
    else:
        print(f"cachedoctor {__version__}: {len(rows)} requests, {len(results)} application(s). "
              "Forecasts come from a local model of Anthropic caching; confirm with `cachedoctor verify`.\n")
        shown = [(v, f) for v, f in results if v.requests >= 3 or len(results) == 1]
        print("\n\n".join(text(v, f) for v, f in shown))
        if len(shown) < len(results):
            print(f"\n{len(results) - len(shown)} application(s) with fewer than 3 requests not shown (use --json for all).")
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
    for v in examine(rows, k, a.min_tokens):
        ok = v.reusable >= a.min_reusable
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {v.app}: {v.reusable:.0%} of input tokens reusable "
              f"(threshold {a.min_reusable:.0%})")
    return 1 if bad else 0


def cmd_gaps(a) -> int:
    rows, k = _load(a)
    k = k or learn_from(rows)
    opts = ["off", "auto_5m", "auto_1h", "plan_5m", "plan_1h"]
    t0 = rows[0]["ts"]
    print("Forecast cost vs no caching as the time between requests grows (local model)\n")
    print("| Pauses scaled by | Median pause | " + " | ".join(OPTIONS[o][0] for o in opts[1:]) + " |")
    print("| --- | --- |" + " --- |" * (len(opts) - 1))
    for scale in [float(x) for x in a.scales.split(",")]:
        scaled = [dict(r, ts=t0 + (r["ts"] - t0) * scale) for r in rows]
        res = simulate(scaled, k, opts, a.min_tokens)
        tot = {o: sum(per[o].cost_units() for per in res.values()) for o in opts}
        base = sum(per["off"].total for per in res.values())
        d = [y["ts"] - x["ts"] for x, y in zip(scaled, scaled[1:])] or [0]
        print(f"| {scale:g}x | {statistics.median(d):.0f} s | " + " | ".join(f"{tot[o] / base:.0%}" for o in opts[1:]) + " |")
    return 0


def cmd_verify(a) -> int:
    from . import live
    rows, k = _load(a)
    if a.limit:
        rows = rows[:a.limit]
    verdicts = examine(rows, k, a.min_tokens)
    k = k or learn_from(rows)
    wanted = ["auto_5m"] + sorted({v.recommended for v in verdicts} | {v.fix_option for v in verdicts if v.fix_option})
    if a.options:
        wanted = a.options.split(",")
    options = [o for o in dict.fromkeys(wanted) if o in OPTIONS and o != "off"]
    est = sum(est_tokens(t) for r in rows for t, _ in prefix_blocks(r["request"]))
    print(f"Live run: {len(rows)} requests x {len(options)} option(s) ({', '.join(options)}), "
          f"about {est * len(options):,} input tokens in total (estimate), max_tokens=1.")
    if not a.yes:
        print("Add --yes to send them.")
        return 2
    measured = live.run(rows, k, options, a.model)
    forecast = simulate(rows, k, ["off"] + options, a.min_tokens)
    print(f"\n{'Application':<34} {'Option':<36} {'Forecast':>9} {'Measured':>9} {'Read':>6}")
    for app, per in measured.items():
        base = per["off"].total
        fbase = max(1, forecast[app]["off"].total)
        for o in options:
            u = per[o]
            print(f"{app[:34]:<34} {OPTIONS[o][0]:<36} {forecast[app][o].cost_units() / fbase:>9.0%} "
                  f"{u.cost_units() / base:>9.0%} {u.cache_read_input_tokens / max(1, u.total):>6.0%}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({app: {o: vars(u) for o, u in per.items()} for app, per in measured.items()}, f, indent=1)
    return 0


def cmd_capture(a) -> int:
    from .proxy import serve
    print(f"Capturing on http://127.0.0.1:{a.port} -> {a.upstream}; writing {a.out}.\n"
          f"In the shell that runs your application set ANTHROPIC_BASE_URL=http://127.0.0.1:{a.port}\n"
          "Ctrl+C to stop.")
    try:
        serve(a.port, a.upstream, a.out, a.app).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_rules(_) -> int:
    for rule, what in RULES.items():
        print(f"{rule}  {what}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cachedoctor", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, learn=True):
        p.add_argument("logs", nargs="+", help="JSONL request log(s)")
        p.add_argument("--min-tokens", type=int, default=1024,
                       help="minimum cacheable prompt length for your model (default 1024)")
        if learn:
            p.add_argument("--learn-from", metavar="LOG",
                           help="learn the plan from this other log (out-of-sample forecast)")

    p = sub.add_parser("report", help="diagnosis, forecast and recommendation per application")
    common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-content", action="store_true", help="never print prompt text")
    p.add_argument("--fail-on", metavar="RULES", help="comma-separated rule ids that make the exit status 1")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("diagnose", help="where and why the reusable prefix breaks")
    common(p, learn=False)
    p.add_argument("-v", "--verbose", action="store_true", help="list every break")
    p.add_argument("--no-content", action="store_true")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("verify", help="replay under the recommended options against the live API")
    common(p)
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

    p = sub.add_parser("capture", help="logging proxy: record the requests your application sends")
    p.add_argument("--out", required=True)
    p.add_argument("--app", default="app", help="label for the application being captured")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--upstream", default="https://api.anthropic.com")
    p.set_defaults(fn=cmd_capture)

    sub.add_parser("rules", help="list rule ids").set_defaults(fn=cmd_rules)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
