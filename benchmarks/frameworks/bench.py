"""Framework benchmark: run the same conversations through each framework's default agent,
capture the requests it sends, and compare caching options.

    python bench.py capture                 # run every driver through the capture proxy
    python bench.py capture pydantic_ai     # or just some
    python bench.py capture --stand-in      # offline plumbing test, no API key, no cost
    python bench.py results                 # forecast table from the captures
    python bench.py verify --model MODEL    # measure automatic caching and the recommendation live

Captures go to captures/<framework>.jsonl; tables to results.md and results.json.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata as md
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
CAPTURES = HERE / "captures"
FRAMEWORKS = {      # driver -> (display name, distribution whose version is reported)
    "anthropic_sdk": ("Anthropic SDK tool loop", "anthropic"),
    "langchain_agent": ("LangChain create_agent", "langchain"),
    "langgraph_template": ("LangGraph ReAct template", "langgraph"),
    "pydantic_ai": ("Pydantic AI Agent", "pydantic-ai-slim"),
    "smolagents": ("smolagents ToolCallingAgent", "smolagents"),
    "crewai": ("CrewAI Agent", "crewai"),
}


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "not installed"


def capture(names: list[str], stand_in: bool, port: int) -> None:
    from cacheplan.proxy import serve
    CAPTURES.mkdir(exist_ok=True)
    upstream = "https://api.anthropic.com"
    if stand_in:
        from http.server import ThreadingHTTPServer
        sys.path.insert(0, str(HERE))
        from stand_in_server import Handler
        fake = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=fake.serve_forever, daemon=True).start()
        upstream = f"http://127.0.0.1:{fake.server_port}"
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in this shell.")
    if "langgraph_template" in names and not (HERE / "_vendor" / "react-agent").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/langchain-ai/react-agent",
                        str(HERE / "_vendor" / "react-agent")], check=True)
    for name in names:
        out = CAPTURES / f"{name}.jsonl"
        out.unlink(missing_ok=True)
        proxy = serve(port, upstream, str(out), name)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{proxy.server_port}"
        env = dict(os.environ, ANTHROPIC_BASE_URL=url, ANTHROPIC_API_BASE=url, ANTHROPIC_API_URL=url,
                   NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", PYTHONIOENCODING="utf-8")
        if stand_in:
            env["ANTHROPIC_API_KEY"] = "stand-in-key"
        code = f"import sys; sys.path.insert(0, {str(HERE)!r}); import drivers.use_{name} as d; d.main()"
        print(f"{name}: running ...", flush=True)
        try:
            r = subprocess.run([sys.executable, "-c", code], env=env, cwd=HERE, capture_output=True,
                               text=True, timeout=1200, encoding="utf-8", errors="replace")
            n = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
            if r.returncode:
                print(f"{name}: FAILED after {n} requests\n" + "\n".join(r.stderr.strip().splitlines()[-6:]))
            else:
                print(f"{name}: captured {n} requests")
        except subprocess.TimeoutExpired:
            print(f"{name}: timed out")
        proxy.shutdown()
        proxy.server_close()


def _examine(name: str):
    from cacheplan.findings import findings
    from cacheplan.forecast import examine, learn_from
    from cacheplan.logs import by_app, load
    rows = load(str(CAPTURES / f"{name}.jsonl")).rows
    verdicts = examine(rows)
    v = max(verdicts, key=lambda x: x.tokens)                 # the framework's main prompt
    k = learn_from(rows)
    return rows, k, v, findings(v, by_app(rows)[v.app], k)


def results(measured: dict | None = None) -> None:
    from cacheplan.forecast import OPTIONS
    table, data = [], {}
    head = "| Framework | Version | Requests | Reusable as written | As shipped | Automatic caching | Recommended | Cost | With prompt fix | Findings |"
    if measured:
        head = head.replace("| Automatic caching |", "| Automatic caching (measured) |").replace("| Cost |", "| Cost (measured) |")
    table += [head, "|" + " --- |" * (head.count("|") - 1)]
    for name, (label, dist) in FRAMEWORKS.items():
        if not (CAPTURES / f"{name}.jsonl").exists():
            continue
        rows, k, v, fs = _examine(name)
        m = (measured or {}).get(name, {})
        auto = m.get("auto_5m", v.cost("auto_5m"))
        rec_cost = 1.0 if v.recommended == "off" else m.get(v.recommended, v.cost(v.recommended))
        fix = m.get(v.fix_option, v.cost(v.fix_option)) if v.fix_option else None
        rules = ", ".join(sorted({f.rule for f in fs})) or "none"
        shipped = f"{m.get('logged', v.cost('logged')):.0%} (caches by default)" if "logged" in v.usage else "100% (caching off)"
        table.append(f"| {label} | {version(dist)} | {len(rows)} | {v.reusable:.0%} | {shipped} | {auto:.0%} | "
                     f"{OPTIONS[v.recommended][0]} | {rec_cost:.0%} | {f'{fix:.0%}' if fix else '-'} | {rules} |")
        data[name] = {"label": label, "version": version(dist), "requests": len(rows), "layout": v.layout,
                      "reusable": round(v.reusable, 4), "recommended": v.recommended,
                      "forecast": {o: round(v.cost(o), 4) for o in v.usage}, "measured": m,
                      "prompt_fix_option": v.fix_option, "findings": [f"{f.rule} {f.message}" for f in fs]}
    text = "\n".join(table)
    print(text)
    (HERE / "results.md").write_text(text + "\n", encoding="utf-8")
    readme = HERE / "README.md"
    doc = readme.read_text(encoding="utf-8")
    if START in doc and END in doc and measured:             # only measured results go into the README
        head, rest = doc.split(START, 1)
        stamp = f"Generated by `bench.py verify` on {datetime.date.today().isoformat()}; costs are measured on the live API."
        readme.write_text(head + START + "\n" + stamp + "\n\n" + text + "\n" + END + rest.split(END, 1)[1], encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


def verify(model: str, names: list[str]) -> None:
    from cacheplan import live
    measured = {}
    for name in names:
        if not (CAPTURES / f"{name}.jsonl").exists():
            continue
        rows, k, v, _ = _examine(name)
        rows = [r for r in rows if r["app"] == v.app]
        opts = [o for o in dict.fromkeys(["auto_5m", v.recommended, v.fix_option,
                                          "logged" if "logged" in v.usage else None]) if o and o != "off"]
        print(f"{name}: {len(rows)} requests x {len(opts)} option(s) ({', '.join(opts)})", flush=True)
        res = live.run(rows, k, opts, model, progress=lambda *_: None)[v.app]
        base = res["off"].total
        measured[name] = {o: round(res[o].cost_units(v.read_x) / base, 4) for o in opts}
        (HERE / "measured.json").write_text(json.dumps(measured, indent=1), encoding="utf-8")
    results(measured)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("names", nargs="*")
    c.add_argument("--stand-in", action="store_true")
    c.add_argument("--port", type=int, default=0)
    sub.add_parser("results")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="*")
    v.add_argument("--model", required=True)
    a = ap.parse_args()
    names = getattr(a, "names", None) or list(FRAMEWORKS)
    unknown = [n for n in names if n not in FRAMEWORKS]
    if unknown:
        raise SystemExit(f"unknown framework(s): {', '.join(unknown)}; choose from {', '.join(FRAMEWORKS)}")
    if a.cmd == "capture":
        capture([n for n in names if (HERE / "drivers" / f"use_{n}.py").exists()], a.stand_in, a.port)
    elif a.cmd == "results":
        results(json.loads((HERE / "measured.json").read_text()) if (HERE / "measured.json").exists() else None)
    else:
        verify(a.model, names)
