"""How long cacheplan takes, and how much memory it needs, as the log grows.

    python benchmarks/scale/scale.py                 # 1,000 / 10,000 / 100,000 requests
    python benchmarks/scale/scale.py 1000 5000       # or the sizes you name

Each size is run in a fresh process. The log is written from the test corpus: three scripted
applications with tools, long system prompts and growing conversations (about 3,500 input
tokens per request). Nothing is sent anywhere.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import json, resource, sys, time
sys.path[:0] = [sys.argv[2] + "/src", sys.argv[2]]
from cacheplan.findings import findings
from cacheplan.forecast import examine, learn_from
from cacheplan.logs import by_app, load
t = {}
a = time.perf_counter(); rows = load(sys.argv[1]).rows; t["load"] = time.perf_counter() - a
a = time.perf_counter(); k = learn_from(rows); t["learn and plan"] = time.perf_counter() - a
a = time.perf_counter(); verdicts = examine(rows, k); t["forecast and diagnose"] = time.perf_counter() - a
a = time.perf_counter(); apps = by_app(rows); n = sum(len(findings(v, apps[v.app], k)) for v in verdicts); t["findings"] = time.perf_counter() - a
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"requests": len(rows), "apps": len(verdicts), "seconds": t,
                  "peak_mb": peak / (1024 if sys.platform != "darwin" else 1024 * 1024)}))
'''


def write_log(path: Path, n: int) -> int:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from tests.corpus.apps import generate
    done = 0
    with open(path, "w", encoding="utf-8") as f:
        seed = 0
        while done < n:
            for row in generate(400, seed=seed):
                if done == n:
                    break
                row = dict(row, ts=row["ts"] + seed * 100000.0)
                f.write(json.dumps(row) + "\n")
                done += 1
            seed += 1
    return path.stat().st_size


def main() -> int:
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 10000, 100000]
    print("| Requests | Log size | Load | Learn and plan | Forecast and diagnose | Findings | Total | Peak memory |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    with tempfile.TemporaryDirectory() as d:
        for n in sizes:
            log = Path(d) / f"{n}.jsonl"
            size = write_log(log, n)
            started = time.perf_counter()
            out = subprocess.run([sys.executable, "-c", CHILD, str(log), str(ROOT)], capture_output=True, text=True)
            if out.returncode:
                print(f"| {n:,} | {size / 1e6:.0f} MB | failed after {time.perf_counter() - started:.0f} s: {out.stderr.strip().splitlines()[-1]} |")
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            s = r["seconds"]
            print(f"| {r['requests']:,} | {size / 1e6:.0f} MB | {s['load']:.1f} s | {s['learn and plan']:.1f} s | "
                  f"{s['forecast and diagnose']:.1f} s | {s['findings']:.1f} s | {sum(s.values()):.1f} s | {r['peak_mb']:,.0f} MB |", flush=True)
            log.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
