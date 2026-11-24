"""Send a small set of Anthropic requests through `cacheplan capture`, with or without a plan.

Three short support conversations that share one long system prompt: 15 requests, about
60,000 input tokens, a few hundred output tokens. Used to check the plan wrapper live:

    # 1. as the application sends them (no caching), behind the capture proxy
    python benchmarks/providers/anthropic_driver.py
    # 2. after `cacheplan plan baseline.jsonl --save plan.json`, the same requests through the wrapper
    python benchmarks/providers/anthropic_driver.py --plan plan.json

Reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the environment. Requires httpx.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_driver import QUESTIONS, RULES  # noqa: E402  (the same scenario as the OpenAI driver)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--plan", help="a file from `cacheplan plan --save`; requests go through PlanTransport")
    ap.add_argument("--app", help="application name in the plan file, when it has more than one")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    a = ap.parse_args()
    key, base = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY in this window first.")
    if "127.0.0.1" not in base and "localhost" not in base:
        print(f"Note: ANTHROPIC_BASE_URL is {base}, so nothing is being captured.")
    transport = httpx.HTTPTransport()
    if a.plan:
        from cacheplan.apply import PlanTransport
        transport = PlanTransport(a.plan, app=a.app, inner=transport)
        print(f"using the plan in {a.plan}: {transport.entry['action']}, {transport.entry['lifetime']}")
    client = httpx.Client(transport=transport, timeout=120)
    n = 0
    for conversation in QUESTIONS:
        history = []
        for question in conversation:
            history.append({"role": "user", "content": question})
            r = client.post(base.rstrip("/") + "/v1/messages",
                            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                            json={"model": a.model, "max_tokens": 200, "system": RULES, "messages": history})
            if r.status_code != 200:
                sys.exit(f"The API answered {r.status_code}: {r.text[:400]}")
            data = r.json()
            answer = " ".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text").strip() or "(no text)"
            history.append({"role": "assistant", "content": answer})
            u = data.get("usage") or {}
            n += 1
            print(f"{n:>2}. uncached {u.get('input_tokens', 0):>6}, written {u.get('cache_creation_input_tokens', 0):>6}, "
                  f"read {u.get('cache_read_input_tokens', 0):>6}  {answer[:50]!r}")
            time.sleep(a.pause)
    print(f"done: {n} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
