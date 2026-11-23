"""Send a small, realistic set of OpenAI requests through `cacheplan capture`.

Three short support conversations that share one long system prompt and grow turn by turn:
15 requests, roughly 45,000 input tokens in total, a few hundred output tokens. Uses only
the standard library. Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.

    python benchmarks/providers/openai_driver.py --model gpt-5.4-mini
    python benchmarks/providers/openai_driver.py --model gpt-5.4-mini --stamp   # adds a timestamp to the system prompt
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

RULES = "You are a support agent for Northwind Outfitters. Answer in two sentences at most.\n" + "\n".join(
    f"Policy {i}: for orders in category {i}, refunds are allowed within {10 + i % 20} days, exchanges within "
    f"{20 + i % 15} days, and shipping to region {i % 9} takes {2 + i % 6} working days." for i in range(1, 91))
QUESTIONS = [
    ["Where is order 1042?", "Can I change the delivery address?", "What if nobody is home?",
     "Can I return one item only?", "How long does the refund take?"],
    ["I got the wrong size boots.", "Do I pay for return shipping?", "Can I exchange for another colour?",
     "Is the new pair sent before you get the old one?", "Can I get a discount for the trouble?"],
    ["Do you ship to Norway?", "What are the customs charges?", "How long does it take?",
     "Can I track it?", "What is the refund window for category 12?"],
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--stamp", action="store_true", help="put the current time in the system prompt")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    a = ap.parse_args()
    key, base = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not key:
        sys.exit("Set OPENAI_API_KEY in this window first.")
    if "127.0.0.1" not in base and "localhost" not in base:
        print(f"Note: OPENAI_BASE_URL is {base}, so nothing is being captured.")
    n = 0
    for conversation in QUESTIONS:
        history = []
        for question in conversation:
            system = RULES + (f"\nCurrent time: {time.strftime('%Y-%m-%dT%H:%M:%S')}" if a.stamp else "")
            history.append({"role": "user", "content": question})
            body = {"model": a.model, "max_completion_tokens": 300,
                    "messages": [{"role": "system", "content": system}, *history]}
            req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as e:
                sys.exit(f"The API answered {e.code}: {e.read().decode(errors='replace')[:400]}")
            answer = (data["choices"][0]["message"].get("content") or "").strip() or "(no text)"
            history.append({"role": "assistant", "content": answer})
            usage = data.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            n += 1
            print(f"{n:>2}. prompt {usage.get('prompt_tokens', '?'):>6} tokens, cached {cached:>6}  {answer[:60]!r}")
            time.sleep(a.pause)
    print(f"done: {n} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
