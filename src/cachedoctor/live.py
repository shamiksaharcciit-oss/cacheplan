"""Verify a forecast against the real API: replay the logged requests under each option
with max_tokens=1 and read the provider's own cache counters."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict

from .forecast import Knowledge, build
from .sim import Usage

API_URL = "https://api.anthropic.com/v1/messages"


def isolate(req: dict, nonce: str) -> dict:
    """Make this run's prefixes unique so options cannot read each other's cache."""
    tag = f"[run {nonce}]"
    for t in req.get("tools") or []:
        t["description"] = (t.get("description", "") + " " + tag).strip()
    s = req.get("system")
    if isinstance(s, str):
        req["system"] = tag + "\n" + s
    elif s:
        s[0] = dict(s[0], text=tag + "\n" + s[0].get("text", ""))
    else:
        req["system"] = tag
    return req


def _post(body: dict, key: str, url: str) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(6):
        rq = urllib.request.Request(url, data=data, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"API error {e.code}: {e.read().decode()[:400]}")
    raise SystemExit("gave up after retries")


def run(rows: list[dict], k: Knowledge, options: list[str], model: str, progress=print,
        url: str = API_URL) -> dict:
    """{app: {option: Usage}} measured live. 'off' is computed from the first option's totals."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set in this shell.")
    out: dict[str, dict[str, Usage]] = defaultdict(dict)
    for opt in [o for o in options if o != "off"]:
        nonce = uuid.uuid4().hex[:10]
        for i, row in enumerate(rows, 1):
            body = isolate(json.loads(json.dumps(build(opt, row, k))), nonce)
            body.update(model=model, max_tokens=1)
            u = _post(body, key, url).get("usage", {})
            creation = u.get("cache_creation_input_tokens") or 0
            one_hour = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens")
            if one_hour is None:
                one_hour = creation if opt.endswith("_1h") else 0
            out[row["app"]].setdefault(opt, Usage()).add(
                Usage(u.get("input_tokens", 0), creation, u.get("cache_read_input_tokens") or 0, one_hour))
            if i % 25 == 0:
                progress(f"  {opt}: {i}/{len(rows)}")
    for app, per in out.items():
        per["off"] = Usage(input_tokens=next(iter(per.values())).total)
    return dict(out)
