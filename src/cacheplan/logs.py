"""Read request logs.

Accepted line formats (JSONL, one request per line, formats may be mixed):
  cacheplan capture   {"ts": ..., "app": ..., "request": {tools, system, messages}}
  raw request body      {"model": ..., "system": ..., "messages": [...]}
  promptcachelint trace {"provider": "anthropic", "body": {...}, "at": ..., "usage": {...}}

Only Anthropic Messages API requests are analysed in this release; other lines are
counted and skipped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .messages import system_text


@dataclass
class Log:
    rows: list[dict] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)


def _row(d: dict, default_app: str, n: int) -> dict | str:
    if not isinstance(d, dict):
        return "not a JSON object"
    usage = None
    if isinstance(d.get("request"), dict):
        req, ts, app, session, usage = d["request"], d.get("ts"), d.get("app"), d.get("session"), d.get("usage")
    elif isinstance(d.get("body"), dict):                       # promptcachelint trace
        if d.get("provider") not in (None, "anthropic"):
            return f"provider {d.get('provider')} not supported yet"
        req, ts, app, session, usage = d["body"], d.get("at"), None, d.get("session_id"), d.get("usage")
    elif "messages" in d:
        req, ts, app, session = d, None, None, None
    else:
        return "no request found"
    if not isinstance(req.get("messages"), list) or "input" in req:
        return "not an Anthropic Messages request"
    return {"ts": float(ts) if ts else float(n), "app": app or default_app, "session": session,
            "request": req, "usage": usage}


def _lines(text: str) -> set[str]:
    return {l for l in text.split("\n") if len(l.strip()) >= 12}


def load(paths) -> Log:
    """Load one or more logs. Requests under one app label are split further when their
    system prompts are different prompts altogether (an agent's main prompt and its
    commit-message prompt, say), so that each group is one prompt template."""
    log, skipped = Log(), defaultdict(int)
    for src, path in enumerate([paths] if isinstance(paths, (str, Path)) else list(paths)):
        p = Path(path)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = _row(json.loads(line), p.stem, n)
            except json.JSONDecodeError:
                row = "invalid JSON"
            if isinstance(row, str):
                skipped[row] += 1
                continue
            row["_src"] = src
            log.rows.append(row)
    log.rows.sort(key=lambda r: (r["_src"], r["ts"]))
    clusters: dict[str, list[set]] = defaultdict(list)
    for r in log.rows:
        ls = _lines(system_text(r["request"]))
        reps = clusters[r["app"]]
        for k, rep in enumerate(reps):
            if (not ls and not rep) or len(ls & rep) >= 0.5 * max(len(ls), len(rep), 1):
                break
        else:
            reps.append(ls)
            k = len(reps) - 1
        r["_cluster"] = k
    for r in log.rows:
        k = r.pop("_cluster")
        if len(clusters[r["app"]]) > 1:
            first = system_text(r["request"]).strip().split("\n")[0][:40]
            r["app"] = f'{r["app"]}#{k + 1} "{first}"'
    log.skipped = dict(skipped)
    return log


def by_app(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["app"]].append(r)
    return dict(out)


def conversation_key(row: dict) -> str:
    msgs = row["request"].get("messages") or [{}]
    return row.get("session") or json.dumps(msgs[0], sort_keys=True)


def real_usage(row: dict):
    """The provider's own token counts for a logged request, if the log has them.
    Accepts Anthropic usage objects and promptcachelint's field names."""
    from .sim import Usage
    u = row.get("usage")
    if not isinstance(u, dict) or u.get("input_tokens") is None:
        return None
    creation = u.get("cache_creation_input_tokens", u.get("cache_write")) or 0
    read = u.get("cache_read_input_tokens", u.get("cache_read")) or 0
    one_hour = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
    return Usage(int(u["input_tokens"]), int(creation), int(read), int(one_hour))
