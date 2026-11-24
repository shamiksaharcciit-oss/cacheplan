"""Read request logs.

Accepted line formats (JSONL, one request per line, formats may be mixed):
  cacheplan capture   {"ts": ..., "app": ..., "request": {tools, system, messages}}
  raw request body      {"model": ..., "system": ..., "messages": [...]}
  promptcachelint trace {"provider": "anthropic", "body": {...}, "at": ..., "usage": {...}}

Requests in OpenAI (Chat Completions, Responses), Bedrock Converse and Gemini
generateContent formats are recognised and converted; see providers.py for what is
forecast for each.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import readers
from .messages import share_blocks, system_text


@dataclass
class Log:
    rows: list[dict] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)


def _field(d: dict, raw_request: dict, path: str | None):
    """A value named by a dotted path, looked up in the log line and then in the request
    body (so `metadata.user_id` or `prompt_cache_key` work on raw request bodies)."""
    if not path:
        return None
    for start in (d, raw_request):
        cur = start
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is not None and not isinstance(cur, (dict, list)):
            return str(cur)
    return None


def _row(d: dict, default_app: str, n: int, app_field: str | None = None, session_field: str | None = None) -> dict | str:
    from .providers import normalise
    if not isinstance(d, dict):
        return "not a JSON object"
    usage, provider, model = None, None, None
    if isinstance(d.get("request"), dict):
        req, ts, app, session, usage = d["request"], d.get("ts"), d.get("app"), d.get("session"), d.get("usage")
        provider, model = d.get("provider"), d.get("model")
    elif isinstance(d.get("body"), dict):                       # promptcachelint trace
        req, ts, app, session, usage = d["body"], d.get("at"), None, d.get("session_id"), d.get("usage")
        provider, model = d.get("provider"), d.get("model")
    elif any(k in d for k in ("messages", "input", "contents")):
        req, ts, app, session = d, None, None, None
    else:
        return "no request found"
    raw = req
    try:
        if not d.get("normalised"):                      # lines from readers.py arrive converted
            provider, req = normalise(req, provider)
    except (AttributeError, TypeError, KeyError):
        return "request could not be read"
    if not isinstance(req.get("messages"), list):
        return "no messages in the request"
    if model and not req.get("model"):
        req = dict(req, model=model)
    req = share_blocks(req)
    app = _field(d, raw, app_field) or app
    session = _field(d, raw, session_field) or session
    return {"ts": float(ts) if ts else float(n), "app": app or default_app, "session": session,
            "request": req, "usage": usage, "provider": provider or "anthropic", "_rendered": None,
            "source": d.get("source") or "capture"}


def _lines(text: str) -> set[str]:
    return {l for l in text.split("\n") if len(l.strip()) >= 12}


def load(paths, app_field: str | None = None, session_field: str | None = None,
         split_prompts: bool = True, one_shot: bool = False) -> Log:
    """Load one or more logs. Requests under one app label are split further when their
    system prompts are different prompts altogether (an agent's main prompt and its
    commit-message prompt, say), so that each group is one prompt template.

    app_field / session_field: dotted paths naming where each line keeps its application
    label and conversation id. split_prompts=False keeps one application per label.
    one_shot=True says the requests are independent queries, not growing conversations, so a
    change in the first message counts as a prefix break."""
    log, skipped = Log(), defaultdict(int)
    for src, path in enumerate([paths] if isinstance(paths, (str, Path)) else list(paths)):
        p = Path(path)
        n = -1
        for record in readers.records(p.read_text(encoding="utf-8")):
            try:
                lines = [record] if isinstance(record, str) else readers.expand(record)
            except (AttributeError, TypeError, KeyError, ValueError):
                lines = "record could not be read"
            for line in ([lines] if isinstance(lines, str) else lines):
                n += 1
                row = line if isinstance(line, str) else _row(line, p.stem, n, app_field, session_field)
                if isinstance(row, str):
                    skipped[row] += 1
                    continue
                row["_src"] = src
                row["one_shot"] = one_shot
                if row["provider"] != "anthropic" and not row["app"].endswith(f"[{row['provider']}]"):
                    row["app"] = f"{row['app']} [{row['provider']}]"
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
        if split_prompts and len(clusters[r["app"]]) > 1:
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
    """The provider's own token counts for a logged request, if the log has them."""
    from .providers import usage_parts
    from .sim import Usage
    parts = usage_parts(row.get("usage"))
    if parts is None:
        return None
    return Usage(max(0, parts[0]), parts[1], parts[2], parts[3])
