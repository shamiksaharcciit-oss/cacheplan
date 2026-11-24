"""Where and why the reusable prefix breaks between requests.

Each request is compared with the earlier request of the same application that shares
the longest prefix with it. If that earlier request had content beyond the shared
prefix, the prefix broke there, and everything behind the break has to be paid for again.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from .messages import rendered

WINDOW = 20
_CAUSES = [
    ("timestamp", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("uuid", re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")),
    ("date", re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")),
    ("time of day", re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")),
    ("random id", re.compile(r"\b[0-9a-f]{12,}\b|\b[A-Za-z0-9_-]{20,}\b")),
    ("number", re.compile(r"\d+")),
]


@dataclass
class Break:
    app: str
    request: int              # position of the request within its application
    path: str                 # e.g. system[0] or messages[8].content[0]
    kind: str                 # changed | inserted | removed | tool order
    lost_tokens: int          # previously sent tokens behind the break
    cause: str | None = None  # for "changed": what the differing text looks like
    offset: int = 0
    was: str = ""
    now: str = ""

    @property
    def where(self) -> str:
        return re.sub(r"tools\[\d+\]", "tools", re.sub(r"messages\[\d+\]\.content\[\d+\]", "messages", self.path))


def _token_at(text: str, i: int) -> str:
    a = i
    while a > 0 and not text[a - 1].isspace() and text[a - 1] not in '"\\':
        a -= 1
    b = i
    while b < len(text) and not text[b].isspace() and text[b] not in '"\\':
        b += 1
    return text[a:b]


def _classify(was: str, now: str, i: int) -> str | None:
    a, b = _token_at(was, min(i, len(was) - 1)), _token_at(now, min(i, len(now) - 1))
    for name, pat in _CAUSES:
        if pat.search(a) and pat.search(b):
            return name
    return None


def _snip(text: str, i: int, width: int = 28) -> str:
    return ("…" if i > width else "") + text[max(0, i - width):i + width // 2].replace("\n", " ")


def _plain(text: str) -> str:
    """The readable part of a serialised block: its text if it is a text block, else as is."""
    try:
        body = json.loads(text.split(":", 1)[1])
    except (ValueError, IndexError):
        return text
    if isinstance(body, dict) and isinstance(body.get("text"), str):
        return body["text"]
    if isinstance(body, dict) and isinstance(body.get("content"), str):
        return body["content"]
    return text


def find_breaks(app: str, rows: list[dict]) -> list[Break]:
    out, seen = [], []
    one_shot = bool(rows and rows[0].get("one_shot"))
    for n, row in enumerate(rows):
        cur = rendered(row).segments
        ids = [x.digest for x in cur]
        best, prev = -1, None
        for cand in seen[-WINDOW:]:
            k = 0
            while k < min(len(cand), len(cur)) and cand[k].digest == ids[k]:
                k += 1
            if k >= best:                              # ties: the most recent one
                best, prev = k, cand
        seen.append(cur)
        if prev is None or best >= len(prev):          # first request, or a pure continuation
            continue
        lost = sum(x.tokens for x in prev[best:])
        if best >= len(cur):
            out.append(Break(app, n, cur[-1].path if cur else "request", "removed", lost))
            continue
        path, now_t, was_t = cur[best].path, cur[best].text, prev[best].text
        if path == "messages[0].content[0]" and not one_shot:   # a different conversation, not a break
            continue
        if path.startswith("tools") and sorted(x.digest for x in prev if x.region == "tools") == \
                sorted(x.digest for x in cur if x.region == "tools"):
            out.append(Break(app, n, path, "tool order", lost))
        elif path.startswith("messages") and any(x.digest in ids[best + off + 1:best + off + 14]   # moved further down
                 for off, x in enumerate(prev[best:best + 4])):
            out.append(Break(app, n, path, "inserted", lost))
        elif ids[best] in [x.digest for x in prev[best + 1:best + 12]]:
            out.append(Break(app, n, path, "removed", lost))
        else:
            was_t, now_t = _plain(was_t), _plain(now_t)
            i = 0
            while i < min(len(was_t), len(now_t)) and was_t[i] == now_t[i]:
                i += 1
            out.append(Break(app, n, path, "changed", lost, _classify(was_t, now_t, i), i,
                             _snip(was_t, i), _snip(now_t, i)))
    return out


def summarise(breaks: list[Break]) -> list[dict]:
    """Group breaks by where and why, most costly first."""
    groups: dict[tuple, list[Break]] = defaultdict(list)
    for b in breaks:
        groups[(b.where, b.kind, b.cause)].append(b)
    out = []
    for (where, kind, cause), bs in groups.items():
        out.append({"where": where, "kind": kind, "cause": cause, "count": len(bs),
                    "lost_tokens": sum(b.lost_tokens for b in bs), "example": bs[0]})
    return sorted(out, key=lambda g: -g["lost_tokens"])
