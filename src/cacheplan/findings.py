"""Findings with stable rule ids, each with a suggested fix."""
from __future__ import annotations

from dataclasses import dataclass

from .diagnose import summarise
from .forecast import Knowledge, Verdict
from .messages import prefix_blocks

RULES = {
    "CP001": "a value in the system prompt or tools changes between requests",
    "CP002": "tool definitions are sent in a different order",
    "CP003": "caching is not switched on",
    "CP004": "prompts are shorter than the minimum cacheable length",
    "CP005": "a per-tenant or per-user value sits before static content",
    "CP006": "new content is inserted before content that was already sent",
    "CP007": "previously sent message content changes between requests",
    "CP008": "many pauses are longer than the 5-minute cache lifetime",
    "CP009": "cache writes are rarely read back with the current settings",
}


@dataclass
class Finding:
    rule: str
    message: str
    fix: str
    tokens: int = 0            # estimated tokens re-sent because of it, over the log

    def __str__(self) -> str:
        cost = f" (~{self.tokens:,} tokens re-sent in this log)" if self.tokens else ""
        return f"{self.rule} {self.message}{cost}\n        fix: {self.fix}"


def findings(v: Verdict, rows: list[dict], k: Knowledge, show_content: bool = True) -> list[Finding]:
    out: list[Finding] = []
    groups = summarise(v.breaks)
    tpl = k.templates.get(v.app)
    head_lost = sum(g["lost_tokens"] for g in groups if g["where"].startswith(("system", "tools")))
    slots = tpl.slots() if tpl else []
    for s in slots:
        ex = f' (e.g. "{s.example.replace(chr(10), " ")[:40]}")' if show_content else ""
        if s.kind == "volatile":
            out.append(Finding("CP001", f"system value <<{s.name}>> changes between requests{ex}; "
                               "everything after it, including the conversation history, cannot be reused",
                               "move it to the latest user message, or make it change rarely "
                               "(a date instead of a timestamp)", head_lost))
            head_lost = 0                                  # count the loss once
        elif s.distinct > 1:
            out.append(Finding("CP005", f"system value <<{s.name}>> has {s.distinct} values{ex} and sits "
                               "before static content", "move it behind the static instructions"))
    for g in groups:
        if g["lost_tokens"] < 0.01 * v.tokens:          # too small to matter
            continue
        b = g["example"]
        detail = f': "{b.was}" became "{b.now}"' if show_content and b.was else ""
        if g["kind"] == "tool order":
            out.append(Finding("CP002", f"tool order differs in {g['count']} requests",
                               "build the tool list in a fixed order, for example sorted by name",
                               g["lost_tokens"]))
        elif g["kind"] == "changed" and g["where"].startswith(("system", "tools")) and not slots:
            what = f"a {g['cause']}" if g["cause"] else "text"
            out.append(Finding("CP001", f"{what} in {g['where']} changes between requests "
                               f"({g['count']} times){detail}",
                               "move the changing value to the latest user message, or make it change rarely",
                               g["lost_tokens"]))
        elif g["kind"] == "inserted":
            out.append(Finding("CP006", f"new content is inserted before previously sent content "
                               f"({g['count']} times, first at {b.path})",
                               "use the planned breakpoints above, which sit just before the insertion "
                               "point, or append new turns at the end", g["lost_tokens"]))
        elif g["kind"] == "changed" and g["where"] == "messages":
            what = f"a {g['cause']}" if g["cause"] else "content"
            out.append(Finding("CP007", f"{what} in an earlier message changes between requests "
                               f"({g['count']} times, first at {b.path}){detail}",
                               "keep earlier messages byte-identical; send changing material "
                               "(file contents, retrieved documents) as late as possible", g["lost_tokens"]))
    if (v.recommended != "off" or v.fix_option) and \
            not any(any(ttl for _, ttl in prefix_blocks(r["request"])) for r in rows):
        out.append(Finding("CP003", "no request sets cache_control, so nothing is cached",
                           "see the recommendation above before switching it on"))
    if v.typical_tokens < v.minimum:
        out.append(Finding("CP004", f"the typical prompt is ~{v.typical_tokens:,} tokens (estimate); the minimum "
                           f"cacheable length for this model is {v.minimum:,}",
                           "nothing to do: prompts this short are not cached"))
    if v.gaps_over_5m >= 0.2:
        out.append(Finding("CP008", f"{v.gaps_over_5m:.0%} of pauses between requests exceed 5 minutes",
                           "compare the 1-hour rows in the forecast"))
    if "logged" in v.usage:
        u = v.usage["logged"]
        if u.cache_creation_input_tokens and u.cache_read_input_tokens < 0.5 * u.cache_creation_input_tokens:
            out.append(Finding("CP009", f"current settings write ~{u.cache_creation_input_tokens:,} tokens "
                               f"to the cache and read back ~{u.cache_read_input_tokens:,}",
                               "follow the recommendation above"))
    return out
