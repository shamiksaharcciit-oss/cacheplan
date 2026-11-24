"""Findings with stable rule ids, each with a suggested fix."""
from __future__ import annotations

from dataclasses import dataclass

from .diagnose import summarise
from .forecast import Knowledge, Verdict
from .messages import rendered

RULES = {
    "CP001": "a value in the system prompt or tools changes between requests",
    "CP002": "tool definitions are sent in a different order",
    "CP003": "caching is not switched on",
    "CP004": "prompts are shorter than the minimum cacheable length",
    "CP005": "a per-tenant or per-user value sits before static content",
    "CP006": "new content is inserted before content that was already sent",
    "CP007": "previously sent message content changes between requests",
    "CP008": "many pauses are longer than the short cache lifetime",
    "CP009": "cache writes are rarely read back with the current settings",
    "CP010": "top-level instructions are billed uncached on every request",
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
    head_lost = min(v.tokens, sum(g["lost_tokens"] for g in groups if g["where"].startswith(("system", "tools"))))
    slots = tpl.slots() if tpl else []
    for s in slots:
        ex = f' (e.g. "{s.example.replace(chr(10), " ")[:40]}")' if show_content else ""
        if s.kind == "volatile" and v.rules.mode == "implicit":
            # These models reuse any shared prefix, so only what follows the value is lost.
            lost = 0
            if "fix_5m" in v.usage and "auto_5m" in v.usage:
                lost = max(0, v.usage["fix_5m"].cache_read_input_tokens - v.usage["auto_5m"].cache_read_input_tokens)
            out.append(Finding("CP001", f"system value <<{s.name}>> changes between requests{ex}; the prompt before "
                               "it is still reused on this model, what follows it (the conversation history) is not",
                               "move it to the latest user message, or make it change rarely "
                               "(a date instead of a timestamp)", lost))
            head_lost = 0
        elif s.kind == "volatile":
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
                               ("use the planned breakpoints above, which sit just before the insertion "
                                "point, or append new turns at the end") if v.rules.mode == "breakpoints" and v.rules.forecast
                               else "append new turns at the end and keep what was already sent in place",
                               g["lost_tokens"]))
        elif g["kind"] == "changed" and g["where"] == "messages":
            what = f"a {g['cause']}" if g["cause"] else "content"
            out.append(Finding("CP007", f"{what} in an earlier message changes between requests "
                               f"({g['count']} times, first at {b.path}){detail}",
                               "keep earlier messages byte-identical; send changing material "
                               "(file contents, retrieved documents) as late as possible", g["lost_tokens"]))
    if v.rules.mode == "breakpoints" and (v.recommended != "off" or v.fix_option) and \
            not any(rendered(r).markers() for r in rows):
        marker = {"bedrock": "a cachePoint", "openai": "a cache breakpoint"}.get(v.rules.provider, "cache_control")
        out.append(Finding("CP003", f"no request sets {marker}, so nothing is cached",
                           "see the recommendation above before switching it on"))
    if v.typical_tokens < v.minimum:
        out.append(Finding("CP004", f"the typical prompt is ~{v.typical_tokens:,} tokens (estimate); the minimum "
                           f"cacheable length for this model is {v.minimum:,}",
                           "nothing to do: prompts this short are not cached"))
    if v.gaps_over_5m >= 0.2:
        short = f"{v.rules.ttl_short // 60} minutes"
        out.append(Finding("CP008", f"{v.gaps_over_5m:.0%} of pauses between requests exceed {short}",
                           f"compare the {v.rules.name_long} rows in the forecast" if v.rules.write_long is not None and v.rules.forecast
                           else "nothing cached survives those pauses on this model"))
    tail = sum(rendered(r).tail_tokens for r in rows)
    if tail >= 0.01 * v.tokens:
        out.append(Finding("CP010", f"top-level `instructions` are ~{tail / max(1, v.tokens):.0%} of input tokens; this model "
                           "places them after the input, outside the cache, and bills them in full every time",
                           "send them as a developer message at the start of `input` instead", tail))
    if "logged" in v.usage:
        u = v.usage["logged"]
        if u.cache_creation_input_tokens and u.cache_read_input_tokens < 0.5 * u.cache_creation_input_tokens:
            out.append(Finding("CP009", f"current settings write ~{u.cache_creation_input_tokens:,} tokens "
                               f"to the cache and read back ~{u.cache_read_input_tokens:,}",
                               "follow the recommendation above"))
    return out
