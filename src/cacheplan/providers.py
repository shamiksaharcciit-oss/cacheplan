"""Providers other than the Anthropic API: request formats, caching rules and usage fields.

Every request is converted to one internal shape ({tools, system, messages}, with cache
markers as `cache_control`) so that diagnosis, the prompt fix and the forecast share one
code path. The conversion keeps order and content; it is not meant to be sent anywhere.

Caching rules were read from each provider's documentation on 23 November 2026:
  OpenAI   developers.openai.com/api/docs/guides/prompt-caching and /pricing
  Bedrock  docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
None of the non-Anthropic paths has been checked against a live API by this project yet.
When a log holds the provider's usage, the report's forecast check shows how well they hold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .models import spec as anthropic_spec

SHORT, LONG = 300, 3600            # marker kinds used internally (cache_control ttl default / "1h")


@dataclass(frozen=True)
class Rules:
    """How one model on one provider caches prompts."""
    provider: str = "anthropic"
    mode: str = "breakpoints"          # "breakpoints": cache at markers; "implicit": automatic, no markers
    minimum: int = 1024
    tool_overhead: int = 300
    input_price: float | None = None
    read_x: float = 0.10
    write_short: float = 1.25
    write_long: float | None = 2.0     # None: no long lifetime on this model
    ttl_short: int = 300
    ttl_long: int = 3600
    name_short: str = "5-minute"
    name_long: str = "1-hour"
    interval: int = 0                  # implicit mode: cache boundaries every N tokens
    forecast: bool = True              # False: diagnosis only, no cost forecast
    automatic: bool = True             # False: no automatic mode, breakpoints only
    message_entries: bool = False      # a cache write also makes earlier message endings readable
    source: str = ""                   # where the rules were read
    evidence: str = "documentation only; not compared with live billing"
    known: bool = False
    note: str = ""


def _anthropic(model: str | None) -> Rules:
    s = anthropic_spec(model)
    return Rules("anthropic", "breakpoints", s.minimum, s.tool_overhead, s.input_price, s.read_x, known=s.known)


# ---------------------------------------------------------------- OpenAI
# (fragment, input price, cached-input price, generation) - first match wins.
# Generation "explicit": GPT-5.6 and later, cache writes are billed and breakpoints exist.
_OPENAI = [
    ("gpt-6.2-astra", 8.0, 0.40, "explicit"), ("gpt-6.2-sol", 2.0, 0.10, "explicit"),
    ("gpt-6.1-sol", 2.0, 0.10, "explicit"), ("gpt-6.1-terra", 1.0, 0.05, "explicit"),
    ("gpt-6-astra", 10.0, 1.00, "explicit"), ("gpt-6-luna", 0.10, 0.01, "explicit"), ("gpt-6-sol", 2.0, 0.20, "explicit"),
    ("gpt-5.6-sol", 4.0, 0.40, "explicit"), ("gpt-5.6-terra", 2.0, 0.20, "explicit"), ("gpt-5.6-luna", 0.20, 0.02, "explicit"),
    ("gpt-6", None, None, "explicit"), ("gpt-5.6", None, None, "explicit"),
    ("gpt-5.5", 5.0, 0.50, "implicit"),
    ("gpt-5.4-mini", 0.75, 0.075, "implicit"), ("gpt-5.4-nano", 0.20, 0.02, "implicit"), ("gpt-5.4", 2.5, 0.25, "implicit"),
    ("gpt-5.2", 1.75, 0.175, "implicit"), ("gpt-5.1", 1.25, 0.125, "implicit"),
    ("gpt-5-mini", 0.25, 0.025, "implicit"), ("gpt-5-nano", 0.05, 0.005, "implicit"), ("gpt-5", 1.25, 0.125, "implicit"),
    ("gpt-4.1-mini", 0.40, 0.10, "implicit"), ("gpt-4.1", 2.0, 0.50, "implicit"),
    ("gpt-4o-mini", 0.15, 0.075, "implicit"), ("gpt-4o", 2.5, 1.25, "implicit"),
    ("o3", 2.0, 0.50, "implicit"),
]


def instructions_after_input(model: str | None) -> bool:
    """True for OpenAI models that place top-level `instructions` after `input`, outside the
    cache: GPT-6.2 and later, per OpenAI's prompt-caching guide (read 24 November 2026)."""
    m = re.search(r"gpt-(\d+)(?:\.(\d+))?", (model or "").lower())
    return bool(m) and (int(m.group(1)), int(m.group(2) or 0)) >= (6, 2)


def _openai(model: str | None) -> Rules:
    name = (model or "").lower()
    for fragment, price, cached, gen in _OPENAI:
        if fragment in name:
            read_x = cached / price if price and cached else 0.10
            if gen == "explicit":
                return Rules("openai", "breakpoints", 1024, 0, price, read_x, 1.25, 2.5, 1800, 43200,
                             "30-minute", "12-hour", known=True, message_entries=True,
                             note="12-hour lifetime is available in the Responses API only")
            return Rules("openai", "implicit", 1024, 0, price, read_x, 1.0, 1.0, 300, 1800,
                         "in-memory", "24h-retention", interval=2048 if "gpt-5.5" in name else 128, known=True,
                         note="caching is automatic on this model with no write fee; the one setting is "
                              "prompt_cache_retention. Lifetimes are the typical values OpenAI documents "
                              "(5 to 10 minutes in memory, about 30 minutes with 24h retention), not guarantees")
    return Rules("openai", "implicit", 1024, 0, None, 0.5, 1.0, 1.0, 300, 1800, "in-memory", "24h-retention",
                 interval=128, note="model not recognised: assumed automatic caching with a 50% cached-input price")


# ---------------------------------------------------------------- Bedrock
_BEDROCK_MIN = [("opus-4-7", 4096), ("opus-4-6", 4096), ("opus-4-5", 4096), ("haiku-4-5", 4096),
                ("opus-4-8", 1024), ("sonnet-5-5", 512), ("sonnet-5", 1024), ("sonnet-4-6", 1024)]


def _bedrock(model: str | None) -> Rules:
    name = (model or "").lower().replace(".", "-")
    if "nova" in name:
        return Rules("bedrock", "breakpoints", 1000, 0, None, 0.10, 1.25, None, known=True, automatic=False,
                     note="Nova: 5-minute lifetime only, system and messages only, at most 20K cached tokens; "
                          "prices are not built in, set --price")
    if "claude" in name or "anthropic" in name:
        base = _anthropic(name)
        minimum = next((m for frag, m in _BEDROCK_MIN if frag in name), base.minimum)
        return replace(base, provider="bedrock", minimum=minimum,
                       note="Claude on Bedrock: Anthropic's prices are assumed; check your Bedrock rates")
    return Rules("bedrock", "breakpoints", 1024, 0, None, forecast=False,
                 note="caching rules for this Bedrock model are not built in: diagnosis only")


DOCS = {
    "anthropic": "https://docs.claude.com/en/docs/build-with-claude/prompt-caching",
    "openai": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "bedrock": "https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html",
    "gemini": "https://ai.google.dev/gemini-api/docs/caching",
}
READ_ON = "2026-11-23"


def _evidence(r: Rules) -> str:
    """What stands behind a rule set. Kept next to the rules so the report, the README table
    and `cacheplan providers` cannot drift apart."""
    if not r.forecast:
        return "no forecast"
    if r.provider == "anthropic":
        return "live: 14 forecast-vs-billed rows on claude-sonnet-5-5, all within 2 points (2026-11-24)"
    if r.provider == "openai" and r.mode == "implicit":
        return "live: 2 runs on gpt-5.4-mini, 19% vs 21% and 23% vs 17% with a warm cache (2026-11-23)"
    if r.provider == "openai":
        return ("live: 2 runs on gpt-5.6-luna, automatic mode only (2026-11-24); breakpoints, the 12-hour "
                "lifetime and GPT-6.x are documentation only")
    return "documentation only; not compared with live billing"


def rules(provider: str, model: str | None) -> Rules:
    if provider == "openai":
        r = _openai(model)
    elif provider == "bedrock":
        r = _bedrock(model)
    elif provider == "gemini":
        name = (model or "").lower()
        minimum = 2048 if "gemini-2" in name else 4096 if "gemini" in name else 2048
        r = Rules("gemini", "implicit", minimum, 0, None, forecast=False, known="gemini" in name,
                  note="Gemini's documentation does not state the cached-token discount or lifetime: "
                       "diagnosis only, no cost forecast")
    else:
        r = _anthropic(model)
    return replace(r, source=f"{DOCS[r.provider]} (read {READ_ON})", evidence=_evidence(r))


# ---------------------------------------------------------------- request conversion
def _mark(block: dict, ttl) -> dict:
    return dict(block, cache_control={"type": "ephemeral", "ttl": "1h"} if ttl in ("1h", "12h") else {"type": "ephemeral"})


def _text(t) -> dict:
    return {"type": "text", "text": t if isinstance(t, str) else str(t)}


def _openai_parts(content, default_ttl=None) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [_text(content)]
    out = []
    for p in content:
        if not isinstance(p, dict):
            out.append(_text(p))
            continue
        bp = p.get("prompt_cache_breakpoint")
        q = {k: v for k, v in p.items() if k != "prompt_cache_breakpoint"}
        if q.get("type") in ("text", "input_text", "output_text") and isinstance(q.get("text"), str):
            q = {"type": "text", "text": q["text"]}
        out.append(_mark(q, bp.get("ttl") or default_ttl) if isinstance(bp, dict) else q)
    return out


def from_openai(body: dict) -> dict:
    """Chat Completions or Responses request -> internal shape. Leading system/developer
    messages and `instructions` become the system prompt; tools keep their order."""
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        tools.append(dict(t, name=fn.get("name") or t.get("type") or "tool"))
    system: list[dict] = []
    tail: list[dict] = []
    if isinstance(body.get("instructions"), str) and body["instructions"]:
        # GPT-6.2 and later append top-level instructions after the input and bill them
        # uncached on every request; earlier models render them ahead of the input.
        (tail if instructions_after_input(body.get("model")) else system).append(_text(body["instructions"]))
    items = body.get("messages")
    if items is None:
        items = body.get("input")
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
    messages, leading = [], True
    ttl = (body.get("prompt_cache_options") or {}).get("ttl")
    for it in items or []:
        role = it.get("role")
        if leading and role in ("system", "developer"):
            system += _openai_parts(it.get("content"), ttl)
            continue
        leading = False
        if role == "tool" or it.get("type") == "function_call_output":
            block = {"type": "tool_result", "tool_use_id": it.get("tool_call_id") or it.get("call_id"),
                     "content": it.get("content") if role == "tool" else it.get("output")}
            messages.append({"role": "user", "content": [block]})
        elif it.get("type") == "function_call":
            messages.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": it.get("call_id"), "name": it.get("name"), "input": it.get("arguments")}]})
        elif role in ("user", "assistant", "system", "developer"):
            blocks = _openai_parts(it.get("content"), ttl)
            for c in it.get("tool_calls") or []:
                f = c.get("function") or {}
                blocks.append({"type": "tool_use", "id": c.get("id"), "name": f.get("name"), "input": f.get("arguments")})
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks or [_text("")]})
        else:
            messages.append({"role": "user", "content": [dict(it)]})        # reasoning items and the like
    out = {"model": body.get("model"), "messages": messages}
    if tools:
        out["tools"] = tools
    if system:
        out["system"] = system
    if tail:
        out["uncached_tail"] = tail
    # GPT-5.6 and later cache automatically unless the request asks for explicit mode.
    options = body.get("prompt_cache_options") or {}
    if _openai(body.get("model")).mode == "breakpoints" and options.get("mode", "implicit") != "explicit":
        out["cache_control"] = _mark({}, options.get("ttl"))["cache_control"]
    return out


def _converse_blocks(blocks) -> list[dict]:
    out: list[dict] = []
    for b in blocks or []:
        if "cachePoint" in b:
            if out:
                out[-1] = _mark(out[-1], (b["cachePoint"] or {}).get("ttl"))
            continue
        if "text" in b and len(b) == 1:
            out.append(_text(b["text"]))
        elif "toolUse" in b:
            u = b["toolUse"]
            out.append({"type": "tool_use", "id": u.get("toolUseId"), "name": u.get("name"), "input": u.get("input")})
        elif "toolResult" in b:
            r = b["toolResult"]
            out.append({"type": "tool_result", "tool_use_id": r.get("toolUseId"), "content": r.get("content")})
        else:
            out.append(dict(b))
    return out


def from_converse(body: dict) -> dict:
    """Bedrock Converse request -> internal shape. cachePoint blocks become markers on the
    block before them."""
    tools: list[dict] = []
    for t in (body.get("toolConfig") or {}).get("tools") or []:
        if "cachePoint" in t:
            if tools:
                tools[-1] = _mark(tools[-1], (t["cachePoint"] or {}).get("ttl"))
            continue
        s = t.get("toolSpec") or t
        tools.append(dict(s, name=s.get("name", "tool")))
    out = {"model": body.get("modelId") or body.get("model"),
           "messages": [{"role": m.get("role", "user"), "content": _converse_blocks(m.get("content"))}
                        for m in body.get("messages") or []]}
    if tools:
        out["tools"] = tools
    system = _converse_blocks(body.get("system"))
    if system:
        out["system"] = system
    auto = (body.get("additionalModelRequestFields") or {}).get("cache_control")
    if auto:
        out["cache_control"] = auto
    return out


def from_gemini(body: dict) -> dict:
    """Gemini generateContent request -> internal shape."""
    def parts(c):
        return [_text(p["text"]) if isinstance(p.get("text"), str) and len(p) == 1 else dict(p)
                for p in (c or {}).get("parts") or []]
    tools = []
    for t in body.get("tools") or []:
        for f in t.get("functionDeclarations") or t.get("function_declarations") or [t]:
            tools.append(dict(f, name=f.get("name", "tool")))
    out = {"model": body.get("model"),
           "messages": [{"role": "assistant" if c.get("role") == "model" else "user", "content": parts(c) or [_text("")]}
                        for c in body.get("contents") or []]}
    if tools:
        out["tools"] = tools
    system = parts(body.get("systemInstruction") or body.get("system_instruction"))
    if system:
        out["system"] = system
    return out


def detect(body: dict) -> str:
    """Which provider's request format this body is."""
    if "contents" in body:
        return "gemini"
    if "toolConfig" in body or "modelId" in body or "inferenceConfig" in body:
        return "bedrock"
    if str(body.get("anthropic_version", "")).startswith("bedrock"):
        return "bedrock"
    if "input" in body or "instructions" in body:
        return "openai"
    msgs = body.get("messages") or []
    model = str(body.get("model") or "").lower()
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if any(isinstance(m, dict) and m.get("role") in ("system", "developer", "tool") for m in msgs):
        return "openai"
    if any(isinstance(t, dict) and t.get("type") == "function" for t in body.get("tools") or []):
        return "openai"
    if msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get("content"), list) and msgs[0]["content"] \
            and isinstance(msgs[0]["content"][0], dict) and "type" not in msgs[0]["content"][0] and "text" in msgs[0]["content"][0]:
        return "bedrock"
    return "anthropic"


def normalise(body: dict, provider: str | None = None) -> tuple[str, dict]:
    """(provider, request in the internal shape)."""
    provider = provider or detect(body)
    if provider == "openai":
        return provider, from_openai(body)
    if provider == "gemini":
        return provider, from_gemini(body)
    if provider == "bedrock" and not str(body.get("anthropic_version", "")).startswith("bedrock"):
        return provider, from_converse(body)
    return provider, body


# ---------------------------------------------------------------- usage
def usage_parts(u: dict) -> tuple[int, int, int, int] | None:
    """(uncached input, cache writes, cache reads, long-lifetime writes) from any provider's usage block."""
    if not isinstance(u, dict):
        return None
    if "inputTokens" in u:                                              # Bedrock Converse
        return (int(u.get("inputTokens") or 0), int(u.get("cacheWriteInputTokens") or 0),
                int(u.get("cacheReadInputTokens") or 0), 0)
    if "promptTokenCount" in u:                                         # Gemini
        cached = int(u.get("cachedContentTokenCount") or 0)
        return int(u["promptTokenCount"]) - cached, 0, cached, 0
    if "prompt_tokens" in u:                                            # OpenAI Chat Completions
        d = u.get("prompt_tokens_details") or {}
        cached, written = int(d.get("cached_tokens") or 0), int(d.get("cache_write_tokens") or 0)
        return int(u["prompt_tokens"]) - cached - written, written, cached, 0
    if "input_tokens_details" in u:                                     # OpenAI Responses
        d = u.get("input_tokens_details") or {}
        cached, written = int(d.get("cached_tokens") or 0), int(d.get("cache_write_tokens") or 0)
        long = int((d.get("cache_write_tokens_details") or {}).get("ttl_12h_tokens") or 0)
        return int(u.get("input_tokens") or 0) - cached - written, written, cached, long
    if u.get("input_tokens") is not None:                               # Anthropic, promptcachelint
        creation = u.get("cache_creation_input_tokens", u.get("cache_write")) or 0
        read = u.get("cache_read_input_tokens", u.get("cache_read")) or 0
        long = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
        return int(u["input_tokens"]), int(creation), int(read), int(long)
    return None
