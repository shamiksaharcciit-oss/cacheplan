"""Read exports from tracing tools: Langfuse, LangSmith, OpenTelemetry and LiteLLM.

Each record is turned into the same line cacheplan's own capture writes. Three things differ
from a capture, and the report says so when a log came through here:

  * A tracer records the messages its integration saw, which is not always byte for byte
    what the provider received. Tool definitions and cache markers are often missing, and
    the order of tools and system prompt is the provider's standard one, not observed.
  * Token usage is taken from the trace when it is there; whether "input tokens" include
    cached ones varies by integration, so it is inferred from the numbers.
  * The provider is taken from the trace, else guessed from the model name.

So diagnosis (what changes between requests) carries over well, and cost forecasts are
rougher than from a capture. None of these readers has been run on a real production export
by this project; they follow the formats' published documentation (read 24 November 2026).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

SKIP = "not a model call"


# ------------------------------------------------------------------ small helpers
def _json(value):
    """Tracers often store structured values as JSON text."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _time(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1e9 if v > 1e17 else v / 1e6 if v > 1e14 else v / 1e3 if v > 1e11 else v
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return _time(float(text))
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def guess_provider(model: str | None, hint: str | None = None) -> str:
    h, m = (hint or "").lower(), (model or "").lower()
    if "bedrock" in h or m.startswith("bedrock/") or re.match(r"(us|eu|apac|global)\.(anthropic|amazon)\.|anthropic\.claude|amazon\.nova", m):
        return "bedrock"
    for name, marks in (("anthropic", ("anthropic", "claude")), ("gemini", ("gemini", "google", "gcp", "vertex")),
                        ("openai", ("openai", "azure", "gpt-", "chatgpt", "o1", "o3", "o4"))):
        if any(k in h for k in marks) or (not h and any(k in m for k in marks)):
            return name
    return "openai" if not h and not m else ("anthropic" if "claude" in m else "gemini" if "gemini" in m else "openai")


def _usage(inputs, read=0, write=0) -> dict | None:
    """Anthropic-style usage from an input count plus cache counts. If the cache counts fit
    inside the input count it is taken to include them (the usual convention); otherwise the
    input count is taken as the uncached remainder."""
    if inputs is None:
        return None
    inputs, read, write = int(inputs), int(read or 0), int(write or 0)
    uncached = inputs - read - write if read + write <= inputs else inputs
    return {"input_tokens": uncached, "cache_read_input_tokens": read, "cache_creation_input_tokens": write}


_ROLES = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "model": "assistant",
          "system": "system", "developer": "developer", "tool": "tool", "function": "tool", "chat": "user"}
_LC = {"HumanMessage": "user", "AIMessage": "assistant", "SystemMessage": "system", "ToolMessage": "tool",
       "FunctionMessage": "tool", "HumanMessageChunk": "user", "AIMessageChunk": "assistant", "ChatMessage": "user"}


def _message(m) -> dict | None:
    """One message in any of the shapes tracers store -> an OpenAI-style message."""
    if isinstance(m, str):
        return {"role": "user", "content": m}
    if isinstance(m, (list, tuple)) and len(m) == 2 and isinstance(m[0], str):      # ("human", "text")
        return {"role": _ROLES.get(m[0], "user"), "content": m[1]}
    if not isinstance(m, dict):
        return None
    if m.get("lc") and isinstance(m.get("kwargs"), dict):                           # LangChain serialised form
        role = _LC.get((m.get("id") or [""])[-1], "user")
        m = dict(m["kwargs"], role=role)
    if isinstance(m.get("message"), dict):                                          # OpenInference nesting
        m = m["message"]
    role = _ROLES.get(str(m.get("role") or m.get("type") or "user").lower(), "user")
    data = m.get("data") if isinstance(m.get("data"), dict) else m                  # older LangChain dumps
    content = data.get("content")
    if content is None and isinstance(m.get("parts"), list):                        # OpenTelemetry parts
        content, calls = [], []
        for p in m["parts"]:
            kind = p.get("type") if isinstance(p, dict) else None
            if kind == "text":
                content.append({"type": "text", "text": p.get("content", "")})
            elif kind == "tool_call":
                calls.append({"id": p.get("id"), "function": {"name": p.get("name"), "arguments": json.dumps(p.get("arguments"))}})
            elif kind == "tool_call_response":
                return {"role": "tool", "tool_call_id": p.get("id"),
                        "content": p.get("response") if isinstance(p.get("response"), str) else json.dumps(p.get("response", p.get("result")))}
            elif isinstance(p, dict):
                content.append(p)
        out = {"role": role, "content": content or None}
        if calls:
            out["tool_calls"] = calls
        return out
    out = {"role": role, "content": content}
    calls = data.get("tool_calls") or (data.get("additional_kwargs") or {}).get("tool_calls")
    if calls:
        out["tool_calls"] = [c if "function" in c else {"id": c.get("id"), "function": {
            "name": c.get("name"), "arguments": json.dumps(c.get("args", c.get("arguments")))}} for c in calls if isinstance(c, dict)]
    if role == "tool":
        out["tool_call_id"] = data.get("tool_call_id") or data.get("id")
    return out


def _messages(value) -> list[dict]:
    value = _json(value)
    if isinstance(value, dict):
        value = value.get("messages") or value.get("input") or value.get("prompt") or value.get("contents") or []
        value = _json(value)
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list) and value and isinstance(value[0], list) and not (len(value[0]) == 2 and isinstance(value[0][0], str)):
        value = value[0]                                                             # a batch of one
    return [x for x in (_message(m) for m in value or []) if x]


def _tools(value) -> list[dict]:
    out = []
    for t in _json(value) or []:
        t = _json(t)
        if not isinstance(t, dict):
            continue
        t = t.get("tool", t)
        if isinstance(t.get("json_schema"), (dict, str)):
            t = _json(t["json_schema"])
        if "function" in t or "input_schema" in t or "toolSpec" in t:
            out.append(t)
        elif t.get("name"):
            out.append({"type": "function", "function": {k: v for k, v in t.items() if k != "type"}})
    return out


def _line(source, provider, model, messages, tools=None, system=None, ts=None, app=None, session=None, usage=None) -> dict:
    from .providers import from_openai
    msgs = list(messages)
    if system:
        text = system if isinstance(system, str) else "\n".join(
            (p.get("content") or p.get("text") or "") if isinstance(p, dict) else str(p) for p in system)
        msgs.insert(0, {"role": "system", "content": text})
    body = {"model": model, "messages": msgs}
    if tools:
        body["tools"] = tools
    return {"ts": ts, "app": app, "session": session, "provider": provider, "usage": usage,
            "request": from_openai(body), "source": source, "normalised": True}


# ------------------------------------------------------------------ LiteLLM
def _litellm(d: dict) -> list[dict] | str:
    if str(d.get("call_type", "")).lstrip("a") not in ("completion", "responses", "nthropic_messages", "text_completion", "generate_content"):
        return SKIP
    model = d.get("model") or ""
    hint = d.get("custom_llm_provider") or (d.get("hidden_params") or {}).get("custom_llm_provider") \
        or (model.split("/")[0] if "/" in model else None)
    params = d.get("model_parameters") or {}
    meta = d.get("metadata") or {}
    usage = meta.get("usage_object") if isinstance(meta.get("usage_object"), dict) else None
    return [_line("litellm", guess_provider(model, hint), model.split("/")[-1], _messages(d.get("messages")), _tools(params.get("tools")),
                  ts=_time(d.get("startTime")), app=meta.get("user_api_key_alias") or d.get("request_tags", [None])[0] if d.get("request_tags") else meta.get("user_api_key_alias"),
                  session=d.get("session_id") or d.get("trace_id"), usage=usage)]


# ------------------------------------------------------------------ Langfuse
def _langfuse(d: dict) -> list[dict] | str:
    if str(d.get("type", "GENERATION")).upper() != "GENERATION":
        return SKIP
    model = d.get("provided_model_name") or d.get("providedModelName") or d.get("model")
    source = _json(d.get("input"))
    meta = _json(d.get("metadata")) or {}
    params = _json(d.get("model_parameters") or d.get("modelParameters")) or {}
    tools = _tools(source.get("tools")) if isinstance(source, dict) else []
    system = source.get("system") if isinstance(source, dict) and isinstance(source.get("system"), (str, list)) else None
    u = _json(d.get("usage_details") or d.get("usageDetails")) or {}
    read = u.get("cache_read_input_tokens") or u.get("input_cached_tokens") or u.get("input_cache_read") or 0
    write = u.get("cache_creation_input_tokens") or u.get("input_cache_creation") or 0
    hint = meta.get("ls_provider") if isinstance(meta, dict) else None
    return [_line("langfuse", guess_provider(model, hint), model, _messages(source), tools or _tools(params.get("tools") if isinstance(params, dict) else None),
                  system, _time(d.get("start_time") or d.get("startTime")), d.get("name"),
                  d.get("session_id") or d.get("sessionId") or d.get("trace_id") or d.get("traceId"), _usage(u.get("input"), read, write))]


# ------------------------------------------------------------------ LangSmith
def _langsmith(d: dict) -> list[dict] | str:
    if d.get("run_type") != "llm":
        return SKIP
    extra = d.get("extra") or {}
    inv, meta = extra.get("invocation_params") or {}, extra.get("metadata") or {}
    model = meta.get("ls_model_name") or inv.get("model") or inv.get("model_name") or inv.get("model_id")
    outputs = d.get("outputs") or {}
    um = outputs.get("usage_metadata") or {}
    if not um:
        try:
            um = outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"] or {}
        except (KeyError, IndexError, TypeError):
            um = {}
    details = um.get("input_token_details") or d.get("prompt_token_details") or {}
    usage = _usage(um.get("input_tokens", d.get("prompt_tokens")), details.get("cache_read"), details.get("cache_creation"))
    session = meta.get("thread_id") or meta.get("session_id") or meta.get("conversation_id") or d.get("trace_id")
    return [_line("langsmith", guess_provider(model, meta.get("ls_provider") or inv.get("_type")), model, _messages(d.get("inputs")),
                  _tools(inv.get("tools") or inv.get("functions")), None, _time(d.get("start_time")), d.get("name"), session, usage)]


# ------------------------------------------------------------------ OpenTelemetry
def _attr_value(v):
    if not isinstance(v, dict):
        return v
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in v:
            return int(v[key]) if key == "intValue" else v[key]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {x["key"]: _attr_value(x.get("value")) for x in v["kvlistValue"].get("values", [])}
    return None


def _attrs(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    return {a["key"]: _attr_value(a.get("value")) for a in raw or [] if isinstance(a, dict) and "key" in a}


def _indexed(attrs: dict, prefix: str) -> list[dict]:
    """gen_ai.prompt.0.role, gen_ai.prompt.0.content, ... -> [{role, content}, ...]"""
    items: dict[int, dict] = {}
    for key, value in attrs.items():
        if key.startswith(prefix + "."):
            n, _, rest = key[len(prefix) + 1:].partition(".")
            if n.isdigit():
                items.setdefault(int(n), {})[rest] = value
    return [items[i] for i in sorted(items)]


def _span(span: dict, service: str | None = None) -> list[dict] | str:
    a = _attrs(span.get("attributes"))
    model = a.get("gen_ai.request.model") or a.get("llm.model_name") or a.get("gen_ai.response.model")
    messages, tools = [], _tools(a.get("gen_ai.tool.definitions"))
    if a.get("gen_ai.input.messages") is not None:                                   # current GenAI conventions
        messages = _messages(a["gen_ai.input.messages"])
    elif any(k.startswith("gen_ai.prompt.") for k in a):                              # OpenLLMetry
        messages = [x for x in (_message({"role": m.get("role"), "content": _json(m.get("content")),
                                          "tool_call_id": m.get("tool_call_id")}) for m in _indexed(a, "gen_ai.prompt")) if x]
        tools = tools or _tools([{"name": f.get("name"), "description": f.get("description"), "parameters": _json(f.get("parameters"))}
                                 for f in _indexed(a, "llm.request.functions")])
    elif any(k.startswith("llm.input_messages.") for k in a):                         # OpenInference
        for m in _indexed(a, "llm.input_messages"):
            parts = [{"type": "text", "text": v} for k, v in sorted(m.items()) if k.startswith("message.contents.") and k.endswith(".text")]
            messages.append({"role": _ROLES.get(str(m.get("message.role", "user")).lower(), "user"),
                             "content": m.get("message.content") if m.get("message.content") is not None else parts or None,
                             **({"tool_call_id": m["message.tool_call_id"]} if m.get("message.tool_call_id") else {})})
        tools = tools or _tools([t.get("tool.json_schema") for t in _indexed(a, "llm.tools")])
    if not messages or not model:
        return SKIP
    read = a.get("gen_ai.usage.cache_read.input_tokens", a.get("gen_ai.usage.cache_read_input_tokens", a.get("llm.token_count.prompt_details.cache_read")))
    write = a.get("gen_ai.usage.cache_creation.input_tokens", a.get("gen_ai.usage.cache_creation_input_tokens", a.get("llm.token_count.prompt_details.cache_write")))
    inputs = a.get("gen_ai.usage.input_tokens", a.get("gen_ai.usage.prompt_tokens", a.get("llm.token_count.prompt")))
    hint = a.get("gen_ai.provider.name") or a.get("gen_ai.system") or a.get("llm.provider") or a.get("llm.system")
    ts = _time(span.get("startTimeUnixNano") or span.get("start_time_unix_nano") or span.get("start_time") or span.get("startTime"))
    return [_line("opentelemetry", guess_provider(model, hint), model, messages, tools, _json(a.get("gen_ai.system_instructions")), ts,
                  a.get("gen_ai.agent.name") or service or span.get("name"),
                  a.get("gen_ai.conversation.id") or a.get("session.id") or span.get("traceId") or span.get("trace_id"),
                  _usage(inputs, read, write))]


def _otlp(d: dict) -> list[dict] | str:
    out = []
    for rs in d.get("resourceSpans") or d.get("resource_spans") or []:
        service = _attrs((rs.get("resource") or {}).get("attributes")).get("service.name")
        for ss in rs.get("scopeSpans") or rs.get("scope_spans") or []:
            for span in ss.get("spans") or []:
                got = _span(span, service)
                if isinstance(got, list):
                    out += got
    return out or SKIP


# ------------------------------------------------------------------ entry points
def expand(d) -> list[dict] | str:
    """A record from any supported export -> cacheplan lines. Records that are already
    cacheplan lines or raw request bodies come back unchanged. A string says why a
    recognised record was left out."""
    if not isinstance(d, dict):
        return [d]
    if "resourceSpans" in d or "resource_spans" in d:
        return _otlp(d)
    if isinstance(d.get("request"), dict) or isinstance(d.get("body"), dict):
        return [d]
    if "run_type" in d and ("inputs" in d or "outputs" in d):
        return _langsmith(d)
    if "call_type" in d and "messages" in d and ("startTime" in d or "model_parameters" in d or "response_cost" in d):
        return _litellm(d)
    if ("trace_id" in d or "traceId" in d) and "input" in d and ("type" in d or "provided_model_name" in d or "usage_details" in d or "usageDetails" in d):
        return _langfuse(d)
    attrs = d.get("attributes")
    if attrs is not None and (isinstance(attrs, list) or any(str(k).startswith(("gen_ai.", "llm.")) for k in attrs)):
        return _span(d, _attrs((d.get("resource") or {}).get("attributes")).get("service.name") if isinstance(d.get("resource"), dict) else None)
    return [d]


def records(text: str):
    """The records in a file: JSON lines, one JSON array, or one JSON object (an OTLP dump
    or an API response with a `data` or `runs` list). Yields parsed values, or the string
    "invalid JSON" for a line that does not parse."""
    stripped = text.lstrip()
    if stripped[:1] in "[{":
        try:
            whole = json.loads(stripped)
        except ValueError:
            whole = None
        if isinstance(whole, list):
            yield from whole
            return
        if isinstance(whole, dict):
            for key in ("data", "runs", "observations", "spans"):
                if isinstance(whole.get(key), list):
                    yield from whole[key]
                    return
            yield whole
            return
    for line in text.splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except ValueError:
                yield "invalid JSON"
