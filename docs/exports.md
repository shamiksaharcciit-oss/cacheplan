# Logs from tracing tools

If you already trace your application, you may not need to capture anything. `cacheplan
report FILE` reads these exports directly, and detects which one it is per record:

| Tool | What to export | Recognised by |
| --- | --- | --- |
| Langfuse | Observations (blob-storage export, or the observations API response) as JSON or JSONL. Only `GENERATION` observations are used | `trace_id` and `input` |
| LangSmith | Runs as JSON or JSONL. Only `run_type: "llm"` runs are used | `run_type` |
| OpenTelemetry | OTLP JSON (`resourceSpans`), or one span per line. Three attribute conventions are read: the GenAI semantic conventions (`gen_ai.input.messages`), OpenLLMetry (`gen_ai.prompt.N.*`) and OpenInference (`llm.input_messages.N.*`) | `resourceSpans`, or `gen_ai.*` / `llm.*` attributes |
| LiteLLM | The standard logging payload, one per line, as written by a logging callback | `call_type` and `messages` |

A file can be JSON lines, one JSON array, or one JSON object with a `data`, `runs`,
`observations` or `spans` list.

## What is different from a capture

A capture holds the request exactly as the provider received it. A trace holds what the
tracing integration recorded, and the report says so when a log came from one:

- **Tool definitions and cache markers are often missing.** Without the tools, the token
  count is low; without the markers, the "as logged" option cannot be shown.
- **Order is assumed, not observed.** Requests are laid out in the provider's standard order
  (tools, system, messages). If your application assembles them differently, a capture shows
  it and a trace does not.
- **Usage conventions vary.** Whether "input tokens" include cached tokens depends on the
  integration. cacheplan infers it: if the cache counts fit inside the input count, it is
  taken to include them.
- **The provider is taken from the trace**, or guessed from the model name when the trace
  does not say.

So what changes between requests (timestamps, reordered tools, inserted turns) carries over
well, and the cost forecast is rougher. If a finding matters to you, confirm it on a capture.

## Conversation ids and application names

| Tool | Conversation | Application |
| --- | --- | --- |
| Langfuse | `session_id`, else `trace_id` | observation `name` |
| LangSmith | `extra.metadata.thread_id`, `session_id` or `conversation_id`, else `trace_id`. The run's top-level `session_id` is the tracing project, not a conversation, and is not used | run `name` |
| OpenTelemetry | `gen_ai.conversation.id`, `session.id`, else the trace id | `gen_ai.agent.name`, else `service.name`, else the span name |
| LiteLLM | `session_id`, else `trace_id` | the key alias, else the file name |

Use `--app-field` and `--session-field` to take them from somewhere else; see [Grouping](grouping.md).

## Status

These readers follow the formats' published documentation as read on 24 November 2026 and
are tested on records built from it. None has been run on a real production export by this
project yet. If yours does not load, or loads wrongly, please open an issue with one record
(scramble it first if needed).
