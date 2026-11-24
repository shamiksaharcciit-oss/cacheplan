# Grouping: applications and conversations

Everything cacheplan reports is per *application*: a set of requests built from one prompt
template. Getting the grouping right matters, because two different prompts mixed together
look like one prompt that keeps changing.

## Defaults

1. **Application label.** From the log line's `app` field, else the file name.
2. **Provider.** Requests to different providers are kept apart (`my-app [openai]`).
3. **Prompt.** Under one label, requests whose system prompts share less than half their
   lines are split (`my-app#1 "You are a…"`, `my-app#2 "Summarise…"`). An agent's main
   prompt and its commit-message prompt are different applications.
4. **Conversation.** From the line's `session` field, else from the first message: requests
   that start with the same first message belong to one conversation.

## Options

Available on `report`, `diagnose`, `check`, `gaps`, `baseline`, `plan`, `verify` and `quality`.

| Option | Use it when |
| --- | --- |
| `--app-field PATH` | The label lives elsewhere in each line. A dotted path, looked up in the line and then in the request body: `metadata.route`, `tags.service` |
| `--session-field PATH` | The conversation id lives elsewhere: `metadata.session_id`, `metadata.user_id`, `prompt_cache_key` |
| `--no-split` | The prompt split is wrong for you, for example a system prompt that is mostly retrieved text and legitimately differs between requests |
| `--one-shot` | Requests are independent queries, not growing conversations |

## One-shot workloads

By default, a request whose *first message* differs from every earlier one is read as the
start of a new conversation, not as a broken prefix. That is right for chat.

It is wrong for a workload like "the same 30-page manual plus a different question each
time". There the first message is where the reusable part and the changing part meet, and
`--one-shot` reports it:

```
CP007 content in an earlier message changes between requests (11 times, first at messages[0].content[0])
      fix: put the part every query shares in its own content block, or in the system prompt,
           ahead of the part that changes
```

On providers that cache whole blocks, a shared manual and a changing question in one text
block cannot be cached at all; in separate blocks the manual can.
