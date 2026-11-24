# Applying a plan

**Experimental.** The report tells you where to put cache markers. This is a way to have
them put there for you, without touching your prompt-building code.

## 1. Save the plan

```
cacheplan plan requests.jsonl --save plan.json
```

```
my-agent: Planned breakpoints, 5-minute, forecast 49% of the uncached input cost
  breakpoints at 3 blocks before the last message block

saved the plan for 1 application(s) to plan.json
```

The file holds the recommended action, the lifetime and the breakpoint positions per
application. It contains no prompt text, so it can be committed.

## 2. Wrap the client

```python
import anthropic
from cacheplan.apply import PlanTransport

client = anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(
    transport=PlanTransport("plan.json", app="my-agent")))
```

`app` can be left out when the file has one application. For the async client use
`AsyncPlanTransport` with `anthropic.DefaultAsyncHttpxClient`.

## What it does

| The plan says | Each Messages API request gets |
| --- | --- |
| Automatic caching | the top-level `cache_control` field |
| Planned breakpoints | `cache_control` markers at the planned positions (at most four) |
| Caching off, or keep your current settings | no change |

## What it never does

- **It does not change, move or drop content.** Only cache markers are added or replaced,
  so the model reads exactly what your application wrote. Where a marker goes on text sent
  as a plain string, that string becomes a one-block list, which the API treats the same.
- **It does not apply the prompt fix.** Moving a timestamp out of the system prompt changes
  what the model reads; that stays a decision for you, checked with `cacheplan quality`.
- **It does not break requests.** If rewriting fails for any reason, the request goes out
  as your application built it.

## Limits

- Anthropic Messages API only.
- Positions are counted in content blocks from the end of the request. If your application
  later changes how it assembles prompts, the plan goes stale: re-run `cacheplan plan`.
  `cacheplan baseline` in CI catches the drift.
- Live evidence is one run: on 24 November 2026, 15 requests on `claude-sonnet-5-5`
  (`benchmarks/providers/anthropic_driver.py`, three conversations sharing a system prompt)
  were forecast by the saved plan at 15% of the uncached input cost and billed at 14% behind
  the wrapper, with 92% of input tokens read from the cache. Check it on your own traffic
  with a capture taken *behind* the wrapper: the report's "Forecast check" line should be
  close to the forecast in `plan.json`.
