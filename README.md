# cacheplan

Diagnose LLM prompt caching, forecast what each setting would cost, and plan the fix.

Prompt caching can cut input cost by more than half, or raise it by a quarter. Which one
you get depends on how your application builds its prompts and how fast requests arrive.
`cacheplan` reads a log of your requests and tells you, per application:

- **where and why** the reusable prefix breaks (a timestamp in the system prompt, tools in
  a different order, new turns inserted before content that was already sent);
- **what each option would cost**: caching off, automatic caching, planned breakpoints, with
  the 5-minute or the 1-hour lifetime, from a local model of the provider's caching rules;
- **what to do**: the cheapest simple option, where to put the breakpoints, and whether a
  small prompt change would unlock more;
- **whether the forecast is right**: an optional replay against the live API, one output
  token per request.

It runs locally, has no dependencies, and needs no API key except for the live check.
Status: early alpha. Anthropic Messages API only.

## Install

```
pip install --pre cacheplan
```

## Use

```
cacheplan report requests.jsonl
```

Example output, abridged (LangChain's ReAct agent template):

```
react-template
  17 requests, ~113,000 input tokens (estimate)
  Layout     changing content at the top of the prompt
  Reusable   2% of input tokens repeat an earlier prefix as written
  Pace       median 4 s between requests; 6% of pauses over 5 minutes

  Option                                Read  Cost vs no caching
  Caching off                             0%                100%  <- recommended
  Automatic caching, 5-minute             0%                125%
  Planned breakpoints, 5-minute           0%                100%
  Prompt fix + breakpoints, 5-minute     62%                 53%

  Recommendation: leave caching off as the prompts stand.
  With the prompt fix in the findings: 53%. It changes what the model reads,
  so compare answer quality before adopting it.

  Findings
    CP001 system value <<system_time>> changes between requests; everything after it,
          including the conversation history, cannot be reused
        fix: move it to the latest user message, or make it change rarely
```

No log yet? Record one. The proxy writes request bodies only; headers, and so your API
key, never reach the file.

```
cacheplan capture --out requests.jsonl --app my-app
# in the shell that runs your application:
#   ANTHROPIC_BASE_URL=http://127.0.0.1:8787
```

| Command | What it does |
| --- | --- |
| `report LOG` | Diagnosis, forecast and recommendation per application. `--json`, `--no-content`, `--fail-on CP001,CP002`. |
| `diagnose LOG -v` | Every prefix break: where, what changed, how many tokens sit behind it. |
| `verify LOG --model M --yes` | Replays the recommended options against the live API and prints forecast next to measured. |
| `gaps LOG` | How the forecast changes as pauses between requests grow. |
| `check LOG --min-reusable 0.5` | For pipelines: exit status 1 when too little of the prompt is reusable. |
| `capture --out LOG` | Logging proxy. |
| `rules` | Rule ids. |

`--learn-from OTHER.jsonl` learns the breakpoint plan from one log and scores another, for
an out-of-sample forecast.

### Log formats

One JSON object per line; formats can be mixed in one file:

- a `cacheplan capture` line;
- a raw Anthropic request body;
- a [promptcachelint](https://github.com/OsmnvAslan/promptcachelint) trace record.

## What it found so far

Measured on 23 November 2026 with one model, short sessions. Cost is input cost relative to
the same prompts sent without caching.

| Application | Layout | Automatic caching | cacheplan | Source |
| --- | --- | --- | --- | --- |
| LangChain ReAct agent template | timestamp in the system prompt | 125% | 51% with the prompt fix | live API |
| Aider | finished turns inserted mid-prompt | 125% | 66% with planned breakpoints (Aider's own caching: 67%) | live API, plan learned on a different session |
| Claude Code | append-only conversation | 27% | 27%: nothing to change | local model |

The local model's forecast was within two points of the live measurement in every comparison.

## Limits

- **Small evidence base.** Three applications, one or two short sessions each.
- **The forecast is a model.** It follows the documented caching rules; token counts are
  estimated at three characters per token. Use `verify` before acting on a close call.
- **The minimum cacheable length varies by model.** The default is 1,024 tokens; set
  `--min-tokens` for yours.
- **The prompt fix changes what the model reads.** It never drops content, but its effect on
  answer quality is not measured. Treat it as a suggestion to implement and test yourself.
- **The 1-hour lifetime has had one live check.** The request was accepted and the read share matched the forecast; its price is taken from the documentation.
- **Plans are learned from the log you give it.** A short or unrepresentative log gives a
  poor plan. Use `--learn-from` to test on a second log.
- **Anthropic only.** OpenAI and others are planned.

## Related tools

[promptcachelint](https://github.com/OsmnvAslan/promptcachelint) and
[cachelint](https://github.com/Diafyx/cachelint) explain why a prompt cache missed, and both
support OpenAI as well. `cacheplan` overlaps with them on diagnosis and adds the forecast,
the breakpoint plan, the lifetime choice and live verification.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

Licence: Apache-2.0.
