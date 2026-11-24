# How it works inside

## One rendering per request

Every request, whatever API it was written for, is converted to one shape (`tools`,
`system`, `messages`, plus an optional uncached tail) and then *rendered* once:

```
Rendered
  segments     one per content block, in the order the provider hashes them
  hashes       hashes[i] identifies the prefix that ends with segment i
  cum          estimated tokens up to and including segment i
  tail_tokens  billed after the conversation and never cached
  has_tools    whether the provider adds its tool-use instructions

Segment
  path         system[0], messages[8].content[0], ...
  text         the block, serialised without cache markers
  digest       equal digests mean equal blocks
  tokens       estimate before calibration
  marker       lifetime of the breakpoint on this block, or 0
  unit         the message it belongs to: tools, system, messages[i]
```

A caching option is then nothing more than a list of breakpoint positions over the same
rendering: none for "off", the last segment for "automatic", the planned positions for
"planned breakpoints", the request's own markers for "as logged". Only the prompt fix
builds a new request, because it is the only option that changes content.

This is why a 100,000-request log is practical: blocks are serialised and hashed once, not
once per option, and blocks repeated across a log (a tool definition, an earlier turn) are
stored once.

## Two cache models

| Model | Used for | Rule |
| --- | --- | --- |
| `CacheSim` | Anthropic, Bedrock, OpenAI GPT-5.6 and later | A write happens at a breakpoint. A lookup at a breakpoint checks that position and up to 20 earlier ones. Entries live for their lifetime and are refreshed by reads. Prefixes under the model's minimum are not cached. On OpenAI, a write also makes earlier message endings readable |
| `ImplicitSim` | OpenAI GPT-5.5 and earlier, Gemini | No markers and no write fee. A request reads the longest prefix it shares with any request seen within the lifetime, in the model's token steps |

What sits outside both: the uncached tail. OpenAI GPT-6.2 and later place top-level
`instructions` after the input and bill them in full every time; they are kept out of the
prefix and added to the uncached count.

## Token counts

Estimated at 3.5 characters per token, plus a per-model allowance for the tool-use
instructions. When the log holds the provider's usage, a scale and an overhead are fitted to
it by least squares and the estimate is corrected. The report says which of the two it used.

## Plans

Candidate breakpoint positions come from where prefixes were seen to repeat. Every
combination of the prompt head, the constant prefix and up to two learned positions is
replayed against the first 300 requests with their real timing, once per cache lifetime,
and the cheapest wins if it beats the simpler plan by at least 0.5%. It is a bounded
search, not a global optimum.
