# cacheplan documentation

Start with the [project README](../README.md): it has the quick start, the report explained,
the command list and the benchmark. These pages go deeper on single topics.

| Page | What it covers |
| --- | --- |
| [Logs from tracing tools](exports.md) | Reading Langfuse, LangSmith, OpenTelemetry and LiteLLM exports, and what is lost compared with a capture |
| [Grouping](grouping.md) | How requests are grouped into applications and conversations, and the options that change it |
| [Applying a plan](apply.md) | `cacheplan plan` and the experimental wrapper that adds cache markers to live requests |
| [Providers](providers.md) | What is modelled per provider and model family, with sources and evidence |
| [Speed and memory](scaling.md) | Measured run time and memory from 1,000 to 100,000 requests |
| [How it works inside](internals.md) | Segments, renderings and the two cache models |

To browse these as a site locally: `pip install mkdocs && mkdocs serve` from the repository root.
