# Speed and memory

## Measured

`python benchmarks/scale/scale.py`, on a 2-core cloud machine (Intel Xeon 2.1 GHz, Python
3.13, Linux), 24 November 2026. Each size runs in a fresh process.

| Requests | Log size | Load | Learn and plan | Forecast and diagnose | Findings | Total | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 10 MB | 0.2 s | 0.1 s | 0.4 s | 0.0 s | 0.7 s | 52 MB |
| 10,000 | 104 MB | 1.6 s | 0.3 s | 4.3 s | 0.0 s | 6.3 s | 294 MB |
| 100,000 | 1,045 MB | 18.5 s | 2.2 s | 47.3 s | 0.3 s | 68.4 s | 2,743 MB |

The log is scripted agent traffic from the test corpus: three applications with tools, long
system prompts and growing conversations, about 3,500 input tokens (10 KB) per request,
spread over many conversations. One run per size; expect some variation.

## What this covers

Everything `cacheplan report` does: reading the log, learning templates and breakpoint
plans, simulating every caching option, diagnosing prefix breaks, and the findings.

- **Time is linear in the number of requests**, at about 0.7 ms per request here.
- **Memory is about two and a half times the log's size on disk.** The whole log is held in
  memory. Blocks that repeat across requests (a tool definition, an earlier turn of a
  conversation) are stored once, so logs with long shared histories need less.
- **The plan search does not grow with the log.** It uses the first 300 requests of each
  application.

## What it does not cover

- **Your traffic.** Requests with very large prompts, or a log where nothing repeats, will
  differ. Run the script, or time `cacheplan report` on your own log.
- **Windows and macOS.** Measured on Linux only. The test suite took about four times longer
  on one Windows laptop than on this machine, so allow for a similar factor.
- **Logs larger than memory.** There is no streaming mode. Beyond a few hundred thousand
  requests, analyse a sample: one day, or one application.
- **`verify` and `quality`.** Those are bounded by the API, not by this tool.

## Before and after

The first measurement, before requests were rendered once and repeated blocks were shared,
took 3.0 s and 57 MB for 1,000 requests and 11.9 s and 215 MB for 5,000. The same sizes now
take 0.7 s and about 3 s. The two changes are described in [How it works inside](internals.md).
