# Changelog

## 0.1.0a1

First alpha.

- `report`: per-application diagnosis, cost forecast for each caching option (off, automatic,
  planned breakpoints; 5-minute and 1-hour lifetimes), recommendation and findings CD001 to CD009.
- `diagnose`: prefix breaks with location, cause and tokens behind them.
- `verify`: live replay with one output token per request.
- `gaps`, `check`, `capture`, `rules`.
- Reads cachedoctor captures, raw Anthropic request bodies and promptcachelint traces.
