# Changelog

## 0.1.0a2

- The minimum cacheable prompt length is looked up from the model named in each logged
  request (or `--model`); `--min-tokens` still overrides it. 0.1.0a1 assumed 1,024 tokens for
  every model.
- Requests that define tools now include an estimate of the tool-use instructions the API
  adds, which count towards the cacheable length.
- The report shows the minimum and the typical prompt size, and warns when prompts sit close
  to the minimum, where the forecast is least reliable.
- New README with diagrams; framework benchmark under `benchmarks/frameworks` with first results.
- CI also builds the package; workflow actions updated.

Known limitation: token counts are estimated from characters, so forecasts for prompts near
the minimum cacheable length can be wrong. 0.1.0a1 missed CrewAI's short prompts this way
(forecast 100%, measured 90%).

## 0.1.0a1

First alpha.

- `report`: per-application diagnosis, cost forecast for each caching option (off, automatic,
  planned breakpoints; 5-minute and 1-hour lifetimes), recommendation and findings CP001 to CP009.
- `diagnose`: prefix breaks with location, cause and tokens behind them.
- `verify`: live replay with one output token per request.
- `gaps`, `check`, `capture`, `rules`.
- Reads cacheplan captures, raw Anthropic request bodies and promptcachelint traces.
