# Changelog

## 0.3.0a1

- **OpenAI, Bedrock and Gemini logs.** Chat Completions, Responses, Converse and
  generateContent requests are detected and read alongside Anthropic ones. Diagnosis and
  findings work for all of them.
- **Forecasts under each provider's own rules.** Bedrock Claude models use `cachePoint`
  blocks and Bedrock's minimums. OpenAI GPT-5.6 and later use 30-minute and 12-hour
  lifetimes with 1.25x and 2.5x writes. OpenAI GPT-5.5 and earlier are modelled as automatic
  prefix caching with no write fee. Gemini gets diagnosis only: its documentation does not
  state the discount or lifetime.
- `capture --provider openai|gemini`, the recording transport for the OpenAI SDK, and
  `record_bedrock` for boto3. Usage is read in every provider's format.
- Not checked against a live OpenAI, Bedrock or Gemini API yet. `verify` and `quality`
  remain Anthropic only.

## 0.2.0a1

- **Real token counts.** The capture proxy now records the usage block of each response.
  When a log has usage, the token estimate is calibrated on it, and the report prints a
  forecast check against what the provider billed.
- **Model prices.** The cache-read price is looked up per model (0.05x on claude-sonnet-5-5,
  for example; earlier versions assumed 0.1x everywhere, which overstated costs for options
  that read from the cache). Input prices are looked up too, so the report shows money;
  `--price` overrides.
- **Tool-use overhead per model**, from Anthropic's published figures.
- **Breakpoint plans are chosen by replaying the log** with its real timing, separately for
  the 5-minute and 1-hour lifetimes, instead of by a reuse-rate rule.
- New commands: `quality` (does the prompt fix change answers, against a control) and
  `baseline` (save and compare a content-free summary).
- `cacheplan.record`: record from inside an application through the SDK's HTTP transport.
- `report --html` writes a self-contained report. A GitHub Action is included (`action.yml`).
- The benchmark script writes its results table into the benchmark README.

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
