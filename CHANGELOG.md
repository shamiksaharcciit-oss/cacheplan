# Changelog

## 0.3.0a2

- **OpenAI GPT-6.2 and later: top-level `instructions` are no longer treated as a cacheable
  system prompt.** Those models append them after the input and bill them uncached on every
  request. They are modelled that way and reported as the new finding `CP010`. Found in an
  external review; forecasts for such requests were too optimistic in 0.3.0a1.
- **The GitHub Action installs the code of the revision it is pinned to**, instead of
  whatever `pip install cacheplan` resolves to. `version` still selects a PyPI release.
- `capture --no-content` (and `no_content=True` on the recorders) scrambles prompt text as it
  is written. Reports still work on such a log. New capture files are owner-readable only.
- `capture` now says at start-up that the log holds everything sent to the model.
- `cacheplan providers` and an "Evidence for these rules" line in reports: each provider's
  rule set carries its source and what live evidence stands behind it.
- A first conformance fixture: the recorded usage of a live gpt-5.6-luna run, replayed in
  the test suite.
- Benchmark and accuracy tables re-measured with one version at the published cache-read
  price; Aider compared like for like with its own caching.
- Workflow actions are pinned to commit SHAs.

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
