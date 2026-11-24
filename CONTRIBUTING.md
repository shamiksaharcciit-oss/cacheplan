# Contributing

Thanks for looking. The most useful contributions right now are evidence, not code: a log
where the forecast and the bill disagree teaches the project more than a new feature.

## Send evidence

**A forecast that was wrong.** Capture with usage recorded, run `cacheplan report`, and open
an issue with the "Forecast check" line and the provider and model. To share the log itself
without sharing your prompts:

```
cacheplan capture --out requests.jsonl --no-content
```

Every word is scrambled with a key that is never stored; structure, lengths and what
repeats are kept, which is all the analysis needs. Still look at the file before you attach
it: field names, model names and numbers of digits are visible.

**An export that does not load.** For Langfuse, LangSmith, OpenTelemetry or LiteLLM, one
record is enough. Replace the message text with placeholders of similar length.

**A provider rule that is out of date.** Link the documentation page and quote the sentence.

## Set up

```
git clone https://github.com/shamiksaharcciit-oss/cacheplan
cd cacheplan
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

Python 3.10 or later. The package has no runtime dependencies and should stay that way;
`httpx` is optional and only for the in-process recorder and the plan wrapper.

## Ground rules for changes

- **Every provider rule carries its source and its evidence.** Rules live in
  `src/cacheplan/providers.py`. If you change one, update `_evidence()` honestly
  ("documentation only" is a fine answer) and regenerate the ledger page:
  `cacheplan providers --markdown > docs/providers.md`. A test fails if the page is stale.
- **A rule changed because of a live run gets a conformance fixture.** Put the recorded
  usage (token counts only, no prompt text) under `tests/conformance/` with a test that
  replays it. Say in the test whether the run is the one the rule was fitted to.
- **Numbers in the README are measured, and say how.** Do not paste a forecast as a result.
  If a figure was produced by an older version, re-measure it or label it.
- **Nothing sensitive in the repository.** No captures, no API keys, no usage files from
  real accounts beyond conformance fixtures. `.gitignore` covers `*.jsonl` and `captures/`.
- **State what was not tested.** In the pull request, in the changelog and in the docs.

## Where things are

| Path | What |
| --- | --- |
| `src/cacheplan/providers.py` | Request formats, caching rules and usage fields per provider |
| `src/cacheplan/readers.py` | Exports from tracing tools |
| `src/cacheplan/messages.py` | Segments and renderings, template learning, breakpoint placement, the prompt fix |
| `src/cacheplan/sim.py` | The two cache models |
| `src/cacheplan/forecast.py` | Calibration, plan search, options, recommendation |
| `src/cacheplan/diagnose.py`, `findings.py` | Prefix breaks and the CP rules |
| `src/cacheplan/proxy.py`, `record.py`, `apply.py` | Capture proxy, in-process recorders, plan wrapper |
| `benchmarks/` | Framework benchmark, provider drivers, scaling benchmark |
| `docs/` | Topic pages; [docs/internals.md](docs/internals.md) explains the design |

## Releasing (maintainers)

1. Bump `__version__` in `src/cacheplan/__init__.py` and add a changelog section.
2. Merge to `main` and wait for CI.
3. `git tag vX.Y.Z && git push origin vX.Y.Z`. The release workflow publishes to PyPI on a
   pushed `v*` tag or a published GitHub Release, whichever comes first; the other is a
   no-op.
4. Check `pip install cacheplan==X.Y.Z` in a fresh environment.
