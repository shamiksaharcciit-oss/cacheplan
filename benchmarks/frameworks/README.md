# Framework benchmark

Does switching on prompt caching save money with a framework's default agent?

Each driver in `drivers/` builds that framework's standard agent with the same instructions
and the same search tool (`scenario.py`), and runs the same three three-turn conversations.
The framework assembles every request itself. `bench.py` records the requests through the
cacheplan proxy, forecasts each caching option and can measure them against the live API.

| Driver | What it runs |
| --- | --- |
| `anthropic_sdk` | The hand-written tool loop from Anthropic's tool-use documentation |
| `langchain_agent` | LangChain `create_agent` |
| `langgraph_template` | The official LangGraph ReAct agent template, unmodified, cloned at run time |
| `pydantic_ai` | Pydantic AI `Agent` |
| `smolagents` | smolagents `ToolCallingAgent` through LiteLLM |
| `crewai` | CrewAI `Agent`, one task per question |

## Run it

```
python -m venv .bench && . .bench/bin/activate        # Windows: .bench\Scripts\Activate.ps1
pip install -e ../..                                  # cacheplan itself
pip install -r requirements.txt

python bench.py capture --stand-in     # offline plumbing test: no key, no cost
python bench.py capture                # real run, needs ANTHROPIC_API_KEY
python bench.py results                # forecast table -> results.md, results.json
python bench.py verify --model MODEL   # live measurement -> measured.json and the final table
```

## What is and is not controlled

- **Same for all:** instructions, tool, questions, model.
- **Different by design:** everything the framework adds or decides: its system prompt
  scaffolding, tool schema, message layout, and whether it sets `cache_control` itself.
- **Search results are canned** (`scenario.search`), so no search key is needed and every
  framework sees the same tool output.
- **Conversations are short** (three turns) and requests arrive seconds apart, so nothing
  expires from the cache. Real traffic with long pauses would do worse.
- **Defaults only.** Several frameworks offer caching options or middleware. This measures
  what you get before you find them.
- **One change was needed to run smolagents** on current Claude models (`litellm.drop_params`);
  see its driver.

## Not included

- **LlamaIndex** (`llama-index-llms-anthropic` 0.13.0) and **AutoGen** (`autogen-ext` 0.7.5)
  could not run when this was written (23 November 2026): both always send `temperature`,
  which the current Anthropic SDK and current Claude models reject. They can be added once
  their integrations are updated.
- Agents that execute model-written code locally (such as smolagents `CodeAgent`).
