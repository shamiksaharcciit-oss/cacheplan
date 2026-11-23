"""Hugging Face smolagents' ToolCallingAgent, through LiteLLM.

One setting is changed so that it runs at all on current Claude models: smolagents asks for
tool_choice="required", which LiteLLM refuses for these models unless drop_params is on."""
import litellm
from smolagents import LiteLLMModel, ToolCallingAgent, tool

from scenario import INSTRUCTIONS, MODEL, SESSIONS
from scenario import search as _search


litellm.drop_params = True


@tool
def search(query: str) -> str:
    """Search the web for up-to-date information. Returns the top results as text.

    Args:
        query: What to search for.
    """
    return _search(query)


def main():
    for questions in SESSIONS:
        agent = ToolCallingAgent(tools=[search], model=LiteLLMModel(model_id=f"anthropic/{MODEL}"),
                                 instructions=INSTRUCTIONS, max_steps=6)
        for i, q in enumerate(questions):
            agent.run(q, reset=(i == 0))


if __name__ == "__main__":
    main()
