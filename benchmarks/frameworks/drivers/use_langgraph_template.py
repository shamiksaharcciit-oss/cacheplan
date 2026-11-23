"""The official LangGraph ReAct agent template (github.com/langchain-ai/react-agent),
unmodified: its own graph, system prompt and search tool definition. The Tavily call
behind its tool is replaced by the shared stand-in search so no Tavily key is needed."""
import asyncio
import os
import sys
from pathlib import Path

from scenario import MODEL, SESSIONS, search

TEMPLATE = Path(os.environ.get("REACT_AGENT_DIR", Path(__file__).resolve().parents[1] / "_vendor" / "react-agent"))


class _Search:
    def __init__(self, **_):
        pass

    async def ainvoke(self, args: dict) -> dict:
        return {"query": args.get("query", ""), "results": search(args.get("query", ""))}


async def run():
    sys.path.insert(0, str(TEMPLATE / "src"))
    os.environ.setdefault("MODEL", f"anthropic/{MODEL}")
    if "/" not in os.environ["MODEL"]:
        os.environ["MODEL"] = f"anthropic/{os.environ['MODEL']}"
    import react_agent.tools as tools
    tools.TavilySearch = _Search
    from react_agent.context import Context
    from react_agent.graph import graph
    for questions in SESSIONS:
        messages = []
        for q in questions:
            messages = (await graph.ainvoke({"messages": messages + [("user", q)]}, context=Context()))["messages"]


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
