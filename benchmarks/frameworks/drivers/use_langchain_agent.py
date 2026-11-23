"""LangChain's create_agent (the prebuilt LangGraph ReAct agent)."""
from langchain.agents import create_agent

from scenario import INSTRUCTIONS, MODEL, SESSIONS, search


def main():
    agent = create_agent(model=f"anthropic:{MODEL}", tools=[search], system_prompt=INSTRUCTIONS)
    for questions in SESSIONS:
        messages = []
        for q in questions:
            messages = agent.invoke({"messages": messages + [{"role": "user", "content": q}]})["messages"]


if __name__ == "__main__":
    main()
