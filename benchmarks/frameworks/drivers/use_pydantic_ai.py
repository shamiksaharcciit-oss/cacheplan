"""Pydantic AI's Agent."""
from pydantic_ai import Agent

from scenario import INSTRUCTIONS, MODEL, SESSIONS, search


def main():
    agent = Agent(f"anthropic:{MODEL}", instructions=INSTRUCTIONS, tools=[search])
    for questions in SESSIONS:
        history = None
        for q in questions:
            history = agent.run_sync(q, message_history=history).all_messages()


if __name__ == "__main__":
    main()
