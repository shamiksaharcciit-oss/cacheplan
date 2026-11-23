"""CrewAI: one agent, one task per question. CrewAI has no built-in multi-turn chat for a
single agent, so each question is a separate kickoff, as in its quickstart."""
from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool

from scenario import INSTRUCTIONS, MODEL, SESSIONS
from scenario import search as _search


@tool("search")
def search(query: str) -> str:
    """Search the web for up-to-date information. Returns the top results as text."""
    return _search(query)


def main():
    llm = LLM(model=f"anthropic/{MODEL}", max_tokens=1024)
    agent = Agent(role="Research assistant", goal="Answer questions accurately and briefly",
                  backstory=INSTRUCTIONS, tools=[search], llm=llm, verbose=False)
    for questions in SESSIONS:
        for q in questions:
            task = Task(description=q, expected_output="An answer in at most three sentences.", agent=agent)
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()


if __name__ == "__main__":
    main()
