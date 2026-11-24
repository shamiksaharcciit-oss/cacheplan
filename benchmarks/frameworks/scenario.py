"""The shared scenario: every framework gets the same instructions, the same search tool
and the same three conversations. Only the framework's own request assembly differs."""
from __future__ import annotations

import os

MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")

INSTRUCTIONS = ("You are a research assistant. Use the search tool to look up facts before "
                "answering a factual question, then answer in at most three sentences.")

SESSIONS = [
    ["What is the Uber H3 spatial index and what is it used for?",
     "How does it compare with Google S2?",
     "Which of the two would you pick for a property search map?"],
    ["What are the main differences between Apache Kafka and Azure Event Hubs?",
     "Does Event Hubs support log compaction?",
     "Summarise what you found as three bullet points."],
    ["What does the EU AI Act say about high-risk AI systems?",
     "When do those obligations start to apply?",
     "How is ISO/IEC 42001 related to it?"],
]

LONG_SESSIONS = [
    ["What is the Uber H3 spatial index and what is it used for?",
     "How does it compare with Google S2?",
     "What resolutions does H3 offer and how large are the cells?",
     "How would you store H3 indexes in PostgreSQL?",
     "What about range queries around a point?",
     "Is there a way to handle the pentagon cells?",
     "How do S2 cell ids compare for range scans?",
     "Which of the two has better library support in Python?",
     "Which would you pick for a property search map?",
     "Summarise the decision as three bullet points."],
    ["What are the main differences between Apache Kafka and Azure Event Hubs?",
     "Does Event Hubs support log compaction?",
     "How do consumer groups differ between the two?",
     "What are the retention limits on each?",
     "How does exactly-once delivery work in Kafka?",
     "Is there an equivalent in Event Hubs?",
     "How do the two compare on schema registries?",
     "What does migration from Kafka to Event Hubs involve?",
     "What are the cost drivers on each side?",
     "Summarise what you found as three bullet points."],
]
if os.environ.get("SCENARIO") == "long":
    SESSIONS = LONG_SESSIONS

SEARCH_DESCRIPTION = "Search the web for up-to-date information. Returns the top results as text."

_BODY = ("This page discusses {q}. It covers the background, the main design choices, typical use "
         "cases, known limitations and how it compares with the common alternatives. Practitioners "
         "usually point to the documentation, published benchmarks and migration guides. ")


def search(query: str) -> str:
    """Search the web for up-to-date information. Returns the top results as text."""
    return "\n\n".join(f"[{i + 1}] Result {i + 1} for {query}\nhttps://example.com/{i + 1}\n"
                       + (_BODY * 3).format(q=query) for i in range(5))
