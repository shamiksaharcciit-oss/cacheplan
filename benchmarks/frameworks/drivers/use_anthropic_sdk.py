"""The plain Anthropic SDK tool loop, as in Anthropic's tool-use documentation."""
import anthropic

from scenario import INSTRUCTIONS, MODEL, SEARCH_DESCRIPTION, SESSIONS, search

TOOLS = [{"name": "search", "description": SEARCH_DESCRIPTION,
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]


def main():
    client = anthropic.Anthropic()
    for questions in SESSIONS:
        messages = []
        for q in questions:
            messages.append({"role": "user", "content": q})
            for _ in range(6):
                resp = client.messages.create(model=MODEL, max_tokens=1024, system=INSTRUCTIONS,
                                              tools=TOOLS, messages=messages)
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    break
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": search(**b.input)}
                    for b in resp.content if b.type == "tool_use"]})


if __name__ == "__main__":
    main()
