"""A stand-in for the Anthropic API, for testing the drivers offline. It answers every
question by calling the search tool once and then replying with fixed text. It supports
both plain and streaming responses. It exists to test plumbing; its answers mean nothing.

    python stand_in_server.py 8799
"""
from __future__ import annotations

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _decide(req: dict) -> tuple[list[dict], str]:
    last = req["messages"][-1]["content"]
    had_result = isinstance(last, list) and any(b.get("type") == "tool_result" for b in last)
    if not had_result:                                  # some frameworks pass tool output back as text
        text = last if isinstance(last, str) else " ".join(b.get("text", "") for b in last if b.get("type") == "text")
        had_result = "Observation" in text or "https://example.com/" in text
    tools = {t["name"]: t for t in req.get("tools") or []}

    def call(name, value):
        props = list((tools[name].get("input_schema") or {}).get("properties") or {"query": {}})
        return [{"type": "text", "text": "Working on it."},
                {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": name,
                 "input": {props[0]: value}}], "tool_use"

    searcher = next((n for n in tools if "search" in n.lower()), None)
    if searcher and not had_result:
        return call(searcher, "stand-in query")
    if "final_answer" in tools:
        return call("final_answer", "A stand-in answer.")
    return [{"type": "text", "text": "A stand-in answer in a sentence or two. " * 4}], "end_turn"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.split("?")[0].endswith("count_tokens"):
            return self._send(json.dumps({"input_tokens": 100}).encode(), "application/json")
        content, stop = _decide(req)
        usage = {"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0}
        msg = {"id": "msg_" + uuid.uuid4().hex[:20], "type": "message", "role": "assistant",
               "model": req.get("model", "stand-in"), "content": content, "stop_reason": stop,
               "stop_sequence": None, "usage": usage}
        if not req.get("stream"):
            return self._send(json.dumps(msg).encode(), "application/json")
        ev = []

        def emit(name, data):
            ev.append(f"event: {name}\ndata: {json.dumps(dict(data, type=name))}\n\n")

        emit("message_start", {"message": dict(msg, content=[], stop_reason=None)})
        for i, block in enumerate(content):
            if block["type"] == "text":
                emit("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}})
                emit("content_block_delta", {"index": i, "delta": {"type": "text_delta", "text": block["text"]}})
            else:
                emit("content_block_start", {"index": i, "content_block": dict(block, input={})})
                emit("content_block_delta", {"index": i, "delta": {"type": "input_json_delta",
                                                                   "partial_json": json.dumps(block["input"])}})
            emit("content_block_stop", {"index": i})
        emit("message_delta", {"delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 20}})
        emit("message_stop", {})
        self._send("".join(ev).encode(), "text/event-stream")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    print(f"stand-in Anthropic API on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
