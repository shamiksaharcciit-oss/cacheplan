"""Logging proxy for the Anthropic, OpenAI and Gemini APIs.

Point a client at it (ANTHROPIC_BASE_URL, OPENAI_BASE_URL or the Gemini SDK's base URL set
to http://127.0.0.1:8787) and every model call is appended to a JSONL file: the request
body exactly as the client assembled it, and the token usage the API reported for it.
Headers are never written, so the API key stays out of the log. Response text is not
written either. Responses are passed through to the client unchanged.

Bedrock requests are signed for their host and cannot go through a proxy; use
cacheplan.record.record_bedrock instead.
"""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "transfer-encoding", "content-length", "keep-alive", "host", "accept-encoding"}
MAX_KEEP = 8 * 1024 * 1024          # response bytes kept in memory to read the usage from
UPSTREAMS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com",
             "gemini": "https://generativelanguage.googleapis.com"}
BASE_URL_HINT = {"anthropic": "ANTHROPIC_BASE_URL=http://127.0.0.1:{port}",
                 "openai": "OPENAI_BASE_URL=http://127.0.0.1:{port}/v1",
                 "gemini": "the Gemini client's base URL (http_options base_url) to http://127.0.0.1:{port}"}
_GEMINI = re.compile(r"/models/([^/:]+):(?:stream)?[gG]enerateContent$")


def classify(method: str, path: str) -> tuple[str, str | None] | None:
    """(provider, model named in the path) for a model call, None for anything else."""
    if method != "POST":
        return None
    path = path.split("?")[0].rstrip("/")
    if path.endswith("/v1/messages"):
        return "anthropic", None
    if path.endswith(("/chat/completions", "/responses")):
        return "openai", None
    m = _GEMINI.search(path)
    if m:
        return "gemini", m.group(1)
    return None


def _usage_in(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    for holder in (obj, obj.get("response"), obj.get("message")):
        if isinstance(holder, dict):
            for key in ("usage", "usageMetadata"):
                if isinstance(holder.get(key), dict):
                    return holder[key]
    return None


def usage_from_response(body: bytes, content_type: str) -> dict | None:
    """Token usage from a model response, plain JSON or a server-sent event stream. Works
    for Anthropic Messages, OpenAI Chat Completions and Responses, and Gemini. Later events
    in a stream add to or replace what earlier ones reported."""
    try:
        text = body.decode("utf-8", errors="replace")
        if "event-stream" not in content_type:
            data = json.loads(text)
            if isinstance(data, list):                    # Gemini streaming without SSE: a JSON array
                usage = {}
                for item in data:
                    usage.update(_usage_in(item) or {})
                return usage or None
            return _usage_in(data)
        usage: dict = {}
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            usage.update({k: v for k, v in (_usage_in(ev) or {}).items() if v is not None})
        return usage or None
    except (ValueError, AttributeError):
        return None


def entry(app: str, started: float, status: int, usage: dict | None, req: dict,
          provider: str, model: str | None = None) -> dict:
    """One log line. Provider and model are written only when the request does not show them."""
    row = {"ts": started, "app": app, "status": status, "usage": usage, "request": req}
    if provider != "anthropic":
        row["provider"] = provider
    if model and not req.get("model"):
        row["model"] = model
    return row


def make_handler(upstream: str, out_path: str, app: str):
    up = urlsplit(upstream)
    lock, counter = threading.Lock(), [0]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _forward(self, body: bytes | None) -> tuple[int, bytes, str]:
            Conn = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
            conn = Conn(up.hostname, up.port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Host"] = up.netloc
            headers["Accept-Encoding"] = "identity"      # so the usage can be read from the response
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")      # one request per connection: simplest to get right
            self.end_headers()
            kept = bytearray()
            while chunk := resp.read1(65536):
                if len(kept) < MAX_KEEP:
                    kept += chunk
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            conn.close()
            self.close_connection = True
            return resp.status, bytes(kept), resp.getheader("content-type") or ""

        def do_GET(self):
            self._forward(None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            started = time.time()
            kind = classify("POST", self.path)
            req = None
            if kind:
                try:
                    req = json.loads(body)
                except ValueError:
                    req = None
            status, kept, ctype = self._forward(body)
            if isinstance(req, dict):
                usage = usage_from_response(kept, ctype) if status == 200 else None
                with lock:
                    counter[0] += 1
                    row = entry(app, started, status, usage, req, *kind)
                    row = {"id": f"{app}-{counter[0]}", **row}
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"captured {counter[0]} requests", end="\r", flush=True)

    return Handler


def serve(port: int, upstream: str, out: str, app: str) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, out, app))
