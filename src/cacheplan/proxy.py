"""Logging proxy for the Anthropic API.

Point any Anthropic client at it (ANTHROPIC_BASE_URL=http://127.0.0.1:8787) and every
/v1/messages call is appended to a JSONL file: the request body exactly as the client
assembled it, and the token usage the API reported for it. Headers are never written, so
the API key stays out of the log. Response text is not written either. Responses are
passed through to the client unchanged.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "transfer-encoding", "content-length", "keep-alive", "host", "accept-encoding"}
MAX_KEEP = 8 * 1024 * 1024          # response bytes kept in memory to read the usage from


def usage_from_response(body: bytes, content_type: str) -> dict | None:
    """Token usage from a Messages API response, plain JSON or a server-sent event stream."""
    try:
        text = body.decode("utf-8", errors="replace")
        if "event-stream" not in content_type:
            u = json.loads(text).get("usage")
            return u if isinstance(u, dict) else None
        usage: dict = {}
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if ev.get("type") == "message_start":
                usage.update((ev.get("message") or {}).get("usage") or {})
            elif ev.get("type") == "message_delta":
                usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
        return usage or None
    except (ValueError, AttributeError):
        return None


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
            path = self.path.split("?")[0].rstrip("/")
            req = None
            if path.endswith("/v1/messages"):
                try:
                    req = json.loads(body)
                except ValueError:
                    req = None
            status, kept, ctype = self._forward(body)
            if req is not None:
                usage = usage_from_response(kept, ctype) if status == 200 else None
                with lock:
                    counter[0] += 1
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": f"{app}-{counter[0]}", "ts": started, "app": app,
                                            "status": status, "usage": usage, "request": req}) + "\n")
                    print(f"captured {counter[0]} requests", end="\r", flush=True)

    return Handler


def serve(port: int, upstream: str, out: str, app: str) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, out, app))
