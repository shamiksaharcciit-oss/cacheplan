"""Logging proxy for the Anthropic API.

Point any Anthropic client at it (ANTHROPIC_BASE_URL=http://127.0.0.1:8787) and every
/v1/messages request body is appended to a JSONL file exactly as the client assembled
it. Headers are never written, so the API key stays out of the log. Responses are
passed through unchanged.
"""
from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "transfer-encoding", "content-length", "keep-alive", "host"}


def make_handler(upstream: str, out_path: str, app: str):
    up = urlsplit(upstream)
    lock, counter = threading.Lock(), [0]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _forward(self, body: bytes | None):
            Conn = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
            conn = Conn(up.hostname, up.port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Host"] = up.netloc
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")      # one request per connection: simplest to get right
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            conn.close()
            self.close_connection = True

        def do_GET(self):
            self._forward(None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            path = self.path.split("?")[0].rstrip("/")
            if path.endswith("/v1/messages"):
                try:
                    req = json.loads(body)
                    with lock:
                        counter[0] += 1
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"id": f"{app}-{counter[0]}", "ts": time.time(),
                                                "app": app, "request": req}) + "\n")
                        print(f"captured {counter[0]} requests", end="\r", flush=True)
                except ValueError:
                    pass
            self._forward(body)

    return Handler


def serve(port: int, upstream: str, out: str, app: str) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, out, app))
