"""Record requests from inside your application, without a proxy.

Wrap the HTTP transport of the Anthropic SDK's client. Every /v1/messages call is appended
to a JSONL file that `cacheplan report` reads: the request body and the token usage from
the response. Headers and response text are never written.

    import anthropic, httpx
    from cacheplan.record import RecordingTransport

    client = anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(
        transport=RecordingTransport("requests.jsonl", app="my-app")))

For the async client use AsyncRecordingTransport with anthropic.DefaultAsyncHttpxClient.
Requires httpx, which the Anthropic SDK already depends on.
"""
from __future__ import annotations

import json
import threading
import time

import httpx

from .proxy import MAX_KEEP, usage_from_response

_lock = threading.Lock()


def _is_messages(request: httpx.Request) -> bool:
    return request.method == "POST" and request.url.path.rstrip("/").endswith("/v1/messages")


def _write(path: str, app: str, started: float, request: httpx.Request, status: int, kept: bytes, ctype: str) -> None:
    try:
        body = json.loads(request.content)
    except ValueError:
        return
    usage = usage_from_response(kept, ctype) if status == 200 else None
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": started, "app": app, "status": status, "usage": usage, "request": body}) + "\n")


class _Tee(httpx.SyncByteStream):
    def __init__(self, inner, done):
        self.inner, self.done, self.kept = inner, done, bytearray()

    def __iter__(self):
        for chunk in self.inner:
            if len(self.kept) < MAX_KEEP:
                self.kept += chunk
            yield chunk

    def close(self):
        try:
            self.inner.close()
        finally:
            self.done(bytes(self.kept))


class _AsyncTee(httpx.AsyncByteStream):
    def __init__(self, inner, done):
        self.inner, self.done, self.kept = inner, done, bytearray()

    async def __aiter__(self):
        async for chunk in self.inner:
            if len(self.kept) < MAX_KEEP:
                self.kept += chunk
            yield chunk

    async def aclose(self):
        try:
            await self.inner.aclose()
        finally:
            self.done(bytes(self.kept))


def _plain(request: httpx.Request) -> None:
    request.headers["accept-encoding"] = "identity"       # so the usage can be read as it streams


class RecordingTransport(httpx.BaseTransport):
    def __init__(self, path: str, app: str = "app", inner: httpx.BaseTransport | None = None):
        self.path, self.app, self.inner = path, app, inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not _is_messages(request):
            return self.inner.handle_request(request)
        _plain(request)
        started = time.time()
        resp = self.inner.handle_request(request)
        ctype = resp.headers.get("content-type", "")
        done = lambda kept: _write(self.path, self.app, started, request, resp.status_code, kept, ctype)
        return httpx.Response(resp.status_code, headers=resp.headers, stream=_Tee(resp.stream, done),
                              extensions=resp.extensions)

    def close(self) -> None:
        self.inner.close()


class AsyncRecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, path: str, app: str = "app", inner: httpx.AsyncBaseTransport | None = None):
        self.path, self.app, self.inner = path, app, inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not _is_messages(request):
            return await self.inner.handle_async_request(request)
        _plain(request)
        started = time.time()
        resp = await self.inner.handle_async_request(request)
        ctype = resp.headers.get("content-type", "")
        done = lambda kept: _write(self.path, self.app, started, request, resp.status_code, kept, ctype)
        return httpx.Response(resp.status_code, headers=resp.headers, stream=_AsyncTee(resp.stream, done),
                              extensions=resp.extensions)

    async def aclose(self) -> None:
        await self.inner.aclose()
