"""Record requests from inside your application, without a proxy.

Anthropic, OpenAI and Gemini SDKs: wrap the client's HTTP transport. Every model call is
appended to a JSONL file that `cacheplan report` reads: the request body and the token
usage from the response. Headers and response text are never written. The log does hold
everything sent to the model; pass no_content=True to scramble prompt text as it is written.

    import anthropic, httpx
    from cacheplan.record import RecordingTransport

    client = anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(
        transport=RecordingTransport("requests.jsonl", app="my-app")))

    import openai
    client = openai.OpenAI(http_client=openai.DefaultHttpxClient(
        transport=RecordingTransport("requests.jsonl", app="my-app")))

For async clients use AsyncRecordingTransport. Requires httpx, which those SDKs depend on.

Bedrock (boto3), Converse and ConverseStream:

    import boto3
    from cacheplan.record import record_bedrock

    client = record_bedrock(boto3.client("bedrock-runtime"), "requests.jsonl", app="my-app")
"""
from __future__ import annotations

import json
import threading
import time

from .proxy import MAX_KEEP, classify, entry, new_salt, open_log, scramble, usage_from_response

try:
    import httpx
except ImportError:                                       # only record_bedrock is usable without httpx
    httpx = None

_lock = threading.Lock()
_Sync = httpx.SyncByteStream if httpx else object
_Async = httpx.AsyncByteStream if httpx else object
_Base = httpx.BaseTransport if httpx else object
_AsyncBase = httpx.AsyncBaseTransport if httpx else object


def _kind(request):
    return classify(request.method, request.url.path)


def _append(path: str, row: dict) -> None:
    with _lock, open_log(path) as f:
        f.write(json.dumps(row, default=str) + "\n")


def _write(path: str, app: str, started: float, request, status: int, kept: bytes, ctype: str, kind, salt=None) -> None:
    try:
        body = json.loads(request.content)
    except ValueError:
        return
    if isinstance(body, dict):
        usage = usage_from_response(kept, ctype) if status == 200 else None
        _append(path, entry(app, started, status, usage, scramble(body, salt) if salt else body, *kind))


def record_bedrock(client, path: str, app: str = "app", no_content: bool = False):
    """Record a boto3 bedrock-runtime client's Converse and ConverseStream calls. Returns the
    same client. Bodies sent with InvokeModel are not recorded."""
    pending = threading.local()
    salt = new_salt() if no_content else None

    def before(params, **_):
        pending.value = (time.time(), scramble(dict(params), salt) if salt else dict(params))

    def after(parsed, **_):
        started, params = getattr(pending, "value", (time.time(), {}))
        status = (parsed.get("ResponseMetadata") or {}).get("HTTPStatusCode", 200)
        stream = parsed.get("stream")
        if stream is None:
            _append(path, entry(app, started, status, parsed.get("usage"), params, "bedrock"))
            return

        def relay():
            usage = None
            try:
                for event in stream:
                    usage = (event.get("metadata") or {}).get("usage") or usage
                    yield event
            finally:
                _append(path, entry(app, started, status, usage, params, "bedrock"))
        parsed["stream"] = relay()

    for op in ("Converse", "ConverseStream"):
        client.meta.events.register(f"provide-client-params.bedrock-runtime.{op}", before)
        client.meta.events.register(f"after-call.bedrock-runtime.{op}", after)
    return client


class _Tee(_Sync):
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


class _AsyncTee(_Async):
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


def _plain(request) -> None:
    request.headers["accept-encoding"] = "identity"       # so the usage can be read as it streams


class RecordingTransport(_Base):
    def __init__(self, path: str, app: str = "app", inner=None, no_content: bool = False):
        self.salt = new_salt() if no_content else None
        self.path, self.app, self.inner = path, app, inner or httpx.HTTPTransport()

    def handle_request(self, request):
        kind = _kind(request)
        if not kind:
            return self.inner.handle_request(request)
        _plain(request)
        started = time.time()
        resp = self.inner.handle_request(request)
        ctype = resp.headers.get("content-type", "")
        done = lambda kept: _write(self.path, self.app, started, request, resp.status_code, kept, ctype, kind, self.salt)
        return httpx.Response(resp.status_code, headers=resp.headers, stream=_Tee(resp.stream, done),
                              extensions=resp.extensions)

    def close(self) -> None:
        self.inner.close()


class AsyncRecordingTransport(_AsyncBase):
    def __init__(self, path: str, app: str = "app", inner=None, no_content: bool = False):
        self.salt = new_salt() if no_content else None
        self.path, self.app, self.inner = path, app, inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        kind = _kind(request)
        if not kind:
            return await self.inner.handle_async_request(request)
        _plain(request)
        started = time.time()
        resp = await self.inner.handle_async_request(request)
        ctype = resp.headers.get("content-type", "")
        done = lambda kept: _write(self.path, self.app, started, request, resp.status_code, kept, ctype, kind, self.salt)
        return httpx.Response(resp.status_code, headers=resp.headers, stream=_AsyncTee(resp.stream, done),
                              extensions=resp.extensions)

    async def aclose(self) -> None:
        await self.inner.aclose()
