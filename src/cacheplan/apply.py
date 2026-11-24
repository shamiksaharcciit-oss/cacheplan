"""Apply a saved caching plan to live requests. Experimental.

    cacheplan plan requests.jsonl --save plan.json

    import anthropic
    from cacheplan.apply import PlanTransport

    client = anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(
        transport=PlanTransport("plan.json", app="my-app")))

What it does to each Messages API request, depending on what the plan recommends:

  automatic caching      adds the top-level cache_control field
  planned breakpoints    puts cache_control markers at the planned positions
  off / keep as is       nothing

What it never does: change, move or drop any content. Only cache markers are added or
replaced, so the model reads exactly what your application wrote. (Where a marker goes on
text sent as a plain string, the string becomes a one-block list, which the API treats
the same.) The prompt fix from the
report is deliberately not applied here, because it does change what the model reads.

If anything goes wrong while rewriting, the request is sent as your application built it.
Anthropic Messages API only. Requires httpx.
"""
from __future__ import annotations

import json
from pathlib import Path

from .messages import FIVE_MIN, ONE_HOUR, BreakPlan, _unmarked, cc, place_breakpoints

try:
    import httpx
except ImportError:                                           # plan files can be written without httpx
    httpx = None

_Base = httpx.BaseTransport if httpx else object
_AsyncBase = httpx.AsyncBaseTransport if httpx else object
FORMAT = 1


def save(verdicts, path: str, version: str) -> dict:
    """Write the recommendation for each application to a file. No prompt text is written."""
    data = {"cacheplan": version, "format": FORMAT, "applications": {}}
    for v in verdicts:
        if v.rules.provider != "anthropic":
            continue
        long = v.recommended.endswith("_1h")
        plan = v.plans.get(ONE_HOUR if long else FIVE_MIN)
        data["applications"][v.app] = {
            "action": v.recommended.split("_")[0],            # off | auto | plan | logged
            "lifetime": "1h" if long else "5m",
            "forecast_cost_vs_no_caching": round(v.cost(v.recommended), 4),
            "plan": None if plan is None else {"head": plan.head, "constant_prefix_blocks": plan.const_prefix,
                                               "offsets_from_end": list(plan.offsets)},
            "requests_in_log": v.requests,
        }
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def load(path: str, app: str | None = None) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    apps = data.get("applications") or {}
    if app is None and len(apps) == 1:
        app = next(iter(apps))
    if app not in apps:
        raise ValueError(f"The plan file has {len(apps)} application(s): {', '.join(apps) or 'none'}. "
                         f"Pass app= with one of these names" + (f"; {app!r} is not among them." if app else "."))
    return apps[app]


def rewrite(body: dict, entry: dict) -> dict:
    """The request body with the plan's cache markers. Content and order are untouched."""
    ttl = ONE_HOUR if entry.get("lifetime") == "1h" else FIVE_MIN
    action = entry.get("action")
    if action == "auto":
        out = dict(body, **_unmarked(body))
        out["cache_control"] = cc(ttl)
        return out
    if action == "plan" and entry.get("plan"):
        p = entry["plan"]
        plan = BreakPlan(bool(p.get("head")), int(p.get("constant_prefix_blocks") or 0), tuple(p.get("offsets_from_end") or ()))
        out = dict(body, **place_breakpoints(body, plan=plan, ttl=ttl))
        out.pop("cache_control", None)
        return out
    return body


def _is_messages(request) -> bool:
    return request.method == "POST" and request.url.path.rstrip("/").endswith("/v1/messages")


def _rewritten(request, entry):
    """A new request carrying the rewritten body, or the original if anything fails."""
    try:
        body = json.loads(request.content)
        new = rewrite(body, entry)
        if new is body:
            return request
        content = json.dumps(new).encode()
        headers = [(k, v) for k, v in request.headers.raw if k.lower() not in (b"content-length", b"transfer-encoding")]
        return httpx.Request(request.method, request.url, headers=headers, content=content, extensions=request.extensions)
    except Exception:                                         # never let the wrapper break a request
        return request


class PlanTransport(_Base):
    def __init__(self, plan_file: str, app: str | None = None, inner=None):
        self.entry, self.inner = load(plan_file, app), inner or httpx.HTTPTransport()

    def handle_request(self, request):
        return self.inner.handle_request(_rewritten(request, self.entry) if _is_messages(request) else request)

    def close(self) -> None:
        self.inner.close()


class AsyncPlanTransport(_AsyncBase):
    def __init__(self, plan_file: str, app: str | None = None, inner=None):
        self.entry, self.inner = load(plan_file, app), inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        return await self.inner.handle_async_request(_rewritten(request, self.entry) if _is_messages(request) else request)

    async def aclose(self) -> None:
        await self.inner.aclose()
