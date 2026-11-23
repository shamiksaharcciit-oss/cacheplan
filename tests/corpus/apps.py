"""Three small applications that assemble Anthropic requests the way application
code commonly does, including habits that break the cache. They are stand-ins for
real traffic: the structure is the point, the conversations are scripted.

  support_agent  tools built in arbitrary order; tenant name, clock and session id in
                 the first line of the system prompt; retrieved articles placed before
                 the long static policy; brand voice at the end.
  sql_analyst    user name in the first line; retrieved few-shot examples placed before
                 the static rules and schema.
  code_agent     well ordered, except for an environment footer (clock, git status) at
                 the end of the system prompt, rebuilt on every step.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import content as c

_SRC = sorted((Path(__file__).resolve().parents[2] / "src" / "cacheplan").glob("*.py"))


def _stamp(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    return f"2026-11-23T{9 + h:02d}:{rem // 60:02d}:{rem % 60:02d}Z"


def _uuid(rng) -> str:
    x = f"{rng.getrandbits(128):032x}"
    return f"{x[:8]}-{x[8:12]}-4{x[13:16]}-a{x[17:20]}-{x[20:]}"


def _zipf(rng, n, s=1.0):
    return rng.choices(range(n), [1 / (i + 1) ** s for i in range(n)])[0]


def support_session(rng: random.Random, t: float):
    tenant = list(c.TENANTS)[_zipf(rng, len(c.TENANTS))]
    session, msgs = _uuid(rng), []

    def request(now):
        picks = []
        while len(picks) < 3:
            d = c.KB[_zipf(rng, len(c.KB))]
            if d not in picks:
                picks.append(d)
        tools = c.SUPPORT_TOOLS[:]
        rng.shuffle(tools)                                   # e.g. built from a set or dict
        system = (f"You are the support assistant for {tenant}. Current time: {_stamp(now)}. Session: {session}\n\n"
                  "Relevant help-centre articles:\n"
                  + "\n".join(f"[{i}] {title}\n{body}" for i, title, body in picks)
                  + "\n\n" + c.SUPPORT_POLICY
                  + f"\nBrand voice for {tenant}:\n{c.TENANTS[tenant]}")
        return {"system": system, "tools": tools, "messages": json.loads(json.dumps(msgs))}

    for _ in range(rng.randint(2, 5)):
        oid = f"NB-{rng.randint(200000, 299999)}"
        msgs.append({"role": "user", "content": rng.choice(c.SUPPORT_QUESTIONS).format(oid=oid)})
        yield t, request(t)
        if rng.random() < 0.5:
            tid = f"toolu_{rng.getrandbits(80):020x}"
            msgs.append({"role": "assistant", "content": [
                {"type": "text", "text": "Let me look that up."},
                {"type": "tool_use", "id": tid, "name": "lookup_order",
                 "input": {"order_id": oid, "include_events": True}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": json.dumps(
                {"order_number": oid, "status": rng.choice(["paid", "shipped", "delivered"]),
                 "carrier": rng.choice(["PostNL", "DHL", "DPD"]), "promised_on": "2026-11-21",
                 "items": [{"sku": f"SKU-{rng.randint(1000, 9999)}", "name": "Commuter helmet M", "qty": 1}],
                 "events": [{"at": "2026-11-20T07:12:00Z", "scan": "sorted at depot"},
                            {"at": "2026-11-19T16:40:00Z", "scan": "handed to carrier"}]})}]})
            t += rng.uniform(2, 6)
            yield t, request(t)
        msgs.append({"role": "assistant", "content": rng.choice(c.SUPPORT_REPLIES)})
        t += rng.uniform(25, 110)


def sql_session(rng: random.Random, t: float):
    user, role = rng.choice(c.SQL_USERS)
    msgs = []
    for turn in range(rng.randint(1, 3)):
        q = rng.choice(c.SQL_QUESTIONS if turn == 0 else c.SQL_FOLLOWUPS)
        msgs.append({"role": "user", "content": q})
        ex = rng.sample(c.SQL_EXAMPLES, 3)                   # "most similar" examples, in score order
        system = (f"Today is 2026-11-23. You are helping {user} ({role}).\n\n"
                  "Similar past questions:\n"
                  + "\n".join(f"Q: {a}\nSQL: {b}" for a, b in ex)
                  + "\n\n" + c.SQL_RULES + "\nSchema\n\n" + c.schema_ddl())
        yield t, {"system": system, "messages": json.loads(json.dumps(msgs))}
        msgs.append({"role": "assistant", "content": "```sql\nSELECT 1;\n```\nAssumed the current calendar year."})
        t += rng.uniform(40, 200)


def code_session(rng: random.Random, t: float):
    msgs = [{"role": "user", "content": rng.choice(c.CODE_TASKS)}]
    changed = 0
    for _ in range(rng.randint(4, 8)):
        system = (c.CODE_SYSTEM + "\nEnvironment\nWorkspace: /work/project (branch main)\n"
                  f"Local time: {_stamp(t)}\nGit status: {changed} files changed")
        yield t, {"system": system, "tools": c.CODE_TOOLS, "messages": json.loads(json.dumps(msgs))}
        f = rng.choice(_SRC)
        tid = f"toolu_{rng.getrandbits(80):020x}"
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"I'll read {f.name} next."},
            {"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": f"src/cacheplan/{f.name}"}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                  "content": f.read_text(encoding="utf-8")[:3200]}]})
        changed += rng.random() < 0.3
        t += rng.uniform(4, 20)


APPS = {"support_agent": (support_session, 0.5), "sql_analyst": (sql_session, 0.3),
        "code_agent": (code_session, 0.2)}


def generate(n_sessions: int = 60, seed: int = 7, span_s: float = 7200.0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    names, weights = list(APPS), [w for _, w in APPS.values()]
    for s in range(n_sessions):
        app = rng.choices(names, weights)[0]
        for t, req in APPS[app][0](rng, rng.uniform(0, span_s)):
            rows.append({"ts": round(t, 2), "app": app, "session": f"s{s:03d}", "request": req})
    rows.sort(key=lambda r: r["ts"])
    for i, r in enumerate(rows):
        r["id"] = f"req-{i:04d}"
    return rows
