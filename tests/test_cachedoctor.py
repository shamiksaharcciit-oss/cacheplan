import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cacheplan import cli, live
from cacheplan.diagnose import find_breaks
from cacheplan.findings import findings
from cacheplan.forecast import build, examine, learn_from, simulate
from cacheplan.logs import by_app, load
from cacheplan.messages import (ONE_HOUR, apply_template, canonicalize_request, learn,
                                  learn_breakpoints, place_breakpoints, prefix_blocks, system_text)
from cacheplan.sim import CacheSim
from tests.corpus.apps import generate

BIG = "policy " * 2000


# ------------------------------------------------------------------ fixtures
def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def corpus_rows():
    return generate(40, seed=5)


@pytest.fixture()
def corpus_log(tmp_path, corpus_rows):
    return _write(tmp_path, "corpus.jsonl", corpus_rows)


def inserting_agent(n=6, edit=True):
    """Few-shot examples first; each finished turn is inserted before the file contents,
    which are re-sent (and usually edited) every time. The layout Aider uses."""
    out, hist = [], []
    for i in range(n):
        msgs = [{"role": "user", "content": "example question " * 200}, {"role": "assistant", "content": "example answer"}]
        for q, a in hist:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs += [{"role": "user", "content": "FILES\n" + f"file body v{i if edit else 0} " * 400},
                 {"role": "assistant", "content": "Ok."},
                 {"role": "user", "content": f"task {i}" + "\n\nREMINDER " * 20}]
        out.append({"ts": i * 10.0, "app": "inserter", "request": {"system": "Act as an expert developer. " * 200, "messages": msgs}})
        hist = hist + [(f"task {i}", f"done {i} " * 50)]
    return out


def appending_agent(n=8, gap=5.0):
    """A plain conversation: every request is the previous one plus new turns."""
    out, msgs = [], []
    for i in range(n):
        msgs = msgs + [{"role": "user", "content": f"question {i} " * 120}]
        out.append({"ts": i * gap, "app": "appender", "request": {"system": BIG, "messages": list(msgs)}})
        msgs = msgs + [{"role": "assistant", "content": f"answer {i} " * 120}]
    return out


def timestamp_agent(n=8):
    """The append-only agent, with a clock in the first line of the system prompt."""
    rows = appending_agent(n)
    for i, r in enumerate(rows):
        r["app"] = "clock"
        r["request"]["system"] = f"You are helpful.\n\nSystem time: 2026-11-23T10:00:{i:02d}.123+00:00\n\n" + BIG
    return rows


# ------------------------------------------------------------------ reading logs
def test_reads_captures_raw_bodies_and_promptcachelint_traces(tmp_path):
    req = {"model": "m", "system": "s", "messages": [{"role": "user", "content": "hi"}]}
    path = _write(tmp_path, "mixed.jsonl", [
        {"ts": 1, "app": "a", "request": req},
        req,
        {"provider": "anthropic", "body": req, "at": 3.0, "usage": {"input_tokens": 5}},
        {"provider": "openai", "body": {"messages": []}},
        {"unrelated": True},
    ])
    log = load(path)
    assert len(log.rows) == 3
    assert sum(log.skipped.values()) == 2


def test_different_system_prompts_under_one_label_are_split(tmp_path):
    rows = inserting_agent() + [
        {"app": "inserter", "ts": 100 + i, "request": {"system": "Write a one-line commit message for this diff.",
                                                         "messages": [{"role": "user", "content": f"diff {i}"}]}}
        for i in range(3)]
    assert len(by_app(load(_write(tmp_path, "l.jsonl", rows)).rows)) == 2


# ------------------------------------------------------------------ cache model
def _req(system, *turns):
    return {"system": system, "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                                           for i, t in enumerate(turns)]}


def test_nothing_is_cached_without_a_breakpoint():
    sim = CacheSim()
    sim.request(_req(BIG, "q"), 0)
    assert sim.request(_req(BIG, "q"), 1).cache_read_input_tokens == 0


def test_five_minute_entries_expire_and_reads_refresh_them():
    sim, r = CacheSim(), place_breakpoints(_req(BIG, "q"))
    sim.request(r, 0)
    assert sim.request(r, 200).cache_read_input_tokens > 0
    assert sim.request(r, 450).cache_read_input_tokens > 0
    assert sim.request(r, 800).cache_read_input_tokens == 0


def test_one_hour_entries_outlive_a_long_pause_and_cost_double_to_write():
    sim, r = CacheSim(), place_breakpoints(_req(BIG, "q"), ttl=ONE_HOUR)
    first = sim.request(r, 0)
    assert first.cache_creation_1h_tokens == first.cache_creation_input_tokens > 0
    assert first.cost_units() == pytest.approx(first.input_tokens + 2 * first.cache_creation_input_tokens)
    assert sim.request(r, 1800).cache_read_input_tokens > 0
    assert sim.request(r, 1800 + 3700).cache_read_input_tokens == 0


def test_a_changed_first_line_loses_everything():
    sim = CacheSim()
    sim.request(place_breakpoints(_req("time 09:00\n" + BIG, "q")), 0)
    assert sim.request(place_breakpoints(_req("time 09:01\n" + BIG, "q")), 1).cache_read_input_tokens == 0


def test_prefix_below_minimum_is_not_cached():
    sim, small = CacheSim(min_tokens=1024), place_breakpoints(_req("short system", "q"))
    sim.request(small, 0)
    u = sim.request(small, 1)
    assert u.cache_read_input_tokens == 0 and u.cache_creation_input_tokens == 0


# ------------------------------------------------------------------ diagnosis
def test_diagnosis_names_the_timestamp():
    breaks = find_breaks("clock", timestamp_agent())
    assert breaks and all(b.kind == "changed" and b.path == "system[0]" for b in breaks)
    assert breaks[0].cause == "timestamp" and "2026-11-23" in breaks[0].was


def test_diagnosis_sees_mid_prompt_insertion():
    kinds = {b.kind for b in find_breaks("inserter", inserting_agent())}
    assert kinds == {"inserted"}


def test_a_plain_conversation_has_no_breaks():
    assert find_breaks("appender", appending_agent()) == []


def test_a_new_conversation_is_not_reported_as_a_break():
    rows = appending_agent(3) + [{"ts": 99, "app": "appender", "request": _req(BIG, "something else")}]
    assert find_breaks("appender", rows) == []


def test_tool_order_is_recognised(corpus_rows):
    rs = by_app(corpus_rows)["support_agent"]
    assert "tool order" in {b.kind for b in find_breaks("support_agent", rs)}


# ------------------------------------------------------------------ forecast and recommendation
def test_append_only_layout_gets_automatic_caching():
    v = examine(appending_agent())[0]
    assert v.layout == "append-only conversation"
    assert v.recommended == "auto_5m" and v.cost("auto_5m") < 0.5


def test_inserting_layout_gets_planned_breakpoints():
    v = examine(inserting_agent())[0]
    assert v.layout == "content inserted or changed mid-prompt"
    assert v.recommended == "plan_5m"
    assert v.cost("plan_5m") < v.cost("auto_5m") - 0.2
    assert v.plan.offsets == (3,) and v.plan.const_prefix == 2


def test_timestamp_layout_recommends_off_and_offers_the_fix():
    rows = timestamp_agent()
    v = examine(rows)[0]
    assert v.layout == "changing content at the top of the prompt"
    assert v.recommended == "off" and v.cost("auto_5m") > 1.2
    assert v.fix_option == "fix_5m" and v.cost("fix_5m") < 0.6
    rules = {f.rule for f in findings(v, rows, learn_from(rows))}
    assert {"CP001", "CP003"} <= rules


def test_slow_append_only_traffic_prefers_the_one_hour_lifetime():
    v = examine(appending_agent(n=10, gap=420.0))[0]        # seven minutes between requests
    assert v.recommended.endswith("_1h")
    assert v.cost("auto_1h") < v.cost("auto_5m")


def test_plan_learned_on_one_session_transfers_to_another():
    k = learn_from(inserting_agent(6))
    other = inserting_agent(8)
    for i, r in enumerate(other):
        r["ts"] += 1000
    v = examine(other, k)[0]
    assert not v.in_sample and v.recommended == "plan_5m" and v.cost("plan_5m") < 0.8


def test_plan_skips_breakpoints_that_are_rarely_read_back():
    plan = learn_breakpoints([r["request"] for r in inserting_agent()])
    assert plan.offsets == (3,) and plan.reuse["end-3"] == 1.0


def test_good_existing_settings_are_kept():
    rows = appending_agent()
    for r in rows:
        r["request"] = place_breakpoints(r["request"])
    v = examine(rows)[0]
    assert v.recommended == "logged"


# ------------------------------------------------------------------ prompt fix
def test_fix_loses_no_system_content(corpus_rows):
    k = learn_from(corpus_rows)
    for row in corpus_rows:
        original = system_text(row["request"])
        static, shared, volatile = apply_template(k.templates[row["app"]], original)
        restored = static
        for name, value in shared + volatile:
            restored = restored.replace(f"<<{name}>>", value)
        for s in k.templates[row["app"]].slots():
            restored = restored.replace(f"<<{s.name}>>", "")
        assert restored == original
        sent = build("fix_5m", row, k)
        body = "\n".join(b["text"] for b in sent["system"]) + "\n".join(
            b.get("text", "") for b in sent["messages"][-1]["content"])
        for _, value in shared + volatile:
            assert value in body


def test_fix_keeps_messages_tools_and_api_rules(corpus_rows):
    k = learn_from(corpus_rows)
    for row in corpus_rows:
        c = build("fix_1h", row, k)
        assert sorted(t["name"] for t in c.get("tools", [])) == sorted(t["name"] for t in row["request"].get("tools", []))
        assert [m["role"] for m in c["messages"]] == [m["role"] for m in row["request"]["messages"]]
        assert 1 <= sum(1 for _, ttl in prefix_blocks(c) if ttl) <= 4
        for m in c["messages"]:
            kinds = [b["type"] for b in m["content"]]
            if "tool_result" in kinds:
                assert kinds[0] == "tool_result"


def test_fix_refuses_to_move_most_of_a_system_prompt():
    bodies = ["Act as a careful code reviewer and list every defect you can find in the diff.\n" * 8,
              "Summarise the customer conversation in two neutral sentences for the CRM record.\n" * 8,
              "Translate the incoming message to Dutch, keeping product names untouched.\n" * 8]
    texts = [bodies[i % 3] + "One shared closing line here." for i in range(9)]
    req = {"system": texts[0], "messages": [{"role": "user", "content": "hi"}]}
    assert canonicalize_request(req, learn(texts))["system"][0]["text"] == texts[0]


def test_value_that_changes_inside_a_conversation_is_not_shared(corpus_rows):
    k = learn_from(corpus_rows)
    assert {s.name: s.kind for s in k.templates["code_agent"].slots()}["git_status"] == "volatile"


# ------------------------------------------------------------------ command line
def test_report_json_and_fail_on(tmp_path, capsys):
    path = _write(tmp_path, "clock.jsonl", timestamp_agent())
    assert cli.main(["report", path, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    app = data["applications"][0]
    assert app["recommended"] == "off" and app["prompt_fix_option"] == "fix_5m"
    assert cli.main(["report", path, "--fail-on", "CP001"]) == 1


def test_report_no_content_hides_prompt_text(tmp_path, capsys):
    path = _write(tmp_path, "clock.jsonl", timestamp_agent())
    cli.main(["report", path, "--no-content"])
    assert "2026-11-23T10" not in capsys.readouterr().out


def test_check_exit_status(tmp_path):
    good = _write(tmp_path, "good.jsonl", appending_agent())
    bad = _write(tmp_path, "bad.jsonl", timestamp_agent())
    assert cli.main(["check", good, "--min-reusable", "0.5"]) == 0
    assert cli.main(["check", bad, "--min-reusable", "0.5"]) == 1


def test_other_commands_run(tmp_path, capsys, corpus_log):
    assert cli.main(["diagnose", corpus_log, "-v"]) == 0
    assert cli.main(["gaps", corpus_log, "--scales", "1,20"]) == 0
    assert cli.main(["rules"]) == 0
    assert cli.main(["verify", corpus_log, "--model", "m"]) == 2      # refuses without --yes
    assert "CP001" in capsys.readouterr().out


# ------------------------------------------------------------------ live plumbing (against a local stand-in)
class _Upstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.headers.get("x-api-key"), body))
        out = json.dumps({"usage": {"input_tokens": 7, "cache_creation_input_tokens": 3,
                                    "cache_read_input_tokens": 90}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def upstream():
    _Upstream.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_verify_sends_one_token_requests_and_reads_usage(upstream, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.build_opener(urllib.request.ProxyHandler({})).open)
    rows = appending_agent(3)
    res = live.run(rows, learn_from(rows), ["auto_5m"], "some-model", progress=lambda *_: None,
                   url=f"http://127.0.0.1:{upstream.server_port}/v1/messages")
    assert res["appender"]["auto_5m"].cache_read_input_tokens == 270
    key, body = _Upstream.seen[0]
    assert key == "sk-test" and body["max_tokens"] == 1 and body["model"] == "some-model"
    assert body["system"].startswith("[run ")


def test_capture_proxy_logs_body_but_not_the_key(upstream, tmp_path):
    from cacheplan.proxy import serve
    out = tmp_path / "cap.jsonl"
    px = serve(0, f"http://127.0.0.1:{upstream.server_port}", str(out), "demo")
    threading.Thread(target=px.serve_forever, daemon=True).start()
    body = json.dumps({"model": "m", "system": "hello", "messages": [{"role": "user", "content": "hi"}]}).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(3):                                             # several requests in a row
        rq = urllib.request.Request(f"http://127.0.0.1:{px.server_port}/v1/messages", data=body,
                                    headers={"x-api-key": "sk-secret", "content-type": "application/json"})
        assert json.loads(opener.open(rq, timeout=10).read())["usage"]["input_tokens"] == 7
    px.shutdown()
    logged = out.read_text()
    assert "sk-secret" not in logged and _Upstream.seen[0][0] == "sk-secret"
    rows = load(str(out)).rows
    assert len(rows) == 3 and rows[0]["app"] == "demo" and rows[0]["request"]["system"] == "hello"
