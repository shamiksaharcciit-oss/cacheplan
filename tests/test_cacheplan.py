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
def load_rows(rows):
    """Rows as the log reader returns them, without touching the disk."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/bot.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        return load(p).rows


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
    assert len(log.rows) == 4 and {r["provider"] for r in log.rows} == {"anthropic", "openai"}
    assert sum(log.skipped.values()) == 1


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
    assert v.plan.offsets == (3,)          # one breakpoint just before the insertion point is enough


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
            if isinstance(m["content"], str):               # untouched messages keep their form
                continue
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
    assert rows[0]["usage"]["cache_read_input_tokens"] == 90          # usage recorded, response text is not


# ------------------------------------------------------------------ model minimums
def test_minimum_cacheable_length_is_looked_up_by_model():
    from cacheplan.models import min_cacheable
    assert min_cacheable("claude-sonnet-5-5") == (512, True)
    assert min_cacheable("anthropic/claude-sonnet-5-5") == (512, True)
    assert min_cacheable("claude-sonnet-5") == (1024, True)
    assert min_cacheable("claude-haiku-4-5-20251001") == (4096, True)
    assert min_cacheable("claude-opus-4-7") == (2048, True)
    assert min_cacheable("some-other-model") == (1024, False)
    assert min_cacheable(None) == (1024, False)


def _short_rows(model=None, n=6):
    """A plain conversation whose prompts are about 700 to 900 estimated tokens."""
    rows, msgs = [], []
    for i in range(n):
        msgs = msgs + [{"role": "user", "content": f"question {i} " * 4}]
        req = {"system": "policy " * 300, "messages": list(msgs)}
        if model:
            req["model"] = model
        rows.append({"ts": i * 5.0, "app": "short", "request": req})
        msgs = msgs + [{"role": "assistant", "content": f"answer {i} " * 4}]
    return rows


def test_short_prompts_are_cached_only_when_the_model_allows_it():
    old = examine(_short_rows("claude-sonnet-5"))[0]            # 1,024-token minimum
    new = examine(_short_rows("claude-sonnet-5-5"))[0]          # 512-token minimum
    assert old.minimum == 1024 and old.cost("auto_5m") == pytest.approx(1.0)
    assert new.minimum == 512 and new.cost("auto_5m") < 0.6
    assert examine(_short_rows(), model="claude-sonnet-5-5")[0].minimum == 512      # --model
    assert examine(_short_rows("claude-sonnet-5-5"), min_tokens=4096)[0].minimum == 4096   # --min-tokens wins


def test_report_warns_when_prompts_sit_near_the_minimum(tmp_path, capsys):
    path = _write(tmp_path, "short.jsonl", _short_rows("claude-sonnet-5-5"))
    cli.main(["report", path])
    out = capsys.readouterr().out
    assert "close to the minimum cacheable length" in out and "512 tokens to be cacheable" in out
    path = _write(tmp_path, "unknown.jsonl", _short_rows())
    cli.main(["report", path])
    assert "model not recognised" in capsys.readouterr().out


def test_hidden_tool_instructions_count_towards_the_minimum():
    """A 300-token prompt with tools clears a 512-token minimum; without tools it does not."""
    tool = {"name": "search", "description": "Search.", "input_schema": {"type": "object", "properties": {}}}
    base = {"system": "policy " * 120, "messages": [{"role": "user", "content": "hi"}]}
    for tools, cached in (([tool], True), (None, False)):
        req = dict(base, **({"tools": tools} if tools else {}))
        sim = CacheSim(min_tokens=512)
        sim.request(place_breakpoints(req), 0)
        assert (sim.request(place_breakpoints(req), 1).cache_read_input_tokens > 0) is cached


# ------------------------------------------------------------------ measured tokens, prices, plans
def test_usage_is_read_from_streamed_responses():
    from cacheplan.proxy import usage_from_response
    sse = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12,'
           b'"cache_read_input_tokens":300,"cache_creation_input_tokens":40}}}\n\n'
           b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":55}}\n\n')
    u = usage_from_response(sse, "text/event-stream")
    assert u == {"input_tokens": 12, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 40, "output_tokens": 55}
    assert usage_from_response(b'{"usage":{"input_tokens":7}}', "application/json") == {"input_tokens": 7}
    assert usage_from_response(b"not json", "application/json") is None


def _with_usage(rows, scale, overhead):
    """Attach the usage a provider would report if real tokens were scale * estimate + overhead."""
    from cacheplan.forecast import _est
    for r in rows:
        real = round(_est(r["request"]) * scale) + (overhead if r["request"].get("tools") else 0)
        r["usage"] = {"input_tokens": real, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    return rows


def test_token_estimate_is_calibrated_from_recorded_responses():
    from cacheplan.forecast import calibrate
    tool = {"name": "search", "description": "Search.", "input_schema": {"type": "object", "properties": {}}}
    rows = appending_agent(n=10)
    for r in rows:
        r["request"]["system"] = "short system prompt"            # so prompt sizes vary a lot
        r["request"]["tools"] = [tool]
    cal = calibrate(_with_usage(rows, 1.3, 420), 300)
    assert cal.responses == 10
    assert cal.scale == pytest.approx(1.3, abs=0.02) and cal.overhead == pytest.approx(420, abs=15)
    assert calibrate(rows[:3], 300).responses == 0                # too few responses: defaults kept


def test_report_checks_its_forecast_against_recorded_usage(tmp_path, capsys):
    rows = _with_usage(appending_agent(), 1.2, 0)
    path = _write(tmp_path, "measured.jsonl", rows)
    cli.main(["report", path])
    out = capsys.readouterr().out
    assert "calibrated on 8 recorded responses" in out and "Forecast check" in out
    v = examine(load(path).rows)[0]
    assert v.measured_cost == pytest.approx(1.0)                   # the log itself ran without caching
    assert v.tokens == pytest.approx(sum(r["usage"]["input_tokens"] for r in rows), rel=0.01)


def test_prices_and_cache_read_price_come_from_the_model(tmp_path, capsys):
    from cacheplan.models import spec
    assert spec("claude-sonnet-5-5").read_x == 0.05 and spec("claude-sonnet-5-5").input_price == 2.0
    assert spec("claude-sonnet-4-5").read_x == 0.10 and spec("claude-opus-4-7").tool_overhead == 675
    rows = appending_agent()
    cheap = examine([dict(r, request=dict(r["request"], model="claude-sonnet-5-5")) for r in rows])[0]
    dear = examine([dict(r, request=dict(r["request"], model="claude-sonnet-4-5")) for r in rows])[0]
    assert cheap.cost("auto_5m") < dear.cost("auto_5m")           # cheaper cache reads
    path = _write(tmp_path, "priced.jsonl", [dict(r, request=dict(r["request"], model="claude-sonnet-5-5")) for r in rows])
    cli.main(["report", path])
    assert "$2 per million" in capsys.readouterr().out
    cli.main(["report", path, "--price", "9"])
    assert "$9 per million" in capsys.readouterr().out


def test_plan_depends_on_the_pace_of_the_log():
    """With seconds between turns breakpoints pay off; with ten-minute pauses the 5-minute
    plan places none, and the 1-hour plan still does."""
    from cacheplan.messages import FIVE_MIN, ONE_HOUR
    fast = learn_from(inserting_agent())
    assert fast.plans["inserter"][FIVE_MIN].offsets == (3,)
    slow_rows = inserting_agent()
    for i, r in enumerate(slow_rows):
        r["ts"] = i * 600.0
    slow = learn_from(slow_rows)
    p5, p60 = slow.plans["inserter"][FIVE_MIN], slow.plans["inserter"][ONE_HOUR]
    assert not (p5.head or p5.const_prefix or p5.offsets)
    assert p60.head or p60.const_prefix or p60.offsets
    assert examine(slow_rows)[0].recommended in ("plan_1h", "auto_1h")


def test_html_report_is_self_contained(tmp_path):
    path = _write(tmp_path, "clock.jsonl", timestamp_agent())
    out = tmp_path / "report.html"
    assert cli.main(["report", path, "--html", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "CP001" in page and "<script" not in page and "http" not in page


# ------------------------------------------------------------------ quality check, baseline, recorder
def test_quality_check_compares_actions_against_a_control(upstream, monkeypatch):
    from cacheplan import quality
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.build_opener(urllib.request.ProxyHandler({})).open)
    rows = timestamp_agent()
    res = quality.run(rows, learn_from(rows), "m", sample=3, progress=lambda *_: None,
                      url=f"http://127.0.0.1:{upstream.server_port}/v1/messages")
    assert len(res) == 1 and res[0].samples == 3
    assert res[0].fix_same_action == 3 and "inconclusive" in res[0].verdict      # three samples prove nothing
    assert len(_Upstream.seen) == 9 and _Upstream.seen[0][1]["max_tokens"] == 400
    assert quality.action({"content": [{"type": "tool_use", "name": "search", "input": {"q": 1}}]}) != \
        quality.action({"content": [{"type": "text", "text": "hi"}]})


def test_baseline_save_and_compare(tmp_path, capsys):
    good = _write(tmp_path, "good.jsonl", appending_agent())
    base = str(tmp_path / "base.json")
    assert cli.main(["baseline", good, "--save", base]) == 0
    assert "question" not in (tmp_path / "base.json").read_text()          # content-free
    assert cli.main(["baseline", good, "--compare", base]) == 0
    rows = timestamp_agent()
    for r in rows:
        r["app"] = "appender"                                             # the same app, now with a clock
    assert cli.main(["baseline", _write(tmp_path, "bad.jsonl", rows), "--compare", base]) == 1


def test_in_process_recorder_logs_request_and_usage(tmp_path):
    httpx = pytest.importorskip("httpx")
    from cacheplan.record import RecordingTransport

    def upstream_reply(request):
        if request.url.path.endswith("/v1/messages"):
            assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "secret answer"}],
                                         "usage": {"input_tokens": 11, "cache_read_input_tokens": 5}})
    out = tmp_path / "rec.jsonl"
    client = httpx.Client(transport=RecordingTransport(str(out), "inproc", inner=httpx.MockTransport(upstream_reply)))
    r = client.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": "sk-secret"},
                    json={"model": "m", "system": "s", "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["usage"]["input_tokens"] == 11
    client.get("https://api.anthropic.com/v1/models")                      # not recorded
    client.close()
    logged = out.read_text()
    assert "sk-secret" not in logged and "secret answer" not in logged
    rows = load(str(out)).rows
    assert len(rows) == 1 and rows[0]["app"] == "inproc" and rows[0]["usage"]["cache_read_input_tokens"] == 5


def test_quality_verdict_needs_the_gap_to_exceed_its_uncertainty():
    from cacheplan.quality import Result
    small = Result("a", samples=10, control_same_action=8, fix_same_action=7,
                   control_similarity=[0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.9, 0.1, 0.8, 0.6],
                   fix_similarity=[0.5, 0.4, 0.6, 0.2, 0.3, 0.5, 0.7, 0.2, 0.4, 0.7],
                   different=[{}], only_control_changed=0)
    assert "inconclusive" in small.verdict
    same = Result("b", samples=30, control_same_action=27, fix_same_action=27,
                  control_similarity=[0.6] * 30, fix_similarity=[0.59] * 30)
    assert same.verdict == "no difference beyond run-to-run variation"
    worse = Result("c", samples=30, control_same_action=28, fix_same_action=20,
                   control_similarity=[0.7] * 15 + [0.6] * 15, fix_similarity=[0.4] * 15 + [0.35] * 15,
                   different=[{}] * 8, only_control_changed=0)
    assert "review before adopting" in worse.verdict


# ------------------------------------------------------------------ other providers
from cacheplan import providers
from cacheplan.proxy import classify, usage_from_response
from cacheplan.sim import ImplicitSim

RULEBOOK = "You are a support agent.\n" + "\n".join(f"Rule {i}: answer politely and cite policy section {i * 7}." for i in range(400))


def _turns(conversations=5, steps=5):
    """(ts, turns so far) for append-only conversations."""
    t = 0.0
    for c in range(conversations):
        turns = [("user", f"Question {c}: where is order {c * 17}? " + "detail " * 60)]
        for s in range(steps):
            yield t, c, s, list(turns)
            turns += [("assistant", f"Answer {c}.{s} " + "words " * 150), ("user", f"Follow-up {c}.{s} " + "more " * 60)]
            t += 20
        t += 60


def openai_chat(model="gpt-5.4", stamp=False):
    return [{"ts": t, "app": "bot", "request": {"model": model, "messages": [
        {"role": "system", "content": RULEBOOK + (f"\nCurrent time: 2026-11-23T10:{c:02d}:{s:02d}" if stamp else "")},
        *({"role": r, "content": x} for r, x in turns)]}} for t, c, s, turns in _turns()]


def converse(model="eu.anthropic.claude-sonnet-5-5-v1:0", cache_point=False):
    system = [{"text": RULEBOOK}] + ([{"cachePoint": {"type": "default"}}] if cache_point else [])
    return [{"ts": t, "app": "bot", "request": {"modelId": model, "system": system,
             "messages": [{"role": r, "content": [{"text": x}]} for r, x in turns]}} for t, c, s, turns in _turns()]


def test_request_formats_are_detected_and_converted():
    chat = {"model": "gpt-5.4", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}
    provider, req = providers.normalise(chat)
    assert provider == "openai" and system_text(req) == "S" and req["tools"][0]["name"] == "f"
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]
    assert req["messages"][1]["content"][0]["type"] == "tool_use" and req["messages"][2]["content"][0]["type"] == "tool_result"

    resp = {"model": "gpt-6.1-sol", "instructions": "S", "prompt_cache_options": {"mode": "explicit"},
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi", "prompt_cache_breakpoint": {}}]}]}
    provider, req = providers.normalise(resp)
    assert provider == "openai" and "cache_control" not in req
    assert [bool(ttl) for _, ttl in prefix_blocks(req)] == [False, True]
    # without explicit mode, GPT-5.6 and later cache automatically
    assert providers.normalise(dict(resp, prompt_cache_options={}))[1]["cache_control"] == {"type": "ephemeral"}

    conv = {"modelId": "anthropic.claude-sonnet-5-5", "system": [{"text": "S"}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "toolConfig": {"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}}}]}}
    provider, req = providers.normalise(conv)
    assert provider == "bedrock" and req["system"][0]["cache_control"]["ttl"] == "1h" and req["tools"][0]["name"] == "f"

    gem = {"systemInstruction": {"parts": [{"text": "S"}]}, "contents": [{"role": "user", "parts": [{"text": "hi"}]},
                                                                           {"role": "model", "parts": [{"text": "yo"}]}]}
    provider, req = providers.normalise(gem)
    assert provider == "gemini" and [m["role"] for m in req["messages"]] == ["user", "assistant"]
    assert providers.detect({"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "hi"}]}) == "anthropic"


def test_caching_rules_per_provider_and_model():
    old, new = providers.rules("openai", "gpt-5.4"), providers.rules("openai", "gpt-6.1-sol")
    assert old.mode == "implicit" and old.write_short == 1.0 and old.read_x == pytest.approx(0.1)
    assert new.mode == "breakpoints" and (new.write_short, new.write_long) == (1.25, 2.5)
    assert (new.ttl_short, new.ttl_long) == (1800, 43200) and new.read_x == pytest.approx(0.05)
    claude = providers.rules("bedrock", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert claude.minimum == 4096 and claude.forecast
    nova = providers.rules("bedrock", "amazon.nova-pro-v1:0")
    assert nova.write_long is None and not nova.automatic
    assert not providers.rules("gemini", "gemini-3-flash").forecast
    assert not providers.rules("bedrock", "meta.llama4").forecast


def test_usage_is_read_in_every_providers_format():
    assert providers.usage_parts({"inputTokens": 10, "cacheReadInputTokens": 90, "cacheWriteInputTokens": 5}) == (10, 5, 90, 0)
    assert providers.usage_parts({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 64}}) == (36, 0, 64, 0)
    assert providers.usage_parts({"input_tokens": 100, "input_tokens_details": {
        "cached_tokens": 50, "cache_write_tokens": 30, "cache_write_tokens_details": {"ttl_12h_tokens": 30}}}) == (20, 30, 50, 30)
    assert providers.usage_parts({"promptTokenCount": 100, "cachedContentTokenCount": 40}) == (60, 0, 40, 0)
    sse = b'data: {"type":"response.completed","response":{"usage":{"input_tokens":9,"input_tokens_details":{"cached_tokens":4}}}}\n\n'
    assert usage_from_response(sse, "text/event-stream")["input_tokens"] == 9
    chunks = b'data: {"choices":[{}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7}}\n\ndata: [DONE]\n\n'
    assert usage_from_response(chunks, "text/event-stream") == {"prompt_tokens": 7}
    assert usage_from_response(b'{"usageMetadata":{"promptTokenCount":3}}', "application/json") == {"promptTokenCount": 3}


def test_model_calls_are_recognised_by_path():
    assert classify("POST", "/v1/messages?beta=true") == ("anthropic", None)
    assert classify("POST", "/v1/chat/completions") == ("openai", None)
    assert classify("POST", "/v1/responses") == ("openai", None)
    assert classify("POST", "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse") == ("gemini", "gemini-3-flash")
    assert classify("POST", "/v1/embeddings") is None and classify("GET", "/v1/messages") is None


def test_implicit_cache_reads_the_shared_prefix_in_steps():
    sim = ImplicitSim(ttl=300, interval=128)
    a = {"system": [{"type": "text", "text": "x" * 35000}], "messages": [{"role": "user", "content": "first question"}]}
    b = {"system": a["system"], "messages": [{"role": "user", "content": "another one"}]}
    first, second = sim.request(a, 0), sim.request(b, 10)
    assert first.cache_read_input_tokens == 0 and first.cache_creation_input_tokens == 0      # no write fee
    assert second.cache_read_input_tokens >= 9000 and second.cache_read_input_tokens % 128 == 0
    assert sim.request(b, 1000).cache_read_input_tokens == 0                                  # expired
    short = {"messages": [{"role": "user", "content": "x" * 1000}]}
    sim.request(short, 2000)
    assert sim.request(short, 2001).cache_read_input_tokens == 0                              # under the minimum


def test_openai_automatic_models_get_a_forecast_without_breakpoint_options():
    v = examine(load_rows(openai_chat()))[0]
    assert v.rules.provider == "openai" and v.rules.mode == "implicit" and v.app.endswith("[openai]")
    assert "plan_5m" not in v.usage and v.cost("auto_5m") < 0.3 and v.recommended.startswith("auto")
    assert not any(f.rule == "CP003" for f in findings(v, load_rows(openai_chat()), learn_from(load_rows(openai_chat()))))
    stamped = load_rows(openai_chat(stamp=True))
    k = learn_from(stamped)
    w = examine(stamped, k)[0]
    assert any(f.rule == "CP001" for f in findings(w, stamped, k)) and w.fix_option and w.cost(w.fix_option) < w.cost("auto_5m")


def test_openai_breakpoint_models_use_their_own_lifetimes_and_write_prices():
    v = examine(load_rows(openai_chat("gpt-6.1-sol")))[0]
    assert v.rules.mode == "breakpoints" and "logged" in v.usage          # automatic unless the request says otherwise
    assert v.label("plan_1h") == "Planned breakpoints, 12-hour"
    u = v.usage["auto_1h"]
    assert v.cost("auto_1h") == pytest.approx(
        (u.input_tokens + 2.5 * u.cache_creation_input_tokens + v.rules.read_x * u.cache_read_input_tokens) / v.usage["off"].total)


def test_bedrock_claude_is_forecast_like_anthropic_and_cache_points_are_read():
    v = examine(load_rows(converse()))[0]
    assert v.rules.provider == "bedrock" and v.minimum == 512 and v.cost(v.recommended) < 0.3
    assert "logged" not in v.usage
    marked = examine(load_rows(converse(cache_point=True)))[0]
    assert "logged" in marked.usage and marked.usage["logged"].cache_read_input_tokens > 0


def test_nova_has_no_long_lifetime_and_gemini_has_no_forecast(tmp_path, capsys):
    nova = examine(load_rows(converse("amazon.nova-pro-v1:0")))[0]
    assert set(nova.usage) == {"off", "plan_5m"}
    gem = [{"ts": r["ts"], "app": "bot", "model": "gemini-3-flash", "request": {
        "systemInstruction": {"parts": [{"text": r["request"]["messages"][0]["content"]}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                     for m in r["request"]["messages"][1:]]}} for r in openai_chat(stamp=True)]
    path = _write(tmp_path, "gem.jsonl", gem)
    assert cli.main(["report", path]) == 0
    out = capsys.readouterr().out
    assert "No cost forecast" in out and "CP001" in out and "Cost vs no caching" not in out
    assert cli.main(["report", path, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["applications"][0]["options"] == {}


def test_forecast_check_uses_openai_usage(tmp_path, capsys):
    rows = openai_chat()
    sim = ImplicitSim(300, 128)
    for r, loaded in zip(rows, load_rows(rows)):
        u = sim.request(loaded["request"], r["ts"], 1024)
        r["usage"] = {"prompt_tokens": u.total, "prompt_tokens_details": {"cached_tokens": u.cache_read_input_tokens}}
    assert cli.main(["report", _write(tmp_path, "oa.jsonl", rows)]) == 0
    assert "Forecast check" in capsys.readouterr().out


def test_live_commands_refuse_other_providers(tmp_path):
    path = _write(tmp_path, "oa.jsonl", openai_chat())
    with pytest.raises(SystemExit, match="Anthropic API logs only"):
        cli.main(["verify", path, "--model", "claude-sonnet-5-5"])
    with pytest.raises(SystemExit, match="record_bedrock"):
        cli.main(["capture", "--out", str(tmp_path / "x.jsonl"), "--provider", "bedrock"])


def test_bedrock_recorder_writes_request_and_usage(tmp_path):
    class Events:
        def __init__(self):
            self.handlers = {}

        def register(self, name, fn):
            self.handlers[name] = fn

    class Client:
        def __init__(self):
            self.meta = type("M", (), {"events": Events()})()

    from cacheplan.record import record_bedrock
    out = tmp_path / "b.jsonl"
    c = record_bedrock(Client(), str(out), app="bot")
    params = converse()[0]["request"]
    h = c.meta.events.handlers
    h["provide-client-params.bedrock-runtime.Converse"](params=params)
    h["after-call.bedrock-runtime.Converse"](parsed={"ResponseMetadata": {"HTTPStatusCode": 200},
                                                    "usage": {"inputTokens": 12, "cacheReadInputTokens": 0}})
    h["provide-client-params.bedrock-runtime.ConverseStream"](params=params)
    parsed = {"ResponseMetadata": {"HTTPStatusCode": 200},
              "stream": iter([{"contentBlockDelta": {}}, {"metadata": {"usage": {"inputTokens": 5}}}])}
    h["after-call.bedrock-runtime.ConverseStream"](parsed=parsed)
    assert len(list(parsed["stream"])) == 2
    log = load(str(out))
    assert [r["provider"] for r in log.rows] == ["bedrock", "bedrock"]
    assert [r["usage"]["inputTokens"] for r in log.rows] == [12, 5]


def test_forecast_check_says_when_the_cache_was_warm_before_the_log(tmp_path, capsys):
    rows = openai_chat()
    for r in rows:
        r["usage"] = {"prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 3200}}
    assert cli.main(["report", _write(tmp_path, "warm.jsonl", rows)]) == 0
    assert "warm from traffic before this log" in capsys.readouterr().out


def test_newer_openai_models_reuse_earlier_message_endings_across_conversations():
    """Measured on gpt-5.6-luna: a new conversation read the shared system message from the
    cache although the only write so far ended at another conversation's first turn."""
    def req(question):
        return {"cache_control": {"type": "ephemeral"}, "system": [{"type": "text", "text": RULEBOOK}],
                "messages": [{"role": "user", "content": question}]}
    plain, openai = CacheSim(), CacheSim(message_entries=True)
    for sim in (plain, openai):
        sim.request(req("first conversation"), 0)
    assert plain.request(req("second conversation"), 5).cache_read_input_tokens == 0
    second = openai.request(req("second conversation"), 5)
    assert second.cache_read_input_tokens > 0.9 * second.total
    assert providers.rules("openai", "gpt-5.6-luna").message_entries
    assert not providers.rules("bedrock", "anthropic.claude-sonnet-5-5").message_entries


def test_capture_without_content_keeps_the_report_and_drops_the_text(tmp_path, capsys):
    from cacheplan.proxy import new_salt, scramble
    salt = new_salt()
    rows = openai_chat("gpt-6.1-sol", stamp=True)
    hidden = [dict(r, request=scramble(r["request"], salt)) for r in rows]
    raw = json.dumps(hidden)
    assert "support agent" not in raw and "politely" not in raw and "gpt-6.1-sol" in raw and '"role": "system"' in raw
    assert len(json.dumps(hidden[3]["request"])) == len(json.dumps(rows[3]["request"]))
    a, b = examine(load_rows(rows))[0], examine(load_rows(hidden))[0]
    assert b.layout == a.layout and b.cost("auto_5m") == pytest.approx(a.cost("auto_5m"), abs=0.02)
    assert b.fix_option and b.cost(b.fix_option) == pytest.approx(a.cost(a.fix_option), abs=0.03)
    assert cli.main(["report", _write(tmp_path, "hidden.jsonl", hidden)]) == 0
    assert "CP001" in capsys.readouterr().out


def test_instructions_on_gpt_6_2_sit_after_the_input_and_are_never_cached(tmp_path, capsys):
    """OpenAI's guide: models starting with GPT-6.2 append top-level instructions after
    input and bill them at the uncached rate even on a cache hit."""
    def responses(model):
        return [{"ts": t, "app": "bot", "request": {"model": model, "instructions": RULEBOOK,
                 "input": [{"role": r, "content": x} for r, x in turns]}} for t, c, s, turns in _turns()]
    assert providers.instructions_after_input("gpt-6.2-sol") and providers.instructions_after_input("gpt-7-sol")
    assert not providers.instructions_after_input("gpt-6.1-sol") and not providers.instructions_after_input("gpt-5.6-luna")
    old = providers.normalise(responses("gpt-6.1-sol")[0]["request"])[1]
    new = providers.normalise(responses("gpt-6.2-sol")[0]["request"])[1]
    assert system_text(old) and not system_text(new) and new["uncached_tail"]
    before, after = examine(load_rows(responses("gpt-6.1-sol")))[0], examine(load_rows(responses("gpt-6.2-sol")))[0]
    assert before.cost("auto_5m") < 0.4                      # instructions lead the prompt and are cached
    assert after.cost("auto_5m") > 0.7                       # the same text is paid in full every time
    assert after.tokens == pytest.approx(before.tokens, rel=0.02)
    rows = load_rows(responses("gpt-6.2-sol"))
    assert any(f.rule == "CP010" for f in findings(after, rows, learn_from(rows)))


def test_conformance_openai_gpt_5_6_luna_append_only():
    """Replay a recorded live run and require the forecast to stay close to what was billed.
    This run is the one the message-ending rule was corrected from, so it guards against
    regressions; it is not independent evidence."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent
    fixture = json.loads((root / "conformance" / "openai_gpt-5.6-luna_append.json").read_text())
    spec = importlib.util.spec_from_file_location("driver", root.parent / "benchmarks" / "providers" / "openai_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    rows, usage, t = [], iter(fixture["usage"]), 0.0
    for conversation in driver.QUESTIONS:
        history = []
        for question in conversation:
            history.append({"role": "user", "content": question})
            rows.append({"ts": t, "app": "driver", "usage": next(usage), "request": {
                "model": fixture["model"], "messages": [{"role": "system", "content": driver.RULES}, *history]}})
            history.append({"role": "assistant", "content": "Your order ships within the window for its category. " * 3})
            t += 3
    v = examine(load_rows(rows))[0]
    assert v.measured_cost == pytest.approx(fixture["billed_cost_vs_uncached"], abs=0.005)
    assert v.cost(v.as_sent) == pytest.approx(v.measured_cost, abs=0.02)
    assert v.recommended == "logged"


def test_capture_files_are_private_and_the_ledger_lists_evidence(tmp_path, capsys):
    import os
    from cacheplan.proxy import open_log
    path = tmp_path / "new.jsonl"
    with open_log(str(path)) as f:
        f.write("{}\n")
    if os.name == "posix":
        assert (path.stat().st_mode & 0o077) == 0
    assert cli.main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "documentation only" in out and "gpt-5.6-luna" in out and "developers.openai.com" in out


def test_scrambling_never_makes_two_short_values_look_the_same():
    from cacheplan.proxy import new_salt, scramble
    for _ in range(20):
        salt = new_salt()
        seen = {scramble(f"{n:02d}", salt) for n in range(100)} | {scramble(f"{n:03d}", salt) for n in range(300)}
        assert len(seen) == 400
        assert all(len(scramble(w, salt)) == len(w) for w in ("a", "Zz", "2026", "abcdef", "abcdefg", "naïve"))
        assert scramble("id 7 id 7", salt).split()[1] == scramble("7", salt)


def test_grouping_controls(tmp_path):
    """Labels and conversation ids can come from any field; prompt splitting can be turned
    off; one-shot workloads report a change in the first message."""
    doc = "Reference manual. " + "section text " * 600
    rows = [{"ts": float(i), "metadata": {"route": "search" if i % 2 else "chat", "thread": f"t{i // 4}"},
             "request": {"model": "claude-sonnet-5-5", "system": "Answer from the manual." if i % 2 else "Chat politely with the user please.",
                         "messages": [{"role": "user", "content": (doc if i < 6 else doc.replace("Reference", "Updated")) + f" Question {i}?"}]}}
            for i in range(12)]
    path = _write(tmp_path, "mixed.jsonl", rows)
    assert {r["app"] for r in load(path).rows} == {'mixed#1 "Chat politely with the user please."', 'mixed#2 "Answer from the manual."'}
    assert {r["app"] for r in load(path, split_prompts=False).rows} == {"mixed"}
    by_field = load(path, app_field="metadata.route", session_field="metadata.thread").rows
    assert {r["app"] for r in by_field} == {"search", "chat"} and by_field[5]["session"] == "t1"
    raw = _write(tmp_path, "raw.jsonl", [dict(r["request"], metadata={"user_id": "u7"}) for r in rows])
    assert {r["session"] for r in load(raw, session_field="metadata.user_id").rows} == {"u7"}
    chat = [r for r in load(path, app_field="metadata.route").rows if r["app"] == "chat"]
    assert not find_breaks("chat", chat)                                   # read as separate conversations
    shots = [r for r in load(path, app_field="metadata.route", one_shot=True).rows if r["app"] == "chat"]
    assert [b.path for b in find_breaks("chat", shots)].count("messages[0].content[0]") >= 1
    assert cli.main(["report", path, "--app-field", "metadata.route", "--one-shot", "--no-split"]) == 0
