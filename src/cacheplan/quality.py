"""Does the prompt fix change the model's answers?

The prompt fix moves changing values out of the system prompt, so the model reads the same
content in a different place. This sends a sample of logged requests three times: as
written, as written again (a control, because answers vary from run to run anyway), and
with the fix applied. It then compares what the model did. It needs an API key and
spends output tokens.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .forecast import Knowledge, build, has_fix
from .live import API_URL, _post
from .logs import by_app


def action(resp: dict) -> tuple:
    """What the model decided to do: which tools it called with which arguments, or 'text'."""
    calls = [(b.get("name"), json.dumps(b.get("input"), sort_keys=True))
             for b in resp.get("content") or [] if b.get("type") == "tool_use"]
    return tuple(sorted(calls)) or ("text",)


def answer(resp: dict) -> str:
    return " ".join(b.get("text", "") for b in resp.get("content") or [] if b.get("type") == "text").strip()


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


@dataclass
class Result:
    app: str
    samples: int = 0
    control_same_action: int = 0
    fix_same_action: int = 0
    control_similarity: list[float] = field(default_factory=list)
    fix_similarity: list[float] = field(default_factory=list)
    different: list[dict] = field(default_factory=list)      # cases where only the fix changed the action

    @property
    def verdict(self) -> str:
        if not self.samples:
            return "no samples"
        mean = lambda xs: sum(xs) / len(xs) if xs else 1.0
        if (self.fix_same_action >= self.control_same_action - max(1, self.samples // 10)
                and mean(self.fix_similarity) >= mean(self.control_similarity) - 0.10):
            return "no difference beyond run-to-run variation"
        return "the fix changes answers more than run-to-run variation: review before adopting"


def run(rows: list[dict], k: Knowledge, model: str, sample: int = 12, max_tokens: int = 400,
        url: str = API_URL, progress=print) -> list[Result]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set in this shell.")
    out = []
    for app, rs in by_app(rows).items():
        if not has_fix(rs, k):
            continue
        step = max(1, len(rs) // sample)
        res = Result(app)
        for row in rs[::step][:sample]:
            def send(option):
                body = json.loads(json.dumps(build(option, row, k)))
                body.update(model=model, max_tokens=max_tokens)
                return _post(body, key, url)
            first, second, fixed = send("off"), send("off"), send("fix_5m")
            res.samples += 1
            same_control, same_fix = action(first) == action(second), action(first) == action(fixed)
            res.control_same_action += same_control
            res.fix_same_action += same_fix
            res.control_similarity.append(similarity(answer(first), answer(second)))
            res.fix_similarity.append(similarity(answer(first), answer(fixed)))
            if same_control and not same_fix:
                res.different.append({"as_written": list(action(first)), "with_fix": list(action(fixed))})
            progress(f"  {app}: {res.samples} compared")
        out.append(res)
    return out


def text(results: list[Result]) -> str:
    if not results:
        return "No application in this log has a prompt fix to check."
    mean = lambda xs: sum(xs) / len(xs) if xs else 1.0
    lines = []
    for r in results:
        lines += [f"{r.app}: {r.samples} requests compared",
                  f"  Same action as the original     control {r.control_same_action}/{r.samples}   "
                  f"with fix {r.fix_same_action}/{r.samples}",
                  f"  Wording similarity to original  control {mean(r.control_similarity):.0%}   "
                  f"with fix {mean(r.fix_similarity):.0%}",
                  f"  Verdict: {r.verdict}"]
        for d in r.different[:3]:
            lines.append(f"    changed: {d['as_written']} -> {d['with_fix']}")
    lines.append("\n'Action' is which tools were called with which arguments, or a plain text answer. "
                 "The control sends the same request twice; it shows how much answers vary without any change.")
    return "\n".join(lines)
