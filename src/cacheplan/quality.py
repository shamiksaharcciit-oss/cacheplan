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
    only_control_changed: int = 0                            # cases where only the repeat changed it

    @property
    def wording_gap(self) -> tuple[float, float]:
        """Mean of (control similarity - fix similarity) per request, and its standard error."""
        d = [c - f for c, f in zip(self.control_similarity, self.fix_similarity)]
        if len(d) < 2:
            return (d[0] if d else 0.0), float("inf")
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
        return mean, (var / len(d)) ** 0.5

    @property
    def action_gap(self) -> tuple[int, int]:
        """(requests where only the fix changed the action, requests where only the control did)."""
        return len(self.different), self.only_control_changed

    @property
    def verdict(self) -> str:
        if not self.samples:
            return "no samples"
        gap, se = self.wording_gap
        only_fix, only_control = self.action_gap
        clear_wording = gap > 0.10 and gap > 2 * se
        clear_action = only_fix - only_control >= 3 and only_fix >= 2 * max(1, only_control)
        if clear_wording or clear_action:
            return "the fix changes answers more than run-to-run variation: review before adopting"
        if self.samples < 20 or (gap > 0.05 and gap <= 2 * se):
            return ("inconclusive: the difference is within what " + str(self.samples)
                    + " samples can distinguish from run-to-run variation; compare more requests")
        return "no difference beyond run-to-run variation"


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
            if same_fix and not same_control:
                res.only_control_changed += 1
            progress(f"  {app}: {res.samples} compared")
        out.append(res)
    return out


def text(results: list[Result]) -> str:
    if not results:
        return "No application in this log has a prompt fix to check."
    mean = lambda xs: sum(xs) / len(xs) if xs else 1.0
    lines = []
    for r in results:
        gap, se = r.wording_gap
        only_fix, only_control = r.action_gap
        lines += [f"{r.app}: {r.samples} requests compared",
                  f"  Same action as the original     control {r.control_same_action}/{r.samples}   "
                  f"with fix {r.fix_same_action}/{r.samples}",
                  f"    action changed only with the fix: {only_fix}; only in the control: {only_control}",
                  f"  Wording similarity to original  control {mean(r.control_similarity):.0%}   "
                  f"with fix {mean(r.fix_similarity):.0%}",
                  f"    per-request gap {gap:+.0%}, give or take {2 * se:.0%}" if se != float("inf") else "",
                  f"  Verdict: {r.verdict}"]
        for d in r.different[:3]:
            lines.append(f"    changed: {d['as_written']} -> {d['with_fix']}")
    lines.append("\n'Action' is which tools were called with exactly which arguments, or a plain text answer. "
                 "The control sends the same request twice; it shows how much answers vary without any change. "
                 "A verdict needs the gap to exceed its own uncertainty, so small samples often end inconclusive.")
    return "\n".join(l for l in lines if l)
