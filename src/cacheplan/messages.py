"""Core operations on Anthropic Messages API requests ({tools, system, messages}).

  learn()                 infer, from a sample of requests produced by one app, which
                          parts of the system prompt are static, shared by a few
                          callers (tenant-like) or volatile. No annotations needed.
  canonicalize_request()  rewrite a request so stable content forms the prefix,
                          inside the order the API fixes (tools -> system -> messages),
                          and place up to four cache breakpoints.
  prefix_blocks()         the request as the ordered list of cacheable blocks.

Nothing is dropped: every value removed from the system prompt reappears either in
a trailing system block (shared values) or in a <request_context> block at the end
of the latest user message (volatile values).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

EPHEMERAL = {"type": "ephemeral"}
FIVE_MIN, ONE_HOUR = 300, 3600
CHARS_PER_TOKEN = 3.0        # rough; three live runs measured 2.6 to 3.2 characters per token


def cc(ttl: int = FIVE_MIN) -> dict:
    """A cache_control marker for the 5-minute or the 1-hour lifetime."""
    return {"type": "ephemeral", "ttl": "1h"} if ttl >= ONE_HOUR else {"type": "ephemeral"}


def _ttl(holder: dict) -> int:
    c = holder.get("cache_control")
    if not c:
        return 0
    return ONE_HOUR if isinstance(c, dict) and c.get("ttl") == "1h" else FIVE_MIN
SLOT_NOTE = ("Values written as <<name>> are given under [name], either in the next "
             "system block or in <request_context> in the latest user message.")


def est_tokens(s: str) -> int:
    """Rough token estimate. Live verification uses the provider's real counts."""
    return max(1, int(len(s) / CHARS_PER_TOKEN))


def system_text(req: dict) -> str:
    s = req.get("system") or ""
    if isinstance(s, str):
        return s
    return "\n".join(b.get("text", "") for b in s if b.get("type") == "text")


def _blocks(content) -> list[dict]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "cache_control"}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def strip_cache_control(req: dict) -> dict:
    return _strip({k: req[k] for k in ("tools", "system", "messages") if req.get(k)})


def prefix_blocks(req: dict) -> list[tuple[str, int]]:
    """(serialised block, lifetime of its breakpoint in seconds or 0) in hashing order."""
    return [(text, ttl) for _, text, ttl in located_blocks(req)]


def located_blocks(req: dict) -> list[tuple[str, str, int]]:
    """(path, serialised block, breakpoint lifetime) in the order the API hashes them:
    tools, then system, then messages."""
    out: list[tuple[str, str, int]] = []
    for i, t in enumerate(req.get("tools") or []):
        out.append((f"tools[{i}]", "tool:" + json.dumps(_strip(t), sort_keys=True), _ttl(t)))
    s = req.get("system")
    for i, b in enumerate(_blocks(s) if s else []):
        out.append((f"system[{i}]", "system:" + json.dumps(_strip(b), sort_keys=True), _ttl(b)))
    for i, m in enumerate(req.get("messages") or []):
        for j, b in enumerate(_blocks(m["content"])):
            out.append((f"messages[{i}].content[{j}]",
                        m["role"] + ":" + json.dumps(_strip(b), sort_keys=True), _ttl(b)))
    if req.get("cache_control") and out:            # top-level field = automatic caching
        p, t, _ = out[-1]
        out[-1] = (p, t, _ttl(req))
    return out


# --------------------------------------------------------------------------- learning
@dataclass
class Slot:
    name: str
    kind: str                 # "shared" (few distinct values) or "volatile"
    distinct: int = 0
    example: str = ""


@dataclass
class Line:
    const: str | None = None                # identical in every sample
    pattern: re.Pattern | None = None       # a line with varying words
    fmt: str = ""                           # that line with <<name>> placeholders
    subslots: list[Slot] = field(default_factory=list)


@dataclass
class Gap:
    const: list[str] | None = None          # identical in every sample
    slot: Slot | None = None                # the whole gap varies (different line counts)
    lines: list[Line] | None = None         # same number of lines everywhere: line by line

    def slots(self) -> list[Slot]:
        if self.slot:
            return [self.slot]
        return [s for l in (self.lines or []) for s in l.subslots]


@dataclass
class Template:
    anchors: list[str]
    gaps: list[Gap]
    samples: int

    def slots(self) -> list[Slot]:
        return [s for g in self.gaps for s in g.slots()]


def _split(text: str, anchors: list[str]) -> list[list[str]] | None:
    lines, gaps, pos = text.split("\n"), [], 0
    for a in anchors:
        try:
            i = lines.index(a, pos)
        except ValueError:
            return None
        gaps.append(lines[pos:i])
        pos = i + 1
    gaps.append(lines[pos:])
    return gaps


def _slug(words: str, used: set[str]) -> str:
    base = "_".join(re.findall(r"[a-z0-9]+", words.lower())[-3:]) or "value"
    name, n = base, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name


def _kind(values: list[str]) -> tuple[str, int]:
    d = len(set(values))
    return ("shared" if d <= max(1, len(values) // 4) else "volatile"), d


def _line(lines: list[str], used: set[str]) -> Line | None:
    """All samples have exactly one line here: find the words that vary."""
    ref = lines[0].split(" ")
    stable = set(range(len(ref)))
    for other in lines[1:]:
        sm = SequenceMatcher(None, ref, other.split(" "), autojunk=False)
        stable &= {i for b in sm.get_matching_blocks() for i in range(b.a, b.a + b.size)}
    parts: list[tuple[str, str]] = []           # ("lit", text) | ("slot", "")
    for i, w in enumerate(ref):
        k = "lit" if i in stable else "slot"
        if parts and parts[-1][0] == k:
            if k == "lit":
                parts[-1] = ("lit", parts[-1][1] + " " + w)
        else:
            parts.append((k, w if k == "lit" else ""))
    if not any(k == "slot" for k, _ in parts) or len(stable) * 2 < len(ref):
        return None                 # nothing varies, or too little is fixed to call it a form
    pat = re.compile(" ".join(re.escape(t) if k == "lit" else "(.+?)" for k, t in parts), re.S)
    found = [pat.fullmatch(l) for l in lines]
    if not all(found):
        return None
    subs, fmt, prev, gi = [], [], "", 0
    for k, t in parts:
        if k == "lit":
            fmt.append(t)
            prev = t
        else:
            vals = [m.group(gi + 1) for m in found]
            kind, d = _kind(vals)
            s = Slot(_slug(prev, used), kind, d, vals[0])
            subs.append(s)
            fmt.append(f"<<{s.name}>>")
            gi += 1
    return Line(pattern=pat, fmt=" ".join(fmt), subslots=subs)


def _block_name(first_lines: list[str], anchor: str) -> str:
    words = [l.split(" ") for l in first_lines]
    n = 0
    while all(len(w) > n and w[n] == words[0][n] for w in words):
        n += 1
    if n >= 2:
        return " ".join(words[0][:n])
    return anchor if anchor.rstrip().endswith(":") else "block"


def learn(texts: list[str], keys: list | None = None, max_samples: int = 40) -> Template | None:
    """Infer the template behind a set of system prompts from one application.

    keys, if given, identify the conversation each text belongs to. A value is
    treated as shared (cacheable per tenant or user) only if it has few distinct
    values and never changes inside one conversation; otherwise it is volatile.
    """
    all_texts = texts
    texts = [t for t in texts if t]
    if len(texts) < 4:
        return None
    step = max(1, len(texts) // max_samples)
    samples = texts[::step][:max_samples]
    ref = samples[0].split("\n")
    cand = {i for i, l in enumerate(ref) if len(l.strip()) >= 12}
    for s in samples[1:]:
        sm = SequenceMatcher(None, ref, s.split("\n"), autojunk=False)
        cand &= {i for b in sm.get_matching_blocks() for i in range(b.a, b.a + b.size)}
    anchors = [ref[i] for i in sorted(cand)]
    splits = [g for g in (_split(s, anchors) for s in samples) if g is not None]
    if len(splits) < 4:
        return None
    used: set[str] = set()
    gaps: list[Gap] = []
    for k in range(len(anchors) + 1):
        vals = [sp[k] for sp in splits]
        if all(v == vals[0] for v in vals):
            gaps.append(Gap(const=vals[0]))
            continue
        g = None
        varying = sum(1 for j in range(len(vals[0])) if any(len(v) > j and v[j] != vals[0][j] for v in vals))
        if all(len(v) == len(vals[0]) for v in vals) and varying <= 2:   # form-like lines
            ls = []
            for j in range(len(vals[0])):
                col = [v[j] for v in vals]
                ls.append(Line(const=col[0]) if all(x == col[0] for x in col) else _line(col, used))
            if all(ls):
                g = Gap(lines=ls)
        if g is None:
            joined = ["\n".join(v).strip("\n") for v in vals]
            kind, d = _kind(joined)
            name = _block_name([j.split("\n")[0] for j in joined], anchors[k - 1] if k else "block")
            g = Gap(slot=Slot(_slug(name, used), kind, d, joined[0]))
        gaps.append(g)
    tpl = Template(anchors, gaps, len(splits))
    if keys:                                    # demote values that change inside a conversation
        seen: dict = {}
        by_name = {sl.name: sl for sl in tpl.slots()}
        for text, key in zip(all_texts, keys):
            got = apply_template(tpl, text) if text else None
            for name, value in (got[1] if got else []):
                if seen.setdefault((key, name), value) != value:
                    by_name[name].kind = "volatile"
    return tpl


def apply_template(tpl: Template, text: str):
    """-> (static text with placeholders, shared [(name, value)], volatile [...]) or None."""
    split = _split(text, tpl.anchors)
    if split is None:
        return None
    lines: list[str] = []
    shared, volatile = [], []

    def put(slot: Slot, value: str):
        if value.strip():
            (shared if slot.kind == "shared" else volatile).append((slot.name, value))

    for k, g in enumerate(tpl.gaps):
        cur = split[k]
        if g.const is not None:
            if cur != g.const:
                return None
            lines += cur
        elif g.slot:
            put(g.slot, "\n".join(cur).strip("\n"))
            body = [i for i, l in enumerate(cur) if l.strip()]
            lead, trail = (body[0], len(cur) - 1 - body[-1]) if body else (0, 0)
            lines += [""] * lead + [f"<<{g.slot.name}>>"] + [""] * trail
        else:
            if len(cur) != len(g.lines):
                return None
            for spec, line in zip(g.lines, cur):
                if spec.const is not None:
                    if line != spec.const:
                        return None
                    lines.append(line)
                    continue
                m = spec.pattern.fullmatch(line)
                if not m:
                    return None
                for s, v in zip(spec.subslots, m.groups()):
                    put(s, v)
                lines.append(spec.fmt)
        if k < len(tpl.anchors):
            lines.append(tpl.anchors[k])
    return "\n".join(lines), shared, volatile


# --------------------------------------------------------------------------- rewriting
def _fmt(pairs) -> str:
    return "\n\n".join(f"[{n}]\n{v}" for n, v in pairs)


def _msg_blocks(req: dict) -> list[str]:
    return [m["role"] + ":" + json.dumps(_strip(b), sort_keys=True)
            for m in req.get("messages") or [] for b in _blocks(m["content"])]


@dataclass
class BreakPlan:
    """Where logged traffic showed a prefix being reused often enough to pay for the write.
    head          cache the tools + system blocks
    const_prefix  leading message blocks that later requests kept matching (examples etc.)
    offsets       distances from the last block where a later request still matched"""
    head: bool = True
    const_prefix: int = 0
    offsets: tuple = (0,)
    reuse: dict = field(default_factory=dict)       # position -> share of writes later read


def learn_breakpoints(reqs: list[dict], window: int = 20,
                      write_x: float = 1.25, read_x: float = 0.10) -> BreakPlan:
    """Cost-aware breakpoint plan for one application, learned from its log.

    For every request, find how far a later request (within the next `window`) repeated
    its prefix. A breakpoint is kept only where the share of writes that were later read
    exceeds the break-even point (write premium / read saving: 28% at the default prices),
    because everything up to the last breakpoint is paid for at the write price."""
    from collections import Counter
    need = (write_x - 1) / (1 - read_x)
    seqs, heads = [], []
    for r in reqs:
        bl = [t for t, _ in prefix_blocks(strip_cache_control(r))]
        seqs.append(bl)
        heads.append(sum(1 for t in bl if t.startswith(("tool:", "system:"))))
    future = []                                   # blocks of request i repeated by a later one
    for i, cur in enumerate(seqs[:-1]):
        best = 0
        for nxt in seqs[i + 1:i + 1 + window]:
            n = 0
            while n < min(len(cur), len(nxt)) and cur[n] == nxt[n]:
                n += 1
            best = max(best, n)
        future.append(best)
    if not future:
        return BreakPlan(head=False, offsets=())      # a single request: nothing to reuse
    k = len(future)
    share = lambda hits: sum(hits) / k
    p_head = share(f >= h > 0 for f, h in zip(future, heads))
    beyond = [f - h for f, h in zip(future, heads) if f > h]
    const = min(beyond) if len(beyond) >= 0.5 * k else 0
    cand = Counter(len(seqs[i]) - f for i, f in enumerate(future) if f > heads[i])
    reuse, keep = {"head": p_head}, []
    for off, _ in cand.most_common(5):
        p = share(f >= len(seqs[i]) - off > heads[i] for i, f in enumerate(future))
        reuse[f"end-{off}"] = p
        if p > need:
            keep.append(off)
    offsets = tuple(sorted({min(keep), keep[0]})) if keep else ()
    return BreakPlan(p_head > need, const, offsets, reuse)


def place_breakpoints(req: dict, sort_tools: bool = False, plan: BreakPlan | None = None,
                      ttl: int = FIVE_MIN) -> dict:
    """At most four breakpoints. Without a plan: end of tools, end of system, last message
    block. With a plan: only the positions the plan found worth paying for. Content and
    order are left exactly as written."""
    out = copy.deepcopy(strip_cache_control(req))
    head = plan is None or plan.head
    if out.get("tools"):
        if sort_tools:
            out["tools"].sort(key=lambda t: t.get("name", ""))
        if head and (plan is None or not out.get("system")):
            out["tools"][-1]["cache_control"] = cc(ttl)
    if out.get("system"):
        out["system"] = _blocks(out["system"])
        if head:
            out["system"][-1]["cache_control"] = cc(ttl)
    flat = []
    for m in out.get("messages", []):
        m["content"] = [dict(b) for b in _blocks(m["content"])]
        flat += m["content"]
    if flat:
        last = len(flat) - 1
        idx = {last} if plan is None else {last - o for o in plan.offsets}
        if plan and plan.const_prefix:
            idx.add(plan.const_prefix - 1)
        for i in sorted(i for i in idx if 0 <= i <= last)[-3:]:
            flat[i]["cache_control"] = cc(ttl)
    return out


def canonicalize_request(req: dict, tpl: Template | None, plan: "BreakPlan | None" = None,
                         ttl: int = FIVE_MIN) -> dict:
    """Sorted tools; static system text; shared values in a second system block;
    volatile values at the end of the latest user message; four breakpoints."""
    applied = apply_template(tpl, system_text(req)) if tpl else None
    hoisted = sum(len(v) for _, v in (applied[1] + applied[2])) if applied else 0
    if applied is None or not hoisted or hoisted > 0.5 * len(system_text(req)):
        # nothing to hoist, or the "template" is mostly variable: leave the content alone
        return place_breakpoints(req, sort_tools=True, plan=plan, ttl=ttl)
    static, shared, volatile = applied
    out = copy.deepcopy(strip_cache_control(req))
    if out.get("tools"):
        out["tools"].sort(key=lambda t: t.get("name", ""))
        out["tools"][-1]["cache_control"] = cc(ttl)
    out["system"] = [{"type": "text", "text": static + "\n\n" + SLOT_NOTE, "cache_control": cc(ttl)}]
    if shared:
        out["system"].append({"type": "text", "text": _fmt(shared), "cache_control": cc(ttl)})
    msgs = out.get("messages", [])
    for m in msgs:
        m["content"] = _blocks(m["content"])
    if msgs and msgs[-1]["content"]:
        last = msgs[-1]["content"]
        last[-1] = dict(last[-1], cache_control=cc(ttl))
        if volatile:
            ctx = {"type": "text", "text": "<request_context>\n" + _fmt(volatile) + "\n</request_context>"}
            if msgs[-1]["role"] == "user":
                last.append(ctx)                 # after the breakpoint: never cached
            else:
                out["system"].append(ctx)        # assistant prefill: keep it in system
    return out
