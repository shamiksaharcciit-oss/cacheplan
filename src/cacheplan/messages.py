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
import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

EPHEMERAL = {"type": "ephemeral"}
FIVE_MIN, ONE_HOUR = 300, 3600
CHARS_PER_TOKEN = 3.5        # rough. Live runs: about 3.4 to 3.5 for prose-heavy prompts once the
                             # tool-use overhead is counted separately, about 2.6 for code-heavy ones.


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
    return _strip({k: req[k] for k in ("tools", "system", "messages", "uncached_tail") if req.get(k)})


def tail_text(req: dict) -> str:
    """Text the provider places after the conversation and never caches (top-level
    `instructions` on OpenAI GPT-6.2 and later). It is billed in full on every request."""
    return "".join(json.dumps(_strip(b), sort_keys=True) for b in req.get("uncached_tail") or [])


# --------------------------------------------------------------------------- cache regions
@dataclass(frozen=True)
class Segment:
    """One block of a request as the provider's cache sees it."""
    path: str                 # e.g. system[0] or messages[8].content[0]
    text: str                 # the block, serialised without cache markers
    digest: bytes             # of text; equal digests mean equal blocks
    tokens: int               # estimate, before calibration
    marker: int               # lifetime of the breakpoint on this block (FIVE_MIN, ONE_HOUR) or 0
    unit: str                 # the message it belongs to: "tools", "system" or "messages[i]"

    @property
    def region(self) -> str:
        return self.unit.split("[")[0]


@dataclass
class Rendered:
    """A request in the order the provider hashes it (tools, system, messages), plus what it
    bills but never caches. Built once per request; every caching option is then a different
    set of breakpoint positions over the same segments."""
    segments: list[Segment]
    hashes: list[bytes]       # hashes[i] identifies the prefix that ends with segment i
    cum: list[int]            # estimated tokens up to and including segment i
    tail_tokens: int          # billed after the conversation, never cached (see tail_text)
    has_tools: bool

    @property
    def tokens(self) -> int:
        return (self.cum[-1] if self.cum else 0) + self.tail_tokens

    @property
    def head(self) -> int:
        """Number of leading tools and system segments."""
        return sum(1 for s in self.segments if s.region != "messages")

    def markers(self) -> list[tuple[int, int]]:
        """(segment index, lifetime) of the breakpoints the request carries."""
        return [(i, s.marker) for i, s in enumerate(self.segments) if s.marker]

    def ends_message(self, i: int) -> bool:
        s = self.segments[i]
        return s.region != "tools" and (i + 1 == len(self.segments) or self.segments[i + 1].unit != s.unit)


_POOL: dict[object, object] = {}              # a block's text or digest -> the one shared copy of it
_POOLED: dict[int, object] = {}               # id of a shared copy -> itself (keeps ids unique)
_CORE: dict[tuple[str, int], tuple[str, bytes, int]] = {}


def _share(block):
    """Requests in a log repeat most of their content. Keeping one copy of each distinct
    block cuts memory several-fold and lets its serialised form be computed once."""
    if isinstance(block, str):
        key = block
    elif isinstance(block, dict):
        key = hashlib.blake2b(json.dumps(block, sort_keys=True).encode(), digest_size=16).digest()
    else:
        return block
    kept = _POOL.get(key)
    if kept is None:
        kept = _POOL[key] = block
        _POOLED[id(kept)] = kept
    return kept


def share_blocks(req: dict) -> dict:
    """Replace the blocks of a freshly parsed request with shared copies. The request must
    not be modified in place afterwards; everything in this package copies before changing."""
    out = dict(req)
    for name in ("tools", "uncached_tail"):
        if isinstance(out.get(name), list):
            out[name] = [_share(t) for t in out[name]]
    s = out.get("system")
    if isinstance(s, (str, list)):
        out["system"] = _share(s) if isinstance(s, str) else [_share(b) for b in s]
    if isinstance(out.get("messages"), list):
        msgs = []
        for m in out["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), (str, list)):
                c = m["content"]
                m = dict(m, content=_share(c) if isinstance(c, str) else [_share(b) for b in c])
            msgs.append(m)
        out["messages"] = msgs
    return out


def _core(prefix: str, block) -> tuple[str, bytes, int]:
    """(text, digest, tokens) of one block. Cached for shared blocks."""
    shared = _POOLED.get(id(block)) is block
    if shared:
        hit = _CORE.get((prefix, id(block)))
        if hit:
            return hit
    body = {"type": "text", "text": block} if isinstance(block, str) else _strip(block)
    text = prefix + ":" + json.dumps(body, sort_keys=True)
    core = (text, hashlib.blake2b(text.encode(), digest_size=12).digest(), est_tokens(text))
    if shared:
        _CORE[(prefix, id(block))] = core
    return core


def _marker(block) -> int:
    return _ttl(block) if isinstance(block, dict) else 0


def render(req: dict) -> Rendered:
    segs: list[Segment] = []

    def add(path, prefix, block, unit):
        text, digest, tokens = _core(prefix, block)
        segs.append(Segment(path, text, digest, tokens, _marker(block), unit))

    for i, t in enumerate(req.get("tools") or []):
        add(f"tools[{i}]", "tool", t, "tools")
    s = req.get("system")
    for i, b in enumerate(([s] if isinstance(s, str) else list(s)) if s else []):
        add(f"system[{i}]", "system", b, "system")
    for i, m in enumerate(req.get("messages") or []):
        c = m["content"]
        for j, b in enumerate([c] if isinstance(c, str) else list(c)):
            add(f"messages[{i}].content[{j}]", m["role"], b, f"messages[{i}]")
    if req.get("cache_control") and segs:            # top-level field = automatic caching
        last = segs[-1]
        segs[-1] = Segment(last.path, last.text, last.digest, last.tokens, _ttl(req), last.unit)
    h, hashes, cum, total = b"", [], [], 0
    for seg in segs:
        h = hashlib.blake2b(h + seg.digest, digest_size=12).digest()
        total += seg.tokens
        hashes.append(h)
        cum.append(total)
    tail = est_tokens(tail_text(req)) if req.get("uncached_tail") else 0
    return Rendered(segs, hashes, cum, tail, bool(req.get("tools")))


def rendered(row: dict) -> Rendered:
    """The logged request of a row. Rows that came from logs.load() keep the rendering;
    rows built by hand are left untouched and rendered each time."""
    if "_rendered" not in row:
        return render(row["request"])
    r = row["_rendered"]
    if r is None:
        r = row["_rendered"] = render(row["request"])
    return r


def plan_positions(r: Rendered, plan: "BreakPlan | None", ttl: int = FIVE_MIN) -> list[tuple[int, int]]:
    """Where place_breakpoints puts its markers, as segment indices, without building a request."""
    head = plan is None or plan.head
    regions = [s.region for s in r.segments]
    n_tools, n_system = regions.count("tools"), regions.count("system")
    out = []
    if n_tools and head and (plan is None or not n_system):
        out.append(n_tools - 1)
    if n_system and head:
        out.append(n_tools + n_system - 1)
    first = n_tools + n_system
    count = len(r.segments) - first
    if count:
        last = count - 1
        idx = {last} if plan is None else {last - o for o in plan.offsets}
        if plan and plan.const_prefix:
            idx.add(plan.const_prefix - 1)
        out += [first + i for i in sorted(i for i in idx if 0 <= i <= last)[-3:]]
    return [(i, ttl) for i in out]


def prefix_blocks(req: dict) -> list[tuple[str, int]]:
    """(serialised block, lifetime of its breakpoint in seconds or 0) in hashing order."""
    return [(s.text, s.marker) for s in render(req).segments]


def located_blocks(req: dict) -> list[tuple[str, str, int]]:
    """(path, serialised block, breakpoint lifetime) in the order the API hashes them:
    tools, then system, then messages."""
    return [(s.path, s.text, s.marker) for s in render(req).segments]


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
        bl = [x.digest for x in r.segments] if isinstance(r, Rendered) else [t for t, _ in prefix_blocks(r)]
        seqs.append(bl)
        heads.append(r.head if isinstance(r, Rendered) else sum(1 for t in bl if t.startswith(("tool:", "system:"))))
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


_CLEAN: dict[int, object] = {}


def _clean(block):
    """The block without cache markers; the same object when it has none. Shared blocks are
    checked once."""
    if not isinstance(block, (dict, list)):
        return block
    shared = _POOLED.get(id(block)) is block
    if shared and id(block) in _CLEAN:
        return _CLEAN[id(block)]
    out = _strip(block) if "cache_control" in json.dumps(block) else block
    if shared:
        _CLEAN[id(block)] = out
    return out


def _unmarked(req: dict) -> dict:
    """A copy of the request without cache markers. Lists and messages are new; blocks that
    need no change are the same objects, so nothing here may be modified in place."""
    out = {}
    if req.get("tools"):
        out["tools"] = [_clean(t) for t in req["tools"]]
    s = req.get("system")
    if s:
        out["system"] = s if isinstance(s, str) else [_clean(b) for b in s]
    if req.get("messages"):
        out["messages"] = [dict(m, content=m["content"] if isinstance(m["content"], str) else [_clean(b) for b in m["content"]])
                           for m in req["messages"]]
    if req.get("uncached_tail"):
        out["uncached_tail"] = list(req["uncached_tail"])
    return out


def _mark_block(msgs: list[dict], i: int, j: int, ttl: int) -> None:
    """Put a breakpoint on block j of message i (messages are this call's own copies)."""
    c = msgs[i]["content"]
    if isinstance(c, str):
        msgs[i]["content"] = [{"type": "text", "text": c, "cache_control": cc(ttl)}]
    else:
        c[j] = dict(c[j], cache_control=cc(ttl))


def _block_positions(msgs: list[dict]) -> list[tuple[int, int]]:
    return [(i, j) for i, m in enumerate(msgs)
            for j in range(1 if isinstance(m["content"], str) else len(m["content"]))]


def place_breakpoints(req: dict, sort_tools: bool = False, plan: BreakPlan | None = None,
                      ttl: int = FIVE_MIN) -> dict:
    """At most four breakpoints. Without a plan: end of tools, end of system, last message
    block. With a plan: only the positions the plan found worth paying for. Content and
    order are left exactly as written."""
    out = _unmarked(req)
    head = plan is None or plan.head
    if out.get("tools"):
        if sort_tools:
            out["tools"].sort(key=lambda t: t.get("name", ""))
        if head and (plan is None or not out.get("system")):
            out["tools"][-1] = dict(out["tools"][-1], cache_control=cc(ttl))
    if out.get("system"):
        out["system"] = _blocks(out["system"])
        if head:
            out["system"][-1] = dict(out["system"][-1], cache_control=cc(ttl))
    msgs = out.get("messages", [])
    flat = _block_positions(msgs)
    if flat:
        last = len(flat) - 1
        idx = {last} if plan is None else {last - o for o in plan.offsets}
        if plan and plan.const_prefix:
            idx.add(plan.const_prefix - 1)
        for n in sorted(i for i in idx if 0 <= i <= last)[-3:]:
            _mark_block(msgs, *flat[n], ttl)
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
    out = _unmarked(req)
    if out.get("tools"):
        out["tools"].sort(key=lambda t: t.get("name", ""))
        out["tools"][-1] = dict(out["tools"][-1], cache_control=cc(ttl))
    out["system"] = [{"type": "text", "text": static + "\n\n" + SLOT_NOTE, "cache_control": cc(ttl)}]
    if shared:
        out["system"].append({"type": "text", "text": _fmt(shared), "cache_control": cc(ttl)})
    msgs = out.get("messages", [])
    flat = _block_positions(msgs)
    if flat:
        i, j = flat[-1]
        if i == len(msgs) - 1:
            _mark_block(msgs, i, j, ttl)
            if volatile:
                ctx = {"type": "text", "text": "<request_context>\n" + _fmt(volatile) + "\n</request_context>"}
                if msgs[-1]["role"] == "user":
                    msgs[-1]["content"].append(ctx)      # after the breakpoint: never cached
                else:
                    out["system"].append(ctx)            # assistant prefill: keep it in system
    return out
