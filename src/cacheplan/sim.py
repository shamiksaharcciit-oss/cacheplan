"""Local model of Anthropic prompt caching, used for forecasts without an API key.

Follows the documented behaviour: caching happens only at breakpoints; the prefix is
hashed block by block in the order tools -> system -> messages; a lookup at a breakpoint
also checks up to 20 earlier block boundaries; an entry lives for the lifetime of the
breakpoint that wrote it (5 minutes or 1 hour) and is refreshed on every read; prefixes
shorter than a model-specific minimum are not cached. Token counts are estimates.
It is a model, not the service: use `cacheplan verify` for real numbers.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .messages import ONE_HOUR, est_tokens, located_blocks, prefix_blocks
from .models import TOOL_USE_OVERHEAD

LOOKBACK = 20
WRITE_5M, WRITE_1H, READ = 1.25, 2.0, 0.10        # price multipliers on ordinary input tokens


@dataclass
class Usage:
    input_tokens: int = 0                    # uncached, after the last breakpoint
    cache_creation_input_tokens: int = 0     # all cache writes
    cache_read_input_tokens: int = 0
    cache_creation_1h_tokens: int = 0        # the part of the writes made with the 1-hour lifetime

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def add(self, o: "Usage") -> None:
        self.input_tokens += o.input_tokens
        self.cache_creation_input_tokens += o.cache_creation_input_tokens
        self.cache_read_input_tokens += o.cache_read_input_tokens
        self.cache_creation_1h_tokens += o.cache_creation_1h_tokens

    def cost_units(self, read_x: float = READ, write_short: float = WRITE_5M, write_long: float | None = WRITE_1H) -> float:
        """Cost in units of one ordinary input token. The multipliers are model-specific."""
        long = self.cache_creation_1h_tokens
        short = self.cache_creation_input_tokens - long
        return (self.input_tokens + short * write_short + long * (write_long or write_short)
                + self.cache_read_input_tokens * read_x)


def _unit(path: str) -> str:
    """The message a block belongs to: "tools", "system" or "messages[i]"."""
    return path.split(".")[0] if path.startswith("messages") else path.split("[")[0]


class CacheSim:
    def __init__(self, min_tokens: int = 1024, ttl_short: int = 300, ttl_long: int = 3600,
                 message_entries: bool = False):
        """ttl_short / ttl_long: seconds an entry lives for the two marker kinds
        (5 minutes and 1 hour on Anthropic; 30 minutes and 12 hours on newer OpenAI models)."""
        self.min_tokens, self.ttl_short, self.ttl_long = min_tokens, ttl_short, ttl_long
        # True: a write also makes every earlier message ending in that prefix readable
        # (measured on OpenAI gpt-5.6-luna: a new conversation read the shared system message
        # although the only write had been at the end of another conversation's first turn).
        self.message_entries = message_entries
        self.entries: dict[bytes, tuple[float, int]] = {}     # prefix hash -> (expiry, lifetime)

    def request(self, req: dict, now: float, min_tokens: int | None = None,
                overhead: int | None = None, scale: float = 1.0) -> Usage:
        """overhead: tokens the API adds when tools are defined; scale: correction applied
        to the character-based token estimate (1.0 = uncalibrated)."""
        floor = self.min_tokens if min_tokens is None else min_tokens
        extra = TOOL_USE_OVERHEAD if overhead is None else overhead
        located = located_blocks(req)
        blocks = [(text, ttl) for _, text, ttl in located]
        h, hashes, cum = b"", [], []
        total = extra if req.get("tools") else 0               # hidden tool-use instructions
        for text, _ in blocks:
            h = hashlib.blake2b(h + text.encode(), digest_size=12).digest()
            total += est_tokens(text) * scale
            hashes.append(h)
            cum.append(total)
        bps = [(i, self.ttl_long if ttl >= ONE_HOUR else self.ttl_short)
               for i, (_, ttl) in enumerate(blocks) if ttl and cum[i] >= floor]
        if not bps:
            return Usage(input_tokens=round(total))
        read = -1
        for bp, _ in reversed(bps):
            for j in range(bp, max(-1, bp - LOOKBACK - 1), -1):
                if self.entries.get(hashes[j], (0, 0))[0] > now:
                    read = j
                    break
            if read >= 0:
                break
        if read >= 0:
            life = self.entries[hashes[read]][1]
            self.entries[hashes[read]] = (now + life, life)
        written = [(bp, ttl) for bp, ttl in bps if bp > read]
        for bp, ttl in written:
            self.entries[hashes[bp]] = (now + ttl, ttl)
            if self.message_entries:
                for j in range(read + 1, bp):
                    here, after = located[j][0], located[j + 1][0]
                    ends_message = _unit(here) != "tools" and _unit(here) != _unit(after)
                    if ends_message and cum[j] >= floor and self.entries.get(hashes[j], (0, 0))[0] <= now + ttl:
                        self.entries[hashes[j]] = (now + ttl, ttl)
        end = max(bps[-1][0], read)
        r = round(cum[read]) if read >= 0 else 0
        w = round(cum[end]) - r
        long_lived = bool(written) and written[-1][1] == self.ttl_long and self.ttl_long != self.ttl_short
        return Usage(round(total) - round(cum[end]), w, r, w if long_lived else 0)


class ImplicitSim:
    """Automatic prefix caching without markers or write charges (OpenAI models before
    GPT-5.6). A request reads the longest prefix it shares with any request seen within
    the lifetime, rounded down to the model's boundary interval, if it reaches the minimum."""

    def __init__(self, ttl: float, interval: int = 128, keep: int = 200):
        self.ttl, self.interval, self.keep = ttl, max(1, interval), keep
        self.seen: list[tuple[float, str]] = []

    def request(self, req: dict, now: float, min_tokens: int = 1024, overhead: int = 0, scale: float = 1.0) -> Usage:
        import os
        text = "\x00".join(t for t, _ in prefix_blocks(req))
        total = round(est_tokens(text) * scale) + (overhead if req.get("tools") else 0)
        best = 0
        for i, (ts, prev) in enumerate(self.seen):
            if now - ts > self.ttl:
                continue
            n = len(os.path.commonprefix([prev, text]))
            if n > best:
                best = n
                self.seen[i] = (now, prev)                 # a read keeps the entry alive
        self.seen = [(ts, t) for ts, t in self.seen if now - ts <= self.ttl][-self.keep:] + [(now, text)]
        cached = round(est_tokens(text[:best]) * scale) if best else 0
        cached -= cached % self.interval
        if cached < min_tokens:
            cached = 0
        return Usage(total - cached, 0, min(cached, total))
