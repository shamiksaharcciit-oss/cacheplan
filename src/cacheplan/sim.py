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

from .messages import ONE_HOUR, est_tokens, prefix_blocks

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

    def cost_units(self) -> float:
        """Cost in units of one ordinary input token."""
        w1h = self.cache_creation_1h_tokens
        w5m = self.cache_creation_input_tokens - w1h
        return self.input_tokens + w5m * WRITE_5M + w1h * WRITE_1H + self.cache_read_input_tokens * READ


class CacheSim:
    def __init__(self, min_tokens: int = 1024):
        self.min_tokens = min_tokens
        self.entries: dict[bytes, tuple[float, int]] = {}     # prefix hash -> (expiry, lifetime)

    def request(self, req: dict, now: float) -> Usage:
        blocks = prefix_blocks(req)
        h, hashes, cum, total = b"", [], [], 0
        for text, _ in blocks:
            h = hashlib.blake2b(h + text.encode(), digest_size=12).digest()
            total += est_tokens(text)
            hashes.append(h)
            cum.append(total)
        bps = [(i, ttl) for i, (_, ttl) in enumerate(blocks) if ttl and cum[i] >= self.min_tokens]
        if not bps:
            return Usage(input_tokens=total)
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
        end = max(bps[-1][0], read)
        r = cum[read] if read >= 0 else 0
        w = cum[end] - r
        long_lived = bool(written) and written[-1][1] >= ONE_HOUR
        return Usage(total - cum[end], w, r, w if long_lived else 0)
