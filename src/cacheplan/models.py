"""Model-specific facts: the minimum prompt length that can be cached.

Source: Anthropic's prompt-caching documentation, read on 23 November 2026. A prompt
shorter than the minimum is processed without caching and no error is returned.
"""
from __future__ import annotations

DEFAULT_MINIMUM = 1024

# Checked in order; the first fragment found in the model id wins.
_MINIMUMS = [
    ("mythos-preview", 2048),
    ("haiku-4-5", 4096),
    ("opus-4-5", 4096), ("opus-4-6", 4096),
    ("opus-4-7", 2048),
    ("opus-4-8", 1024), ("opus-4", 1024),
    ("sonnet-4", 1024),
    ("sonnet-5-5", 512),
    ("sonnet-5", 1024),
    ("opus-5", 512), ("haiku-5", 512), ("fable-5", 512), ("mythos-5", 512),
]


def min_cacheable(model: str | None) -> tuple[int, bool]:
    """(minimum cacheable tokens, whether the model was recognised)."""
    name = (model or "").lower().replace(".", "-")
    for fragment, tokens in _MINIMUMS:
        if fragment in name:
            return tokens, True
    return DEFAULT_MINIMUM, False


# When a request defines tools, the API adds its own tool-use instructions to the prompt.
# They are billed as input tokens and count towards the cacheable prefix. The size is
# model-specific and a few hundred tokens. 346 is the figure recalled from Anthropic's
# documentation for earlier models; it has not been verified for current ones, and one
# live capture (CrewAI, claude-sonnet-5-5, 23 November 2026) is consistent with it.
TOOL_USE_OVERHEAD = 346
