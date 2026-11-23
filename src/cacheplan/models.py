"""Model-specific facts used by the forecast.

Source: Anthropic's prompt-caching and pricing documentation, read on 23 November 2026.
Check current values before relying on them; `--min-tokens` and `--price` override them.

  minimum        shortest prompt that can be cached; shorter prompts are processed
                 without caching and no error is returned
  tool_overhead  tokens of tool-use instructions the API adds when a request defines tools
                 (tool_choice auto); they are billed and count towards the cacheable length
  input_price    USD per million ordinary input tokens
  read_x         price of a cache read relative to an ordinary input token
Cache writes cost 1.25x (5-minute lifetime) or 2x (1-hour lifetime) on every model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    minimum: int = 1024
    tool_overhead: int = 300
    input_price: float | None = None
    read_x: float = 0.10
    known: bool = False


# Checked in order; the first fragment found in the model id wins.
_SPECS = [
    ("fable-5-2", ModelSpec(512, 286, 8.0, 0.025, True)), ("mythos-5-2", ModelSpec(512, 286, 8.0, 0.025, True)),
    ("fable-5-1", ModelSpec(512, 286, 10.0, 0.025, True)), ("mythos-5-1", ModelSpec(512, 286, 10.0, 0.025, True)),
    ("mythos-preview", ModelSpec(2048, 300, None, 0.10, True)),
    ("fable-5", ModelSpec(512, 286, 10.0, 0.10, True)), ("mythos-5", ModelSpec(512, 286, 10.0, 0.10, True)),
    ("opus-5-5", ModelSpec(512, 286, 4.0, 0.05, True)),
    ("opus-5", ModelSpec(512, 286, 5.0, 0.10, True)),
    ("sonnet-5-5", ModelSpec(512, 286, 2.0, 0.05, True)),
    ("sonnet-5", ModelSpec(1024, 354, 2.0, 0.10, True)),
    ("haiku-5", ModelSpec(512, 286, 0.10, 0.10, True)),
    ("opus-4-8", ModelSpec(1024, 290, 5.0, 0.10, True)),
    ("opus-4-7", ModelSpec(2048, 675, 5.0, 0.10, True)),
    ("opus-4-6", ModelSpec(4096, 497, 5.0, 0.10, True)),
    ("opus-4-5", ModelSpec(4096, 496, 5.0, 0.10, True)),
    ("opus-4", ModelSpec(1024, 313, 15.0, 0.10, True)),
    ("sonnet-4-6", ModelSpec(1024, 497, 3.0, 0.10, True)),
    ("sonnet-4-5", ModelSpec(1024, 496, 3.0, 0.10, True)),
    ("sonnet-4", ModelSpec(1024, 313, 3.0, 0.10, True)),
    ("haiku-4-5", ModelSpec(4096, 496, 1.0, 0.10, True)),
]
DEFAULT = ModelSpec()
DEFAULT_MINIMUM = DEFAULT.minimum
TOOL_USE_OVERHEAD = DEFAULT.tool_overhead


def spec(model: str | None) -> ModelSpec:
    name = (model or "").lower().replace(".", "-")
    for fragment, s in _SPECS:
        if fragment in name:
            return s
    return DEFAULT


def min_cacheable(model: str | None) -> tuple[int, bool]:
    """(minimum cacheable tokens, whether the model was recognised)."""
    s = spec(model)
    return s.minimum, s.known
