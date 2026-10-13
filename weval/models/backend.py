"""Generation backend interface.

A backend turns (prompt text, seed, budget, deadline) into a GenerationResult.
Exactly two backends exist:

  * ScriptedBackend  - deterministic fake outputs for tests and dry-runs
                       (no torch; token ids are synthetic and labeled as such),
  * TransformersBackend - the real cluster implementation (BF16, eval mode,
                       batch 1, merged generation config recorded, deadline
                       enforced during decoding).

Retry policy lives in the runner: at most 2 retries per item+seed and ONLY for
infrastructure failures (timeout / OOM / crash). A finished generation with a
wrong or empty answer is a result, never retried. Token-limit termination is
an evaluated outcome (finish_reason "length"), not an infrastructure failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class InfrastructureError(RuntimeError):
    """Timeout / OOM / crash. Retryable (<=2 per item+seed); no partial output
    is ever scored."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "timeout" | "oom" | "crash"


class DeadlineExceeded(InfrastructureError):
    def __init__(self, deadline_seconds: float, elapsed_seconds: float):
        super().__init__(
            "timeout",
            f"generation deadline {deadline_seconds:.1f}s exceeded after {elapsed_seconds:.1f}s",
        )
        self.deadline_seconds = deadline_seconds
        self.elapsed_seconds = elapsed_seconds


@dataclass
class GenerationResult:
    item_id: str
    seed: int
    item_seed: int
    prompt_text: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    output_text: str
    finish_reason: str  # "eos" | "length"
    n_prompt_tokens: int
    n_generated_tokens: int
    wall_seconds: float
    backend: str
    tokenizer_used: str  # real tokenizer name/path, or "synthetic" for scripted
    merged_generation_params: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "seed": self.seed,
            "item_seed": self.item_seed,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "output_text": self.output_text,
            "finish_reason": self.finish_reason,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_generated_tokens": self.n_generated_tokens,
            "wall_seconds": self.wall_seconds,
            "backend": self.backend,
            "tokenizer_used": self.tokenizer_used,
            "merged_generation_params": self.merged_generation_params,
            "attempts": self.attempts,
            "extra": self.extra,
        }


class GenerationBackend(Protocol):
    """Generates one output for one (item, seed)."""

    name: str

    def cache_identity(self) -> Dict[str, str]:
        """Stable identity included in prediction-cache fingerprints."""
        ...

    def generate(
        self,
        item_id: str,
        seed: int,
        prompt_text: str,
        max_new_tokens: int,
        deadline_seconds: float,
    ) -> GenerationResult:
        ...

    def close(self) -> None:
        ...


def make_item_seed(run_seed: int, item_id: str) -> int:
    """Stable per-item seed, independent of run order.

    item_seed = (run_seed * 2654435761 + first 8 bytes of sha256(item_id)) mod 2^32
    The same (run_seed, item_id) always yields the same seed regardless of
    which other items ran first.
    """
    import hashlib

    h = hashlib.sha256(item_id.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "little")
    return (run_seed * 2654435761 + x) % (2**32)


class _NullBackend:
    """Backend for eligibility-only paths (never called)."""

    name = "null"

    def generate(self, *args, **kwargs):
        raise NotImplementedError("null backend cannot generate")

    def close(self) -> None:
        pass


NULL_BACKEND = _NullBackend()
