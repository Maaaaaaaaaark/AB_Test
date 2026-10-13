"""Common eligibility rule: context-length check before generation.

prompt_tokens + max_new_tokens > model context limit  =>  the item is
EXCLUDED (recorded with the exact numbers), never silently truncated.

The check needs the REAL tokenizer; the scripted (dry-run) backend cannot do
it and the skip is recorded explicitly, so dry-run eligibility columns are
never mistaken for real ones.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..models.backend import GenerationBackend
from ..models.scripted import ScriptedBackend


def check_context_fits(
    backend: GenerationBackend,
    prompt_text: str,
    max_new_tokens: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Returns (fits, info). info carries the numbers and the reason."""
    if isinstance(backend, ScriptedBackend):
        return True, {
            "check": "skipped_synthetic_tokenizer",
            "reason": "scripted backend has no real tokenizer; the check runs on the cluster",
        }
    limit = getattr(backend, "context_limit", None)
    if limit is None:
        return True, {
            "check": "skipped_no_context_limit",
            "reason": "context limit could not be resolved from the model config",
        }
    n_prompt = len(backend.tokenize(prompt_text))
    required = n_prompt + max_new_tokens
    fits = required <= limit
    info: Dict[str, Any] = {
        "check": "passed" if fits else "excluded_context_too_long",
        "context_limit": limit,
        "n_prompt_tokens": n_prompt,
        "max_new_tokens": max_new_tokens,
        "required_tokens": required,
    }
    if not fits:
        info["reason"] = f"{required} tokens required, limit is {limit}"
    return fits, info
