"""Scripted backend: deterministic fake generations for tests and dry-runs.

No torch. Outputs come from an optional script file (JSONL: one {"item_id",
"text"} per line) or from a deterministic generator derived from the item seed.

The generated token ids are SYNTHETIC (a hash-derived sequence, not a real
tokenization) and every record they produce is labeled tokenizer_used:
"synthetic", so compute metrics from a dry-run are never mistaken for real
measurements. The scripted backend exercises the full pipeline (render,
store, parse, score, report) without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional

from .backend import GenerationBackend, GenerationResult, make_item_seed

SCRIPTED_BACKEND_VERSION = "scripted-backend-v2"

DEFAULT_SCRIPTED_TEXT = "NOT FOUND"  # plausible for abstention plumbing checks


class ScriptedBackend(GenerationBackend):
    """Deterministic dry-run backend.

    If `script_path` is given, outputs are read from it (item_id -> text);
    entries missing from the script file use the seeded generator.
    """

    name = "scripted"

    def cache_identity(self) -> Dict[str, str]:
        return {"name": self.name, "version": SCRIPTED_BACKEND_VERSION}

    def __init__(
        self,
        script_path: Optional[Path] = None,
        default_text: str = DEFAULT_SCRIPTED_TEXT,
        reasoning_mode: Optional[str] = None,
    ):
        self.script_path = Path(script_path) if script_path else None
        self.default_text = default_text
        self.reasoning_mode = reasoning_mode
        self._script: Optional[Dict[str, str]] = None

    def _default_raw_text(self) -> str:
        """Render the semantic default as raw generated model text.

        A prefilled-thinking prompt ends inside an open reasoning region, so
        its generated text must close that region before emitting the answer.
        Explicit JSONL script entries remain raw text and are never rewritten.
        """
        if self.reasoning_mode == "prefilled_thinking":
            return f"</think>\n{self.default_text}"
        return self.default_text

    def _load_script(self) -> Dict[str, str]:
        if self._script is None:
            self._script = {}
            if self.script_path and self.script_path.exists():
                with open(self.script_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        self._script[row["item_id"]] = str(row["text"])
        return self._script

    def generate(
        self,
        item_id: str,
        seed: int,
        prompt_text: str,
        max_new_tokens: int,
        deadline_seconds: float,
    ) -> GenerationResult:
        script = self._load_script()
        text = script[item_id] if item_id in script else self._default_raw_text()
        # Synthetic token ids: deterministic per (seed, item, text) but NOT a
        # real tokenization. Length capped to look plausible, unrelated to any
        # real tokenizer.
        h = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
        n_tokens = 1 + (int.from_bytes(h[:4], "little") % 24)
        token_ids = [1 + (int.from_bytes(h[i : i + 4], "little") % 1000) for i in range(4, 4 + n_tokens * 4, 4)]
        # Prompt "tokens": a nominal count so cost plumbing is exercised.
        prompt_hash = hashlib.sha256(prompt_text.encode()).digest()
        n_prompt = 1 + (int.from_bytes(prompt_hash[:4], "little") % 64)
        prompt_ids = [1 + (int.from_bytes(prompt_hash[i : i + 4], "little") % 1000) for i in range(4, 4 + n_prompt * 4, 4)]
        item_seed = make_item_seed(seed, item_id)
        return GenerationResult(
            item_id=item_id,
            seed=seed,
            item_seed=item_seed,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_ids,
            generated_token_ids=token_ids,
            output_text=text,
            finish_reason="eos",
            n_prompt_tokens=len(prompt_ids),
            n_generated_tokens=len(token_ids),
            wall_seconds=0.0,
            backend=self.name,
            tokenizer_used="synthetic",
            merged_generation_params={
                "backend_version": SCRIPTED_BACKEND_VERSION,
                "note": "synthetic dry-run",
                "reasoning_mode": self.reasoning_mode,
            },
        )

    def eos_strings(self):
        return None  # synthetic backend: parsers trim whitespace only

    def close(self) -> None:
        pass
