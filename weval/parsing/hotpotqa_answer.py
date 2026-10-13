"""HotpotQA answer parsing: free text.

The answer is the whole final answer-channel region; official normalization
and F1 are applied by the scorer (vendored hotpot_evaluate_v1.py).
"""

from __future__ import annotations

from typing import List, Optional

from .base import ParsedAnswer, missing_final
from .reasoning import split_reasoning

HOTPOTQA_PARSER_VERSION = "hotpotqa-answer-parser-v1"


def parse_hotpotqa_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    if split.region_state == "open_at_eos":
        return missing_final("hotpotqa", "hotpotqa_free_text", raw_text, split.cleaned_text, split.region_state)
    final = split.final.strip()
    if final == "":
        return missing_final("hotpotqa", "hotpotqa_free_text", raw_text, split.cleaned_text, split.region_state)
    return ParsedAnswer(
        task="hotpotqa",
        parser="hotpotqa_free_text",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=split.final,
        region_state=split.region_state,
        value=final,
        status="ok",
        diagnostics={
            "reasoning_present": split.reasoning is not None,
            "parser_version": HOTPOTQA_PARSER_VERSION,
        },
    )
