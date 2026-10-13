"""MuSiQue answer parsing: NOT FOUND detection + short answer.

NOT FOUND detection is strict and pre-fixed: after whitespace trimming (and
eos-token trimming by the reasoning splitter), the ENTIRE final region must
equal "not found" case-insensitively. No punctuation stripping, no substring
matching, no synonym matching. Anything else is treated as an answer attempt
and scored against the gold answers by the vendored official normalization.
"""

from __future__ import annotations

from typing import List, Optional

from .base import ParsedAnswer, missing_final
from .reasoning import split_reasoning

MUSIQUE_PARSER_VERSION = "musique-answer-parser-v1"
NOT_FOUND = "NOT FOUND"


def parse_musique_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    final = split.final
    if split.region_state == "open_at_eos":
        return missing_final("musique_local", "musique_answer", raw_text, split.cleaned_text, split.region_state)

    is_not_found = final.strip().lower() == NOT_FOUND.lower()
    if final.strip() == "":
        # empty final region with a closed channel: no answer attempt
        return missing_final("musique_local", "musique_answer", raw_text, split.cleaned_text, split.region_state)

    return ParsedAnswer(
        task="musique_local",
        parser="musique_answer",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=final,
        region_state=split.region_state,
        value=final.strip(),
        status="ok",
        diagnostics={
            "is_not_found": is_not_found,
            "not_found_literal": NOT_FOUND,
            "reasoning_present": split.reasoning is not None,
            "parser_version": MUSIQUE_PARSER_VERSION,
        },
    )
