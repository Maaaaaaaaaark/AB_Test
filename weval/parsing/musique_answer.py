"""MuSiQue answer parsing: short-answer extraction + NOT FOUND detection.

The answer is extracted from the answer channel with the shared answer-line
rules (weval/parsing/answer_line.py): the last "Answer: X" line, else a
single-line reply, else an invalid-format failure (scored 0, reported).

NOT FOUND detection is strict and pre-fixed: the EXTRACTED answer (after
trailing stop-string trimming) must equal "not found" case-insensitively. No
punctuation stripping, no substring matching, no synonym matching. Anything
else is treated as an answer attempt and scored against the gold answers by
the vendored official normalization.
"""

from __future__ import annotations

from typing import List, Optional

from .answer_line import ANSWER_LINE_VERSION, extract_short_answer
from .base import ParsedAnswer, missing_final
from .reasoning import split_reasoning

MUSIQUE_PARSER_VERSION = "musique-answer-parser-v2"
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
    if final.strip() == "":
        # empty final region with a closed channel: no answer attempt
        return missing_final("musique_local", "musique_answer", raw_text, split.cleaned_text, split.region_state)

    value, source = extract_short_answer(final)
    diagnostics = {
        "answer_source": source,
        "reasoning_present": split.reasoning is not None,
        "not_found_literal": NOT_FOUND,
        "parser_version": MUSIQUE_PARSER_VERSION,
        "answer_line_version": ANSWER_LINE_VERSION,
    }
    if value is None:
        diagnostics["is_not_found"] = False
        return ParsedAnswer(
            task="musique_local",
            parser="musique_answer",
            raw_text=raw_text,
            cleaned_text=split.cleaned_text,
            final_region=final,
            region_state=split.region_state,
            value=None,
            status="invalid_format",
            diagnostics=diagnostics,
        )
    diagnostics["is_not_found"] = value.strip().lower() == NOT_FOUND.lower()
    return ParsedAnswer(
        task="musique_local",
        parser="musique_answer",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=final,
        region_state=split.region_state,
        value=value.strip(),
        status="ok",
        diagnostics=diagnostics,
    )
