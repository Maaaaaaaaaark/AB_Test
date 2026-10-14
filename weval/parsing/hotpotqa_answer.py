"""HotpotQA answer parsing: short-answer extraction.

The answer is extracted from the answer channel with the shared answer-line
rules (weval/parsing/answer_line.py): the last "Answer: X" line, else a
single-line reply, else an invalid-format failure (scored 0, reported).
Official normalization and EM/F1 are applied by the scorer (vendored
hotpot_evaluate_v1.py). Note: the official normalization does not map
digits to number words ("5" vs "five" is a miss) - kept as official.
"""

from __future__ import annotations

from typing import List, Optional

from .answer_line import ANSWER_LINE_VERSION, extract_short_answer
from .base import ParsedAnswer, missing_final
from .reasoning import split_reasoning

HOTPOTQA_PARSER_VERSION = "hotpotqa-answer-parser-v3"


def parse_hotpotqa_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    if split.region_state == "open_at_eos":
        return missing_final("hotpotqa", "hotpotqa_answer_line", raw_text, split.cleaned_text, split.region_state)
    final = split.final.strip()
    if final == "":
        return missing_final("hotpotqa", "hotpotqa_answer_line", raw_text, split.cleaned_text, split.region_state)
    value, source = extract_short_answer(split.final)
    return ParsedAnswer(
        task="hotpotqa",
        parser="hotpotqa_answer_line",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=split.final,
        region_state=split.region_state,
        value=value,
        status="ok" if value is not None else "invalid_format",
        diagnostics={
            "answer_source": source,
            "reasoning_present": split.reasoning is not None,
            "parser_version": HOTPOTQA_PARSER_VERSION,
            "answer_line_version": ANSWER_LINE_VERSION,
        },
    )
