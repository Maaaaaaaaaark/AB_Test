"""MuSiQue answer parsing: short-answer extraction + NOT FOUND detection.

The answer is extracted from the answer channel with the shared answer-line
rules (weval/parsing/answer_line.py): the last "Answer: X" line, else a
single-line reply, else an invalid-format failure (scored 0, reported).

NOT FOUND detection (rule revised after the pilot, frozen before
selection): the EXTRACTED answer counts as an abstention when, after
  * lower-casing,
  * turning underscores/hyphens into spaces (Qwen3 wrote "NOT_FOUND"),
  * stripping surrounding whitespace, quotes, brackets, markdown and
    punctuation ("NOT FOUND.", "**NOT FOUND**", '"NOT FOUND"'),
  * collapsing whitespace,
it equals exactly "not found". Still NO substring matching and NO synonym
matching: "not found, sorry", "I think NOT FOUND", "notfound" and free-text
refusals such as "The provided information is insufficient ..." remain
answer attempts (scored as such). The raw variant is recorded in
diagnostics.not_found_variant whenever it differs from "NOT FOUND".
"""

from __future__ import annotations

import re
from typing import List, Optional

from .answer_line import ANSWER_LINE_VERSION, extract_short_answer
from .base import ParsedAnswer, missing_final
from .reasoning import split_reasoning

MUSIQUE_PARSER_VERSION = "musique-answer-parser-v3"
NOT_FOUND = "NOT FOUND"

_NF_SEPARATORS = re.compile(r"[_\-]+")
_NF_EDGE_CHARS = " \t\r\n\"'`*_.,;:!?()[]{}<>"


def is_not_found_answer(value: Optional[str]) -> bool:
    """Exact NOT FOUND up to case, _/- separators and edge punctuation."""
    if value is None:
        return False
    v = _NF_SEPARATORS.sub(" ", value.strip().lower())
    v = v.strip(_NF_EDGE_CHARS)
    v = re.sub(r"\s+", " ", v)
    return v == "not found"


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
    diagnostics["is_not_found"] = is_not_found_answer(value)
    if diagnostics["is_not_found"] and value.strip() != NOT_FOUND:
        diagnostics["not_found_variant"] = value.strip()
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
