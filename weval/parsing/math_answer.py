"""MATH answer parsing: \\boxed{} extraction (equivalence is scored separately).

Extraction is a separate step from equivalence checking. The primary
extraction applies to the final answer-channel region (thinking models must
close their reasoning before the answer counts); the full-text extraction is
recorded as a diagnostic. Nested braces are handled by the vendored official
helper when available, with a fallback greedy matcher that yields identical
results on the official test cases.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .base import ParsedAnswer
from .reasoning import split_reasoning

MATH_PARSER_VERSION = "math-answer-parser-v1"


def extract_boxed_last(text: str) -> Optional[str]:
    """Last \\boxed{...} with nested-brace support.

    Mirrors the official MATH eval harness helper
    (hendrycks/math math_eval/utilities.py last_boxed_only_string +
    remove_boxed): rfind the marker, then track brace depth. The content
    between the matching braces is returned without the wrapper.
    """
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def parse_math_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    boxed_final = extract_boxed_last(split.final) if split.final else None
    boxed_full = extract_boxed_last(split.cleaned_text) if split.cleaned_text else None

    status = "ok" if boxed_final is not None else ("missing_final" if split.region_state == "open_at_eos" else "invalid_format")
    if split.region_state == "open_at_eos":
        # open reasoning region: nothing in the answer channel counts
        boxed_final = None
        status = "missing_final"

    return ParsedAnswer(
        task="math",
        parser="math_boxed",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=split.final,
        region_state=split.region_state,
        value=boxed_final,
        status=status,
        diagnostics={
            "boxed_in_final_region": boxed_final,
            "boxed_in_full_text": boxed_full,
            "full_and_final_agree": boxed_full == boxed_final,
            "region_state": split.region_state,
            "reasoning_present": split.reasoning is not None,
            "parser_version": MATH_PARSER_VERSION,
        },
    )
