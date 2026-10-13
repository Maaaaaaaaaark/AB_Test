"""GSM8K answer parsing.

Primary (official parity): the vendored ANS_RE extraction applied to the full
cleaned completion - exactly what the official GSM8K evaluator does (first
"#### <number>" match, commas stripped, INVALID_ANS when absent).

Secondary (diagnostic only, never the primary score): a relaxed last-number
extraction on the final answer-channel region. The extraction failure rate is
reported; relaxed matches never replace official extraction.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..datasets.gsm8k import ANS_RE, INVALID_ANS, extract_answer
from .base import ParsedAnswer
from .reasoning import split_reasoning

GSM8K_PARSER_VERSION = "gsm8k-answer-parser-v1"

# Relaxed diagnostic: the last numeric literal (optionally comma-grouped or
# with a decimal point) in the final region. Diagnostic only.
RELAXED_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
# Reasoning-region contamination diagnostic: does the official match fall
# inside a thinking region?


def parse_gsm8k_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    official = extract_answer(split.cleaned_text)
    is_invalid = official == INVALID_ANS

    relaxed = None
    relaxed_source = "none"
    if split.final:
        m = RELAXED_NUM_RE.findall(split.final)
        if m:
            relaxed = m[-1].replace(",", "")
            relaxed_source = "last_number_in_final_region"
    elif not is_invalid:
        relaxed = official
        relaxed_source = "official_fallback"

    # Where did the official match land relative to the reasoning region?
    match_in_reasoning = False
    if not is_invalid and split.reasoning is not None:
        m = ANS_RE.search(split.cleaned_text)
        if m is not None:
            match_in_reasoning = m.start() < len(split.reasoning)

    status = "invalid_format" if is_invalid else "ok"
    value = None if is_invalid else official
    return ParsedAnswer(
        task="gsm8k",
        parser="gsm8k_official_regex",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=split.final,
        region_state=split.region_state,
        value=value,
        status=status,
        diagnostics={
            "official_extracted": official if not is_invalid else None,
            "official_invalid": is_invalid,
            "relaxed_extracted": relaxed,
            "relaxed_source": relaxed_source,
            "official_match_in_reasoning_region": match_in_reasoning,
            "region_state": split.region_state,
            "reasoning_present": split.reasoning is not None,
            "parser_version": GSM8K_PARSER_VERSION,
        },
    )
