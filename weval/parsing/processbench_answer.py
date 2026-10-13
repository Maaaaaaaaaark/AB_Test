"""ProcessBench answer parsing: official boxed-index extraction.

Official (vendored run_eval.py): find all \\boxed{...}, take the LAST match,
strip, int() it; any failure => None (scored incorrect for both error and
correct solutions). The official evaluator applies this to the FULL generated
text; we do the same for the primary, and record the final-region extraction
as a diagnostic.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .base import ParsedAnswer
from .reasoning import split_reasoning

PROCESSBENCH_PARSER_VERSION = "processbench-boxed-parser-v1"

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def official_extract_index(text: str) -> Optional[int]:
    """Official extraction: last boxed match -> int, else None."""
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1].strip())
    except (ValueError, TypeError):
        return None


def parse_processbench_answer(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ParsedAnswer:
    split = split_reasoning(raw_text, reasoning_mode, eos_strings)
    pred = official_extract_index(split.cleaned_text)
    pred_final = official_extract_index(split.final) if split.final else None

    if split.region_state == "open_at_eos":
        pred = None
        status = "missing_final"
    elif pred is None:
        status = "invalid_format"
    else:
        status = "ok"

    return ParsedAnswer(
        task="processbench",
        parser="processbench_boxed_index",
        raw_text=raw_text,
        cleaned_text=split.cleaned_text,
        final_region=split.final,
        region_state=split.region_state,
        value=pred,
        status=status,
        diagnostics={
            "index_in_final_region": pred_final,
            "full_and_final_agree": pred_final == pred,
            "region_state": split.region_state,
            "reasoning_present": split.reasoning is not None,
            "parser_version": PROCESSBENCH_PARSER_VERSION,
        },
    )
