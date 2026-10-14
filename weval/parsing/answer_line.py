"""Short-answer extraction for free-text QA tasks (MuSiQue local, HotpotQA).

The instructions ask for a short answer and say: "If you write anything
before your answer, put the answer alone on the last line in the form:
Answer: <short answer>". The prompt itself ends with an "Answer:" cue, so a
model that answers directly writes just the answer.

Extraction from the answer channel (after any reasoning region):

  1. answer_line   - the LAST line of the form "Answer: X" (also tolerates
                     markdown decoration such as "**Answer:** X" and
                     "Final answer: X"); if X is empty, the next non-empty
                     line is the answer;
  2. single_line   - otherwise, if the answer channel is exactly one
                     non-empty line, that line is the answer (the model
                     answered right after the prompt's "Answer:" cue);
  3. none          - otherwise (several lines, no Answer line): no answer is
                     extracted; the item is an invalid-format failure (scored
                     0, reported as a format failure, never guessed).

Inline "Answer:" marker (rule added after the pilot, frozen before
selection): if the extracted line still contains an "Answer:" marker after
other text - Gemma wrote "Belgium Answer:" and "8 April 2013 Answer: 8 April
2013" - the text after the LAST marker is the answer when it is non-empty,
otherwise the text before it. Such single-line replies are reported with
source "inline_marker" (a format deviation that is counted, not hidden).

A \\boxed{X} wrapper around the extracted answer is unwrapped. Nothing else is
rewritten: normalization is left to the official scorers.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

ANSWER_LINE_VERSION = "answer-line-v2"

ANSWER_LINE_RE = re.compile(
    r"^[\s>#*_\-]*(?:final\s+)?answer[\s*_]*[:：][\s*_]*(?P<rest>.*?)[\s*_]*$",
    re.IGNORECASE,
)
# "Answer:" appearing INSIDE a line (after other text), optionally bold.
INLINE_MARKER_RE = re.compile(r"(?:\*\*|__)?\banswer(?:\*\*|__)?\s*[:：]", re.IGNORECASE)
_BOXED_RE = re.compile(r"^\$*\s*\\boxed\{(?P<inner>.*)\}\s*\$*$")
_DECORATION = "*_` \t"


def _clean(value: str) -> str:
    v = value.strip().strip(_DECORATION).strip()
    m = _BOXED_RE.match(v)
    if m:
        v = m.group("inner").strip()
    return v


def _resolve_inline_marker(value: str) -> Tuple[Optional[str], bool]:
    """(answer, marker_found). Text after the last inline "Answer:" marker if
    non-empty, else the text before it."""
    matches = list(INLINE_MARKER_RE.finditer(value))
    if not matches:
        return value, False
    last = matches[-1]
    after = _clean(value[last.end():])
    if after:
        return after, True
    before = _clean(value[: last.start()])
    return (before or None), True


def extract_short_answer(final_region: str) -> Tuple[Optional[str], str]:
    """Return (answer or None, source) with source in
    answer_line | single_line | inline_marker | none."""
    lines = [ln for ln in (final_region or "").splitlines()]
    non_empty = [ln for ln in lines if ln.strip()]
    last_idx = None
    last_rest = None
    for i, ln in enumerate(lines):
        m = ANSWER_LINE_RE.match(ln)
        if m:
            last_idx, last_rest = i, m.group("rest")
    if last_idx is not None:
        rest = _clean(last_rest or "")
        if not rest:
            following = [ln for ln in lines[last_idx + 1:] if ln.strip()]
            rest = _clean(following[0]) if following else ""
        if rest:
            # "Answer: X Answer: X" -> X
            rest, _ = _resolve_inline_marker(rest)
        return (rest or None), "answer_line"
    if len(non_empty) == 1:
        v = _clean(non_empty[0])
        v2, marker = _resolve_inline_marker(v)
        if marker:
            return (v2 or None), "inline_marker"
        return (v or None), "single_line"
    return None, "none"
