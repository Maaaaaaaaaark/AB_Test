"""Text normalization for cross-dataset problem matching.

The normalization intentionally PRESERVES digits and mathematical operators so
that "x+2=5" and "x + 2 = 5" match but "x+3=5" does not. Whitespace adjacent
to a math operator is removed; every other whitespace run collapses to a
single space, so alphanumeric tokens never merge ("x 2" stays distinct from
"x2" - no invented token boundaries). Used for exact-match grouping and
overlap exclusions only; it is not a deduplication claim and it does not
(and cannot) prove absence of pretraining contamination.
"""

from __future__ import annotations

import re

_KEEP = set("abcdefghijklmnopqrstuvwxyz0123456789+-*/=^()[]{}.,%<>:;_'")
_WS = re.compile(r"\s+")
# whitespace immediately adjacent to a math operator is insignificant
_OP_SPACE = re.compile(r"\s*([+\-*/=^()<>%])\s*")


def normalize_for_matching(text: str) -> str:
    """Lowercase, keep alphanumerics and math operators, collapse whitespace,
    and drop whitespace adjacent to math operators."""
    out = []
    for ch in text.lower():
        if ch in _KEEP:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(" ")
    s = _WS.sub(" ", "".join(out)).strip()
    return _OP_SPACE.sub(r"\1", s)


def token_set(text: str) -> set[str]:
    return set(normalize_for_matching(text).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


NEAR_DUP_JACCARD = 0.95  # flag threshold; review only, never auto-exclude
