"""MATH scoring: vendored official equivalence checker.

Extraction (\\boxed{}) and equivalence are separate steps. Guards:
  * a missing predicted boxed answer is INCORRECT (never retried, never
    matched), and
  * a missing GOLD boxed answer makes the item unscorable (recorded with an
    explicit count, never silently dropped).
Two missing answers are NEVER treated as a match. The equivalence checker is
the official heuristic (vendored math_equivalence.py), not a full symbolic
solver - scores are not claimed to be more than that heuristic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from ..parsing.base import ParsedAnswer
from .base import ScoreRecord, unscorable

MATH_SCORER_VERSION = "math-equivalence-v1"

# Vendored official MATH equivalence checker (verbatim). The vendored file
# lives in vendor/math/, which collides with the stdlib builtin `math` - the
# builtin always wins parent-package resolution, so the file is loaded
# directly under a collision-free module name.
import importlib.util as _ilu

_VENDORED_MATH = Path(__file__).resolve().parents[2] / "vendor" / "math" / "math_equivalence.py"
_spec = _ilu.spec_from_file_location("vendored_math_equivalence", _VENDORED_MATH)
_vendored = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_vendored)
is_equiv = _vendored.is_equiv


def score_math(item: Dict[str, Any], parsed: ParsedAnswer) -> ScoreRecord:
    gold = item["gold"].get("boxed_answer")
    if gold is None or str(gold).strip() == "":
        return unscorable(
            item["item_id"], item["group_id"], "math", item["split"],
            "math_equivalence", "gold_answer_has_no_boxed",
        )
    pred = parsed.value
    if pred is None:
        return ScoreRecord(
            item_id=item["item_id"],
            group_id=item["group_id"],
            task="math",
            split=item["split"],
            scorer="math_equivalence",
            value=0.0,
            status="missing_final" if parsed.status == "missing_final" else "invalid_format",
            detail={
                "predicted_boxed": None,
                "gold_boxed": gold,
                "correct": False,
                "reason": "no_boxed_in_answer",
                "scorer_version": MATH_SCORER_VERSION,
            },
        )
    try:
        equiv = bool(is_equiv(str(pred), str(gold)))
    except Exception as e:  # noqa: BLE001 - heuristic crash must not kill the run
        return unscorable(
            item["item_id"], item["group_id"], "math", item["split"],
            "math_equivalence", f"equivalence_checker_crashed: {e}",
        )
    return ScoreRecord(
        item_id=item["item_id"],
        group_id=item["group_id"],
        task="math",
        split=item["split"],
        scorer="math_equivalence",
        value=1.0 if equiv else 0.0,
        status="scored",
        detail={
            "predicted_boxed": str(pred),
            "gold_boxed": str(gold),
            "correct": equiv,
            "scorer_version": MATH_SCORER_VERSION,
        },
    )
