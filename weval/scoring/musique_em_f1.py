"""MuSiQue scoring: vendored official EM/F1 + abstention flags.

Positives: predicted answer scored against gold answers (max over golds) with
the vendored official normalize_answer/compute_exact/compute_f1. A NOT FOUND
on a positive is a FALSE abstention: value 0, flag set.

Negatives: the only correct behavior is abstention (NOT FOUND). An answer
attempt on a negative is wrong regardless of content: value 0. A NOT FOUND on
a negative is a correct abstention: value 1.

The flags produced here feed the selection thresholds:
  * answerable F1       - mean F1 over positives where the model answered,
  * false-abstention    - fraction of positives where the model abstained,
  * reviewed-negative correct-abstention - fraction of approved negatives
    where the model abstained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from ..parsing.base import ParsedAnswer
from .base import ScoreRecord, unscorable

MUSIQUE_SCORER_VERSION = "musique-em-f1-v2"

# Vendored official MuSiQue answer metric (verbatim). The vendored files
# import each other with absolute paths (metrics.metric), so BOTH the vendor
# root and vendor/musique must be on sys.path.
_VENDOR = Path(__file__).resolve().parents[2] / "vendor"
for _p in (str(_VENDOR / "musique"), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from musique.metrics.answer import compute_exact, compute_f1  # noqa: E402


def score_musique(item: Dict[str, Any], parsed: ParsedAnswer) -> ScoreRecord:
    golds = item.get("reference_answers") or item.get("gold", {}).get("reference_answers") or []
    kind = item.get("kind", "positive")
    is_not_found = bool(parsed.diagnostics.get("is_not_found"))

    if kind == "negative":
        if is_not_found:
            return ScoreRecord(
                item_id=item["item_id"],
                group_id=item["group_id"],
                task="musique_local",
                split=item["split"],
                scorer="musique_em_f1",
                value=1.0,
                status="scored",
                detail={
                    "kind": "negative",
                    "abstained": True,
                    "correct_abstention": True,
                    "false_answer": False,
                    "predicted_answer": parsed.value,
                    "scorer_version": MUSIQUE_SCORER_VERSION,
                },
            )
        return ScoreRecord(
            item_id=item["item_id"],
            group_id=item["group_id"],
            task="musique_local",
            split=item["split"],
            scorer="musique_em_f1",
            value=0.0,
            status=parsed.status if parsed.status != "ok" else "scored",
            detail={
                "kind": "negative",
                "abstained": False,
                "correct_abstention": False,
                "false_answer": True,
                "predicted_answer": parsed.value,
                "scorer_version": MUSIQUE_SCORER_VERSION,
            },
        )

    # positive item
    if not golds:
        return unscorable(item["item_id"], item["group_id"], "musique_local", item["split"], "musique_em_f1", "no_gold_answers")
    if is_not_found:
        return ScoreRecord(
            item_id=item["item_id"],
            group_id=item["group_id"],
            task="musique_local",
            split=item["split"],
            scorer="musique_em_f1",
            value=0.0,
            status="scored",
            detail={
                "kind": "positive",
                "abstained": True,
                "false_abstention": True,
                # ``answerable`` describes the gold item, not the model's
                # decision to abstain.  Keeping this True ensures that a
                # false abstention contributes zero to answerable F1 instead
                # of disappearing from its denominator.
                "answerable": True,
                "em": 0,
                "f1": 0.0,
                "predicted_answer": parsed.value,
                "scorer_version": MUSIQUE_SCORER_VERSION,
            },
        )
    if parsed.status == "missing_final":
        return ScoreRecord(
            item_id=item["item_id"],
            group_id=item["group_id"],
            task="musique_local",
            split=item["split"],
            scorer="musique_em_f1",
            value=0.0,
            status="missing_final",
            detail={
                "kind": "positive",
                "abstained": False,
                "false_abstention": False,
                "answerable": True,
                "em": 0,
                "f1": 0.0,
                "predicted_answer": None,
                "scorer_version": MUSIQUE_SCORER_VERSION,
            },
        )
    pred = str(parsed.value or "")
    em = max(compute_exact(pred, g) for g in golds)
    f1 = max(compute_f1(pred, g) for g in golds)
    return ScoreRecord(
        item_id=item["item_id"],
        group_id=item["group_id"],
        task="musique_local",
        split=item["split"],
        scorer="musique_em_f1",
        value=f1,
        status="scored",
        detail={
            "kind": "positive",
            "abstained": False,
            "false_abstention": False,
            "answerable": True,
            "em": int(em),
            "f1": f1,
            "is_correct": bool(em),
            "predicted_answer": pred,
            "scorer_version": MUSIQUE_SCORER_VERSION,
        },
    )


def score_musique_batch(items: List[Dict[str, Any]], parsed_by_id: Dict[str, ParsedAnswer]) -> List[ScoreRecord]:
    return [score_musique(it, parsed_by_id[it["item_id"]]) for it in items if it["item_id"] in parsed_by_id]
