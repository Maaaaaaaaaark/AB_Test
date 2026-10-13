"""ProcessBench scoring: official error acc / correct acc / F1.

Official (vendored run_eval.py): pred == label (int), computed separately over
error solutions (label != -1) and correct solutions (label == -1);
subset F1 = 2*a*b/(a+b). The macro average preserves the subset structure
(mean of subset F1s) - computed in weval/metrics/stats.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parsing.base import ParsedAnswer
from .base import ScoreRecord

PROCESSBENCH_SCORER_VERSION = "processbench-official-v1"


def score_processbench(item: Dict[str, Any], parsed: ParsedAnswer) -> ScoreRecord:
    gold_label = int(item["gold"]["label"])
    pred: Optional[int] = parsed.value
    match = pred is not None and pred == gold_label
    kind = "correct_solution" if gold_label == -1 else "error_solution"
    return ScoreRecord(
        item_id=item["item_id"],
        group_id=item["group_id"],
        task="processbench",
        split=item["split"],
        scorer="processbench_official",
        value=1.0 if match else 0.0,
        status=parsed.status if parsed.status != "ok" else "scored",
        detail={
            "subset": item.get("subset"),
            "predicted_index": pred,
            "gold_label": gold_label,
            "kind": kind,
            "match": bool(match),
            "scorer_version": PROCESSBENCH_SCORER_VERSION,
        },
    )


def subset_error_correct_f1(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Error acc, correct acc, F1 over one subset's records (official formula).

    Accepts persisted record dicts (the form the report layer works with).
    Missing kinds stay None (never fabricated as 0.0); the report layer
    handles them explicitly.
    """
    err = [r for r in records if (r.get("detail") or {}).get("kind") == "error_solution"]
    cor = [r for r in records if (r.get("detail") or {}).get("kind") == "correct_solution"]
    a = sum(r["value"] for r in err) / len(err) if err else None
    b = sum(r["value"] for r in cor) / len(cor) if cor else None
    f1 = None
    if a is not None and b is not None:
        f1 = 2 * a * b / (a + b) if (a + b) > 0 else 0.0
    return {
        "error_acc": a,
        "correct_acc": b,
        "f1": f1,
        "n_error": len(err),
        "n_correct": len(cor),
    }
