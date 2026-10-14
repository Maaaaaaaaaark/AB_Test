"""HotpotQA scoring: vendored official answer normalization/EM/F1.

Official scoring also covers supporting facts and joint EM/F1; this project
does not predict supporting facts, so only the answer metrics are computed
(adapted scope, documented in docs/PROTOCOL.md). The vendored functions are
used verbatim for the parts that apply.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from ..parsing.base import ParsedAnswer
from .base import ScoreRecord

HOTPOTQA_SCORER_VERSION = "hotpotqa-answer-f1-v2"

# Vendored official HotpotQA evaluation (verbatim).
_VENDOR = Path(__file__).resolve().parents[2] / "vendor"
for _p in (str(_VENDOR / "hotpotqa"), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hotpotqa.hotpot_evaluate_v1 import exact_match_score, f1_score  # noqa: E402


def score_hotpotqa(item: Dict[str, Any], parsed: ParsedAnswer) -> ScoreRecord:
    gold = item["gold"]["answer"]
    pred = parsed.value or ""
    em = exact_match_score(pred, gold)
    f1, _, _ = f1_score(pred, gold)
    return ScoreRecord(
        item_id=item["item_id"],
        group_id=item["group_id"],
        task="hotpotqa",
        split=item["split"],
        scorer="hotpotqa_answer_f1",
        value=f1,
        status=parsed.status if parsed.status != "ok" else "scored",
        detail={
            "em": int(em),
            "f1": f1,
            "predicted_answer": pred,
            "gold_answer": gold,
            "answer_source": parsed.diagnostics.get("answer_source"),
            "scorer_version": HOTPOTQA_SCORER_VERSION,
        },
    )
