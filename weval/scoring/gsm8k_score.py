"""GSM8K scoring: official extracted-answer equality.

is_correct = (official extraction of the completion == official extraction of
the gold answer). The gold extraction is asserted at build time, so a gold
INVALID_ANS cannot occur here. Extraction failure on the model side is the
official INVALID_ANS semantics: scored incorrect, counted in the reported
extraction-failure rate.
"""

from __future__ import annotations

from typing import Any, Dict

from ..datasets.gsm8k import INVALID_ANS
from ..parsing.base import ParsedAnswer
from .base import ScoreRecord

GSM8K_SCORER_VERSION = "gsm8k-official-v1"


def score_gsm8k(item: Dict[str, Any], parsed: ParsedAnswer) -> ScoreRecord:
    gold = item["gold"]["extracted_answer"]
    pred = parsed.value
    correct = pred is not None and pred == gold
    return ScoreRecord(
        item_id=item["item_id"],
        group_id=item["group_id"],
        task="gsm8k",
        split=item["split"],
        scorer="gsm8k_official",
        value=1.0 if correct else 0.0,
        status=parsed.status if parsed.status != "ok" else "scored",
        detail={
            "extracted": pred,
            "extraction_failed": parsed.status == "invalid_format",
            "gold_extracted": gold,
            "relaxed_extracted": parsed.diagnostics.get("relaxed_extracted"),
            "relaxed_matches_gold": parsed.diagnostics.get("relaxed_extracted") == gold,
            "correct": bool(correct),
            "scorer_version": GSM8K_SCORER_VERSION,
        },
    )
