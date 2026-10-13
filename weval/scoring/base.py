"""Shared score-record structure.

One ScoreRecord per (item, parsed answer). Aggregation (means, subsets, CIs)
happens in weval/metrics/; scorers here only judge single items and attach
the flags the metrics layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SCORE_RECORD_VERSION = "score-record-v1"


@dataclass
class ScoreRecord:
    item_id: str
    group_id: str
    task: str
    split: str
    scorer: str
    value: Optional[float]  # 0..1 (or None for unscorable; NaN is never written)
    status: str  # "scored" | "missing_final" | "gold_missing" | "unscorable"
    detail: Dict[str, Any] = field(default_factory=dict)
    version: str = SCORE_RECORD_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "task": self.task,
            "split": self.split,
            "scorer": self.scorer,
            "value": self.value,
            "status": self.status,
            "detail": self.detail,
            "version": self.version,
        }


def unscorable(item_id: str, group_id: str, task: str, split: str, scorer: str, reason: str, detail: Optional[Dict[str, Any]] = None) -> ScoreRecord:
    """An item that cannot be scored (e.g. gold answer missing). Excluded from
    aggregates with an explicit count - never silently dropped."""
    return ScoreRecord(
        item_id=item_id,
        group_id=group_id,
        task=task,
        split=split,
        scorer=scorer,
        value=None,
        status="unscorable",
        detail={"reason": reason, **(detail or {})},
    )
