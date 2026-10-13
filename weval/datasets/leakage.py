"""Cross-dataset problem overlap detection.

ProcessBench contains GSM8K and MATH test problems. We detect EXACT
normalized matches (normalization preserves digits and operators) and flag
near-duplicates for review (Jaccard >= 0.95 on normalized tokens).

This controls our own experimental leakage between local tasks and the
reference subsets. It does NOT prove absence of pretraining contamination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .normalization import NEAR_DUP_JACCARD, jaccard, normalize_for_matching

LEAKAGE_VERSION = "leakage-v1"


@dataclass
class LeakageReport:
    exact_matches: List[Dict[str, Any]] = field(default_factory=list)
    near_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    reference_exclusions: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": LEAKAGE_VERSION,
            "exact_matches": self.exact_matches,
            "near_duplicates": self.near_duplicates,
            "reference_exclusions": self.reference_exclusions,
            "counts": self.counts,
        }


def detect_overlap(
    processbench: Dict[str, List[Dict[str, Any]]],  # subset -> [{item_id, problem, ...}]
    gsm8k_records: Optional[List[Dict[str, Any]]],
    math_records: Optional[List[Dict[str, Any]]],
    local_splits_used: Tuple[str, ...] = ("debug", "selection"),
) -> LeakageReport:
    """Exact matches and near-duplicate flags between ProcessBench and the
    reference datasets. Reference exclusions are problems that appear in a
    local task split and therefore must not be presented as unseen references.
    """
    report = LeakageReport()
    targets: List[Tuple[str, List[Dict[str, Any]]]] = []
    if gsm8k_records is not None:
        targets.append(("gsm8k", gsm8k_records))
    if math_records is not None:
        targets.append(("math", math_records))

    for subset in ("gsm8k", "math"):
        for pb in processbench.get(subset, []):
            if pb.get("split") not in local_splits_used:
                continue
            pb_norm = normalize_for_matching(pb["problem"])
            for tname, trecs in targets:
                for t in trecs:
                    tq = t.get("question") or t.get("problem")
                    t_norm = normalize_for_matching(tq)
                    if pb_norm == t_norm:
                        report.exact_matches.append(
                            {
                                "processbench_item": pb["item_id"],
                                "processbench_subset": subset,
                                "processbench_split": pb.get("split"),
                                "reference_dataset": tname,
                                "reference_id": t.get("id") or t.get("item_id"),
                            }
                        )
                    else:
                        j = jaccard(pb_norm, t_norm)
                        if j >= NEAR_DUP_JACCARD:
                            report.near_duplicates.append(
                                {
                                    "processbench_item": pb["item_id"],
                                    "processbench_subset": subset,
                                    "reference_dataset": tname,
                                    "reference_id": t.get("id") or t.get("item_id"),
                                    "jaccard": round(j, 4),
                                    "note": "review required; not auto-excluded",
                                }
                            )

    report.counts = {
        "exact_matches": len(report.exact_matches),
        "near_duplicates": len(report.near_duplicates),
    }
    return report


def save_report(path: Path, report: LeakageReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
