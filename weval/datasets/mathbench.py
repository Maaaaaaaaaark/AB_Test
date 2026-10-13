"""Original MATH adapter: full problem, official equivalence checker.

Source policy (see docs/UNRESOLVED.md):
  - the original author URL (people.eecs.berkeley.edu/~hendrycks/MATH.tar)
    returned HTTP 403 on 2026-10-13 -> not accessible;
  - the author's HF mirror hendrycks/competition_math was DISABLED on the Hub
    (verified 2026-10-13: API reports disabled=True, file endpoints 401) ->
    not accessible;
  - we use EleutherAI/hendrycks_math @ 21a5633873b6a120296cce3e2df9d5550074f4a3:
    per-category configs mirroring the original layout, counts verified at
    data-prep (train 7500 / test 5000) with the official fields
    {problem, solution, level, type}. MATH-500 or any other subset is never
    silently substituted. This mirror choice is documented in UNRESOLVED.md
    and awaits explicit user sign-off before the reference plan runs.

Answer extraction (\\boxed{}) and equivalence checking
(vendor/math/math_equivalence.py, verbatim from hendrycks/math @
985bdc1696e88e8643f081a0ff4719da39f2ae2a) are separate steps. Two missing
answers are NEVER treated as a correct match. The equivalence checker is a
heuristic, not a full symbolic solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MATH_DATASET = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
ADAPTER_VERSION = "math-adapter-v2"  # v1 pinned the disabled hendrycks/competition_math
MATH_EXPECTED_COUNTS = {"train": 7500, "test": 5000}
MATH_REQUIRED_FIELDS = {"problem", "solution", "level", "type"}


@dataclass
class MathItem:
    item_id: str
    group_id: str
    split: str
    task: str
    problem: str
    solution: str
    level: str
    type_: str
    gold: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "split": self.split,
            "task": self.task,
            "problem": self.problem,
            "solution": self.solution,
            "level": self.level,
            "type": self.type_,
            "gold": self.gold,
        }


def load_math(cache_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load pinned MATH mirror (per-category configs) and run authenticity checks."""
    from datasets import load_dataset

    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ("train", "test")}
    for config in MATH_CONFIGS:
        for split in ("train", "test"):
            ds = load_dataset(
                MATH_DATASET, config, split=split, revision=MATH_REVISION, cache_dir=cache_dir
            )
            missing = MATH_REQUIRED_FIELDS - set(ds.column_names)
            if missing:
                raise ValueError(
                    f"MATH/{config}/{split} missing fields {sorted(missing)}; "
                    f"available: {sorted(ds.column_names)}"
                )
            out[split].extend(dict(r) for r in ds)
    for split in ("train", "test"):
        if len(out[split]) != MATH_EXPECTED_COUNTS[split]:
            raise ValueError(
                f"MATH {split} has {len(out[split])} records across configs, "
                f"expected {MATH_EXPECTED_COUNTS[split]}. "
                "Counts alone do not prove authenticity, but a mismatch blocks usage."
            )
    return out


def extract_boxed_gold(solution: str) -> Optional[str]:
    """Gold answer extraction: last \\boxed{...} in the official solution.

    Returns None when no boxed answer exists (recorded, not silently skipped).
    """
    import re

    matches = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if not matches:
        return None
    return matches[-1].strip()


def build_items(records: List[Dict[str, Any]], split: str, start: int = 0, n: Optional[int] = None) -> List[MathItem]:
    items: List[MathItem] = []
    chosen = records[start : start + n] if n is not None else records[start:]
    for i, r in enumerate(chosen):
        boxed = extract_boxed_gold(r["solution"])
        items.append(
            MathItem(
                item_id=str(start + i),
                group_id=f"math:{start + i}",
                split=split,
                task="math",
                problem=r["problem"],
                solution=r["solution"],
                level=str(r["level"]),
                type_=str(r["type"]),
                gold={"boxed_answer": boxed, "solution": r["solution"], "level": str(r["level"]), "type": str(r["type"])},
            )
        )
    return items
