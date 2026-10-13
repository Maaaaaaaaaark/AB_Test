"""ProcessBench adapter: official data, prompt, extraction and scoring.

Official repo: QwenLM/ProcessBench @ e8024636bcabdf8bd514440551b531d3f90dd18b
Official data: HF Qwen/ProcessBench @ 3bdcd5371ed567559a78f559c01c13a6deee7604
(splits gsm8k / math / olympiadbench / omnimath are the four subsets).

The task: find the index of the earliest incorrect step of the given solution,
or return -1 (the official no-error label). We do not add distractors and do
not alter the supplied solutions.

ProcessBench ships no train/test split that fits our needs, so we define a
PROJECT partition (explicitly labeled as such): groups of records that share
the same problem (all solution variants together), ordered deterministically,
split into debug / selection / holdout with disjoint problem groups per
subset. Quotas are met by whole groups; the error/no-error ratio is preserved
approximately and reported.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ConfigBundle
from .normalization import normalize_for_matching

PROCESSBENCH_DATASET = "Qwen/ProcessBench"
PROCESSBENCH_REVISION = "3bdcd5371ed567559a78f559c01c13a6deee7604"
PROCESSBENCH_SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]
ADAPTER_VERSION = "processbench-adapter-v1"

# Official prompt (vendored). Loaded from vendor/ at build time so the
# vendored copy is the single source of truth.
_VENDOR_TEMPLATE = Path(__file__).resolve().parents[2] / "vendor" / "processbench" / "critique_template.txt"


@dataclass
class ProcessBenchItem:
    item_id: str
    group_id: str
    subset: str
    split: str
    task: str
    problem: str
    steps: List[str]
    label: int
    final_answer_correct: Optional[bool]
    messages_provider: str = "processbench"
    gold: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "subset": self.subset,
            "split": self.split,
            "task": self.task,
            "problem": self.problem,
            "steps": self.steps,
            "label": self.label,
            "final_answer_correct": self.final_answer_correct,
            "gold": self.gold,
        }


def official_prompt_template() -> str:
    with open(_VENDOR_TEMPLATE, encoding="utf-8") as f:
        return f.read().strip()


def render_official_prompt(problem: str, steps: List[str]) -> str:
    tagged = ""
    for sdx, step in enumerate(steps):
        tagged += f"<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n"
    tagged = tagged.strip()
    return official_prompt_template().format(problem=problem, tagged_response=tagged)


def problem_group_key(problem: str) -> str:
    return normalize_for_matching(problem)


def load_processbench(subsets: Optional[List[str]] = None, cache_dir: Optional[str] = None):
    """Load the pinned ProcessBench data. Returns {subset: list of records}."""
    from datasets import load_dataset

    subsets = subsets or PROCESSBENCH_SUBSETS
    data: Dict[str, List[Dict[str, Any]]] = {}
    for subset in subsets:
        ds = load_dataset(
            PROCESSBENCH_DATASET,
            split=subset,
            revision=PROCESSBENCH_REVISION,
            cache_dir=cache_dir,
        )
        # Schema of the PINNED revision (verified by data-prep): every subset
        # carries exactly {id, generator, problem, steps, final_answer_correct,
        # label}. The README mentions final_answer/reference_answer columns;
        # they are NOT present at this revision - recorded, never guessed.
        required = {"problem", "steps", "label", "final_answer_correct"}
        missing = required - set(ds.column_names)
        if missing:
            raise ValueError(
                f"ProcessBench/{subset} is missing expected fields {sorted(missing)}; "
                f"available: {sorted(ds.column_names)}. Refusing to guess."
            )
        data[subset] = [dict(r) for r in ds]
    return data


def partition_records(
    records: List[Dict[str, Any]],
    quotas: Dict[str, int],
    order_seed: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Partition records by whole problem groups into debug/selection/holdout.

    Group order = the position of each group's first record in the official
    order, then a deterministic seeded shuffle (order_seed), so the same seed
    always reproduces the same partition and a different seed varies which
    groups land in which split. Quotas are met by taking whole groups; actual
    counts may exceed the quota slightly and are reported.
    """
    # group records by exact problem (preserving official order)
    seen: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in records:
        k = problem_group_key(r["problem"])
        if k not in seen:
            seen[k] = []
            order.append(k)
        seen[k].append(r)
    rng = random.Random(order_seed)
    rng.shuffle(order)

    parts: Dict[str, List[Dict[str, Any]]] = {"debug": [], "selection": [], "holdout": []}
    ratios: Dict[str, Any] = {"dataset": _error_ratio(records)}
    taken = 0
    for part, quota in [("debug", quotas["debug"]), ("selection", quotas["selection"]), ("holdout", quotas["holdout"])]:
        while taken < len(order) and len(parts[part]) < quota:
            for r in seen[order[taken]]:
                parts[part].append(r)
            taken += 1
        ratios[part] = _error_ratio(parts[part])
    leftover = [r for k in order[taken:] for r in seen[k]]
    ratios["leftover"] = _error_ratio(leftover)
    return parts, {"group_order": order, "n_groups_used": taken, "error_ratio": ratios, "order_seed": order_seed}


def _error_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    if not records:
        return None
    n_err = sum(1 for r in records if r["label"] != -1)
    return round(n_err / len(records), 4)


def build_items(records: List[Dict[str, Any]], subset: str, split: str) -> List[ProcessBenchItem]:
    items: List[ProcessBenchItem] = []
    for r in records:
        item_id = str(r.get("id", f"{subset}_{len(items)}"))
        fac = bool(r["final_answer_correct"]) if r.get("final_answer_correct") is not None else None
        items.append(
            ProcessBenchItem(
                item_id=item_id,
                group_id=f"{subset}:{problem_group_key(r['problem'])}",
                subset=subset,
                split=split,
                task="processbench",
                problem=r["problem"],
                steps=list(r["steps"]),
                label=int(r["label"]),
                final_answer_correct=fac,
                gold={
                    "label": int(r["label"]),
                    "final_answer_correct": fac,
                    "subset": subset,
                },
            )
        )
    return items


def processbench_quotas(cfg: ConfigBundle) -> Dict[str, int]:
    stages = cfg.splits["stages"]
    return {
        "debug": stages["pilot"]["processbench_per_subset"],
        "selection": stages["selection"]["processbench_per_subset"],
        "holdout": stages["holdout"]["processbench_per_subset"],
    }
