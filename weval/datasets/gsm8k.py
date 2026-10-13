"""GSM8K (main split) adapter: full problem, official extraction and scoring.

Official data: HF openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
(config 'main', official test split).
Official extractor: openai/grade-school-math @
3101c7d5072418e28b9008a6636bde82a006892c (vendored). The official
ANS_RE/"#### <number>" extraction is the primary scorer; a relaxed parser
exists only as a secondary diagnostic and never replaces the primary score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
ADAPTER_VERSION = "gsm8k-adapter-v1"

# Official extractor, mirrored verbatim from the vendored file
# vendor/gsm8k/grade_school_math_dataset.py (ANS_RE / INVALID_ANS /
# extract_answer / is_correct). A consistency test compares this constant to
# the vendored source byte-for-byte.
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion: str) -> str:
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return INVALID_ANS


def is_correct(model_completion: str, gt_example: Dict[str, Any]) -> bool:
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS, "gold answer must be extractable"
    return extract_answer(model_completion) == gt_answer


@dataclass
class GSM8KItem:
    item_id: str
    group_id: str
    split: str
    task: str
    question: str
    answer: str  # full gold answer containing the #### marker
    gold: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "split": self.split,
            "task": self.task,
            "question": self.question,
            "answer": self.answer,
            "gold": self.gold,
        }


def load_gsm8k_main_test(cache_dir: Optional[str] = None):
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, "main", split="test", revision=GSM8K_REVISION, cache_dir=cache_dir)
    required = {"question", "answer"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"GSM8K missing fields {sorted(missing)}; available: {sorted(ds.column_names)}")
    return [dict(r) for r in ds]


def build_items(records: List[Dict[str, Any]], split: str, start: int = 0, n: Optional[int] = None) -> List[GSM8KItem]:
    items: List[GSM8KItem] = []
    chosen = records[start : start + n] if n is not None else records[start:]
    for i, r in enumerate(chosen):
        gt = extract_answer(r["answer"])
        assert gt != INVALID_ANS, f"gold answer not extractable for record {i}"
        items.append(
            GSM8KItem(
                item_id=str(start + i),
                group_id=f"gsm8k:{start + i}",
                split=split,
                task="gsm8k",
                question=r["question"],
                answer=r["answer"],
                gold={"extracted_answer": gt, "full_answer": r["answer"]},
            )
        )
    return items
