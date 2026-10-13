"""HotpotQA distractor adapter: full question and all ten context paragraphs.

Official data: HF hotpotqa/hotpot_qa @ 1908d6afbbead072334abe2965f91bd2709910ab
(config 'distractor', official dev split).
Official scoring: hotpotqa/hotpot @
3635853403a8735609ee997664e1528f4480762a (vendored). We use the official
answer normalization/EM/F1 only; the supporting-facts and joint metrics are
not computed because this project does not predict supporting facts
(adapted score, documented in docs/PROTOCOL.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HOTPOTQA_DATASET = "hotpotqa/hotpot_qa"
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
ADAPTER_VERSION = "hotpotqa-adapter-v1"


@dataclass
class HotpotQAItem:
    item_id: str
    group_id: str
    split: str
    task: str
    question: str
    answer: str
    paragraphs: List[Dict[str, str]]  # [{title, text}] in context order
    gold: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "split": self.split,
            "task": self.task,
            "question": self.question,
            "answer": self.answer,
            "paragraphs": self.paragraphs,
            "gold": self.gold,
        }


def _context_paragraphs(context: Any) -> List[Dict[str, str]]:
    """HF distractor context: list of [title, [sentence, ...]] pairs, in order."""
    paras: List[Dict[str, str]] = []
    for pair in context:
        if isinstance(pair, dict):
            # alternate representation: {"title": t, "sentences": [...]}
            paras.append({"title": pair["title"], "text": " ".join(pair["sentences"])})
        else:
            title, sentences = pair[0], pair[1]
            paras.append({"title": title, "text": " ".join(sentences)})
    return paras


def load_hotpotqa_distractor_dev(cache_dir: Optional[str] = None):
    from datasets import load_dataset

    ds = load_dataset(
        HOTPOTQA_DATASET, "distractor", split="validation",
        revision=HOTPOTQA_REVISION, cache_dir=cache_dir,
    )
    required = {"id", "question", "answer", "context"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"HotpotQA missing fields {sorted(missing)}; available: {sorted(ds.column_names)}")
    return [dict(r) for r in ds]


def build_items(records: List[Dict[str, Any]], split: str, start: int = 0, n: Optional[int] = None) -> List[HotpotQAItem]:
    items: List[HotpotQAItem] = []
    chosen = records[start : start + n] if n is not None else records[start:]
    for i, r in enumerate(chosen):
        items.append(
            HotpotQAItem(
                item_id=str(r["id"]),
                group_id=f"hotpotqa:{r['id']}",
                split=split,
                task="hotpotqa",
                question=r["question"],
                answer=r["answer"],
                paragraphs=_context_paragraphs(r["context"]),
                gold={"answer": r["answer"], "supporting_facts": list(r.get("supporting_facts", []))},
            )
        )
    return items
