"""MuSiQue-Ans local evidence task builder.

Builds a custom LOCAL task from the official MuSiQue-Ans question
decomposition (pinned release, authors' distribution):

  * one subquestion per parent question (seeded, avoiding the first hop when
    possible),
  * dependency references (#1, #2, ...) resolved deterministically with the
    canonical prerequisite subanswers (never an LLM rewrite),
  * five visible paragraphs: the annotated supporting paragraph + four
    paragraphs that are not supporting for ANY subquestion of the parent
    (strict rule: other supporting paragraphs are never used as distractors
    in this controlled test),
  * the current answer, later answers, supporting labels and the full
    solution are never revealed.

Gold answers: canonical subanswer only for intermediate steps; parent-question
answer + aliases for the terminal step, and only when the canonical subanswer
matches the parent answer under the official normalization (alias source is
recorded per item).

Paired negatives replace the supporting paragraph with a fifth non-supporting
paragraph at the same position. String screening (answer/alias presence) is a
screening check only; ambiguous yes/no and numeric negatives are excluded by
policy unless explicitly reviewed. Every pair lands in a review file with
human review status; only approved pairs enter the final abstention
evaluation. Eligibility is frozen before any model output is seen.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ConfigBundle, canonical_json, fingerprint
from ..provenance import DatasetProvenance
from .normalization import normalize_for_matching

# Import the vendored official MuSiQue answer metric (verbatim). The vendored
# files import each other with absolute paths (metrics.metric), so BOTH the
# vendor root and vendor/musique must be on sys.path.
_VENDOR = Path(__file__).resolve().parents[2] / "vendor"
for _p in (str(_VENDOR / "musique"), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from musique.metrics.answer import normalize_answer  # noqa: E402

BUILDER_VERSION = "musique-local-builder-v1"

REF_RE = re.compile(r"#(\d+)\b")
RELATION_MARK = ">>"
NUMERIC_RE = re.compile(r"^-?\d[\d,.]*$")
YES_NO = {"yes", "no"}

MUSIQUE_ANS_RELEASE = "musique_ans_v1.0"  # authors' distribution, zip SHA recorded at data-prep


class UnresolvableReference(Exception):
    pass


@dataclass
class VisibleParagraph:
    slot: int  # 1..5, position in the visible list (original idx order)
    source_idx: int  # original paragraph idx in the parent record
    title: str
    text: str
    role: str  # "support" | "distractor"


@dataclass
class MusiqueItem:
    item_id: str
    group_id: str  # parent id
    split: str
    task: str
    kind: str  # "positive" | "negative"
    pair_id: Optional[str]
    parent_id: str
    subq_index: int
    n_hops: int
    is_terminal: bool
    question: str  # with references resolved inline
    reference_answers: List[str]
    alias_source: str
    expected: str  # "answer" for positives, "not_found" for negatives
    visible: List[VisibleParagraph]
    syntax: Dict[str, bool]
    support_literal_screening: bool
    review_status: str  # positives: "na"; negatives: proposed/approved/excluded
    human_reviewed: bool
    excluded_reason: Optional[str]
    messages_provider: str = "musique_local"
    gold: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "item_id": self.item_id,
            "group_id": self.group_id,
            "split": self.split,
            "task": self.task,
            "kind": self.kind,
            "pair_id": self.pair_id,
            "parent_id": self.parent_id,
            "subq_index": self.subq_index,
            "n_hops": self.n_hops,
            "is_terminal": self.is_terminal,
            "question": self.question,
            "reference_answers": self.reference_answers,
            "alias_source": self.alias_source,
            "expected": self.expected,
            "visible": [
                {"slot": p.slot, "source_idx": p.source_idx, "title": p.title, "text": p.text, "role": p.role}
                for p in self.visible
            ],
            "syntax": self.syntax,
            "support_literal_screening": self.support_literal_screening,
            "review_status": self.review_status,
            "human_reviewed": self.human_reviewed,
            "excluded_reason": self.excluded_reason,
            "gold": self.gold,
        }
        return d


def _subq_answer(record: Dict[str, Any], i: int) -> str:
    return record["question_decomposition"][i]["answer"]


def resolve_references(
    question: str, decomposition: List[Dict[str, Any]], subq_index: int
) -> Tuple[str, List[Tuple[int, str]]]:
    """Deterministic, non-LLM resolution of #N references.

    #N refers to subquestion N (1-based). Only references to EARLIER
    subquestions are resolvable; anything else raises UnresolvableReference.
    """
    used: List[Tuple[int, str]] = []

    def repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if n < 1 or n - 1 >= subq_index:
            raise UnresolvableReference(
                f"subq {subq_index} references #{n} which is not an earlier subquestion"
            )
        ans = decomposition[n - 1]["answer"]
        used.append((n, ans))
        return ans

    resolved = REF_RE.sub(repl, question)
    return resolved, used


def non_supporting_anywhere(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Paragraphs that are not supporting for the parent NOR for any subquestion.

    Strict rule for this controlled test: other supporting paragraphs of the
    original example are never used as distractors.
    """
    support_idxs = {
        d["paragraph_support_idx"]
        for d in record["question_decomposition"]
        if d.get("paragraph_support_idx", -1) is not None and d.get("paragraph_support_idx", -1) >= 0
    }
    return [
        p
        for p in record["paragraphs"]
        if not p["is_supporting"] and p["idx"] not in support_idxs
    ]


def eligible_subquestions(record: Dict[str, Any]) -> List[int]:
    decomp = record["question_decomposition"]
    paras = {p["idx"]: p for p in record["paragraphs"]}
    pool = non_supporting_anywhere(record)
    eligible: List[int] = []
    for i, d in enumerate(decomp):
        supp_idx = d.get("paragraph_support_idx", -1)
        if supp_idx is None or supp_idx < 0 or supp_idx not in paras:
            continue  # subquestion without usable annotated support
        if len(pool) < 4:
            continue  # cannot build 5 visible paragraphs under the strict rule
        try:
            resolve_references(d["question"], decomp, i)
        except UnresolvableReference:
            continue
        eligible.append(i)
    return eligible


def _gold_for_subquestion(record: Dict[str, Any], subq_index: int) -> Tuple[List[str], str]:
    """Gold answers + alias source.

    Intermediate steps: canonical subanswer only (no invented aliases).
    Terminal step: parent answer + parent aliases, but ONLY when the canonical
    subanswer matches the parent answer under the official normalization.
    """
    decomp = record["question_decomposition"]
    canon = decomp[subq_index]["answer"]
    is_terminal = subq_index == len(decomp) - 1
    if is_terminal and normalize_answer(canon) == normalize_answer(record["answer"]):
        gold = [record["answer"]] + list(record.get("answer_aliases", []))
        source = "parent_answer_aliases"
    else:
        gold = [canon]
        source = "subquestion_canonical_only"
        if is_terminal:
            source = "subquestion_canonical_only_terminal_mismatch"  # anomaly to review
    return gold, source


def _literal_in_text(gold_answers: List[str], text: str) -> bool:
    norm_text = normalize_for_matching(text)
    return any(normalize_for_matching(a) in norm_text for a in gold_answers if a.strip())


def _negative_exclusion(
    gold_answers: List[str], paragraphs: List[VisibleParagraph], resolved_question: str
) -> Optional[str]:
    """Screening + policy checks for negative pairs. None = candidate passes."""
    for p in paragraphs:
        if _literal_in_text(gold_answers, p.title):
            return "answer_visible_in_title"
        if _literal_in_text(gold_answers, p.text):
            return "answer_visible_in_paragraph"
    if _literal_in_text(gold_answers, resolved_question):
        return "answer_visible_in_prerequisites"
    norm_golds = [normalize_for_matching(a) for a in gold_answers]
    if all(g in YES_NO for g in norm_golds):
        return "ambiguous_yes_no"
    if all(NUMERIC_RE.match(g) for g in norm_golds):
        return "numeric_answer_policy"
    return None


def _syntax_audit(question: str) -> Dict[str, bool]:
    return {
        "has_ref": bool(REF_RE.search(question)),
        "has_relation": RELATION_MARK in question,
    }


def build_positive(
    record: Dict[str, Any], subq_index: int, split: str, syntax: Optional[Dict[str, bool]] = None
) -> MusiqueItem:
    decomp = record["question_decomposition"]
    d = decomp[subq_index]
    question, used = resolve_references(d["question"], decomp, subq_index)
    gold, alias_source = _gold_for_subquestion(record, subq_index)
    pool = non_supporting_anywhere(record)
    pool_by_idx = {p["idx"]: p for p in pool}
    supp_idx = d["paragraph_support_idx"]
    supp = next(p for p in record["paragraphs"] if p["idx"] == supp_idx)
    distractor_idxs = sorted(p["idx"] for p in pool)[:4]
    chosen = [supp] + [pool_by_idx[i] for i in distractor_idxs]
    chosen_sorted = sorted(chosen, key=lambda p: p["idx"])
    visible = [
        VisibleParagraph(
            slot=pos + 1,
            source_idx=p["idx"],
            title=p["title"],
            text=p["paragraph_text"],
            role="support" if p["idx"] == supp_idx else "distractor",
        )
        for pos, p in enumerate(chosen_sorted)
    ]
    item = MusiqueItem(
        item_id=f"{record['id']}:s{subq_index}",
        group_id=record["id"],
        split=split,
        task="musique_local",
        kind="positive",
        pair_id=f"{record['id']}:s{subq_index}:pair",
        parent_id=record["id"],
        subq_index=subq_index,
        n_hops=len(decomp),
        is_terminal=subq_index == len(decomp) - 1,
        question=question,
        reference_answers=gold,
        alias_source=alias_source,
        expected="answer",
        visible=visible,
        syntax=syntax or _syntax_audit(d["question"]),
        support_literal_screening=_literal_in_text(gold, supp["paragraph_text"]),
        review_status="na",
        human_reviewed=False,
        excluded_reason=None,
        gold={
            "reference_answers": gold,
            "alias_source": alias_source,
            "used_prerequisites": used,
            "subquestion_answer": d["answer"],
            "parent_answer": record["answer"],
            "parent_answer_aliases": list(record.get("answer_aliases", [])),
            "support_paragraph_idx": supp_idx,
        },
    )
    return item


def build_pair(
    record: Dict[str, Any],
    subq_index: int,
    positive: MusiqueItem,
    split: str,
) -> Tuple[Optional[MusiqueItem], Optional[str]]:
    """Paired negative. Returns (item or None, exclusion_reason or None)."""
    decomp = record["question_decomposition"]
    d = decomp[subq_index]
    question, used = resolve_references(d["question"], decomp, subq_index)
    gold, _ = _gold_for_subquestion(record, subq_index)
    pool = non_supporting_anywhere(record)
    pool_by_idx = {p["idx"]: p for p in pool}
    if len(pool) < 5:
        return None, "insufficient_non_supporting_paragraphs"
    supp_idx = d["paragraph_support_idx"]
    fifth = pool_by_idx[sorted(p["idx"] for p in pool)[4]]
    visible = []
    for p in positive.visible:
        if p.role == "support":
            visible.append(
                VisibleParagraph(
                    slot=p.slot,
                    source_idx=fifth["idx"],
                    title=fifth["title"],
                    text=fifth["paragraph_text"],
                    role="distractor",
                )
            )
        else:
            visible.append(p)
    reason = _negative_exclusion(gold, visible, question)
    if reason is not None:
        return None, reason
    item = MusiqueItem(
        item_id=f"{record['id']}:s{subq_index}:neg",
        group_id=record["id"],
        split=split,
        task="musique_local",
        kind="negative",
        pair_id=f"{record['id']}:s{subq_index}:pair",
        parent_id=record["id"],
        subq_index=subq_index,
        n_hops=len(decomp),
        is_terminal=subq_index == len(decomp) - 1,
        question=question,
        reference_answers=gold,
        alias_source=positive.alias_source,
        expected="not_found",
        visible=visible,
        syntax=positive.syntax,
        support_literal_screening=True,  # not applicable to negatives
        review_status="proposed",
        human_reviewed=False,
        excluded_reason=None,
        gold={
            "reference_answers": gold,
            "alias_source": positive.alias_source,
            "used_prerequisites": used,
            "subquestion_answer": d["answer"],
            "parent_answer": record["answer"],
            "parent_answer_aliases": list(record.get("answer_aliases", [])),
            "support_paragraph_idx": supp_idx,
            "replaced_with_paragraph_idx": fifth["idx"],
        },
    )
    return item, None


@dataclass
class MusiqueBuildReport:
    version: str
    n_parents_total: int
    n_parents_eligible: int
    n_positives_built: int = 0
    n_pair_candidates: int = 0
    n_pairs_approved: int = 0
    exclusions: Dict[str, int] = field(default_factory=dict)
    syntax_distribution: Dict[str, int] = field(default_factory=dict)
    hop_distribution: Dict[str, int] = field(default_factory=dict)
    parent_eligibility_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "n_parents_total": self.n_parents_total,
            "n_parents_eligible": self.n_parents_eligible,
            "n_positives_built": self.n_positives_built,
            "n_pair_candidates": self.n_pair_candidates,
            "n_pairs_approved": self.n_pairs_approved,
            "exclusions": self.exclusions,
            "syntax_distribution": self.syntax_distribution,
            "hop_distribution": self.hop_distribution,
            "parent_eligibility_reasons": self.parent_eligibility_reasons,
        }


def _parent_eligibility_reason(record: Dict[str, Any]) -> str:
    decomp = record["question_decomposition"]
    pool = non_supporting_anywhere(record)
    if len(pool) < 4:
        return "insufficient_non_supporting_paragraphs"
    paras = {p["idx"]: p for p in record["paragraphs"]}
    if not any(
        d.get("paragraph_support_idx", -1) is not None and d.get("paragraph_support_idx", -1) >= 0
        and d.get("paragraph_support_idx") in paras
        for d in decomp
    ):
        return "no_supported_subquestion"
    if not any(_refs_resolvable(d, decomp, i) for i, d in enumerate(decomp)):
        return "no_resolvable_subquestion"
    return "unknown"


def _refs_resolvable(d: Dict[str, Any], decomp: List[Dict[str, Any]], i: int) -> bool:
    try:
        resolve_references(d["question"], decomp, i)
        return True
    except UnresolvableReference:
        return False


def _item_seed(run_seed: int, item_id: str) -> int:
    import hashlib

    h = hashlib.sha256(item_id.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "little")
    return (run_seed * 2654435761 + x) % (2**32)


def choose_subquestion(record: Dict[str, Any], split_seed: int) -> Tuple[int, List[int]]:
    eligible = eligible_subquestions(record)
    if not eligible:
        return -1, eligible
    candidates = eligible[1:] if len(eligible) > 1 else eligible
    rng = random.Random(_item_seed(split_seed, record["id"]))
    return rng.choice(candidates), eligible


def build_musique(
    records: List[Dict[str, Any]],
    split: str,
    split_seed: int,
    parent_ids: Optional[List[str]] = None,
    require_approved: bool = False,
    approved_pairs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[MusiqueItem], MusiqueBuildReport, Dict[str, str]]:
    """Build items for a split from given parents (deterministic order).

    Returns (items, report, pair_exclusions) where pair_exclusions maps
    parent id -> exclusion reason for parents whose pair could not be built.
    """
    approved_pairs = approved_pairs or {}
    selected: List[MusiqueItem] = []
    pair_exclusions: Dict[str, str] = {}
    report = MusiqueBuildReport(version=BUILDER_VERSION, n_parents_total=len(records), n_parents_eligible=0)
    order = parent_ids if parent_ids is not None else sorted(r["id"] for r in records)
    by_id = {r["id"]: r for r in records}
    for pid in order:
        if pid not in by_id:
            continue
        record = by_id[pid]
        subq, eligible = choose_subquestion(record, split_seed)
        if subq < 0:
            report.parent_eligibility_reasons[_parent_eligibility_reason(record)] = (
                report.parent_eligibility_reasons.get(_parent_eligibility_reason(record), 0) + 1
            )
            continue
        report.n_parents_eligible += 1
        pos = build_positive(record, subq, split)
        selected.append(pos)
        report.n_positives_built += 1
        report.syntax_distribution[
            "relation" if pos.syntax["has_relation"] else "natural"
        ] = report.syntax_distribution.get(
            "relation" if pos.syntax["has_relation"] else "natural", 0
        ) + 1
        report.hop_distribution[
            f"hop_{pos.subq_index}_of_{pos.n_hops - 1}"
        ] = report.hop_distribution.get(f"hop_{pos.subq_index}_of_{pos.n_hops - 1}", 0) + 1
        neg, reason = build_pair(record, subq, pos, split)
        if reason is not None:
            report.exclusions[reason] = report.exclusions.get(reason, 0) + 1
            pair_exclusions[pid] = reason
            continue
        report.n_pair_candidates += 1
        entry = approved_pairs.get(neg.item_id)
        if entry is not None and entry.get("review_status") == "approved" and entry.get("human_reviewed"):
            neg.review_status = "approved"
            neg.human_reviewed = True
            report.n_pairs_approved += 1
            selected.append(neg)
        elif entry is not None and entry.get("review_status") == "excluded":
            reason2 = entry.get("excluded_reason") or "review_excluded"
            report.exclusions[f"review:{reason2}"] = report.exclusions.get(f"review:{reason2}", 0) + 1
            pair_exclusions[pid] = f"review:{reason2}"
        elif not require_approved:
            # development mode (smoke): unreviewed candidates are included but
            # clearly labeled; they cannot support qualification claims.
            selected.append(neg)
    return selected, report, pair_exclusions


def review_entry(
    record: Dict[str, Any], subq_index: int, positive: MusiqueItem,
    negative: Optional[MusiqueItem], exclusion_reason: Optional[str],
) -> Dict[str, Any]:
    """One row of the negative review file (provenance + checks + status)."""
    checks = {
        "positive_support_literal_screening": positive.support_literal_screening,
    }
    if negative is not None:
        checks.update(_negative_checks(positive.reference_answers, negative))
    entry = {
        "parent_id": record["id"],
        "subq_index": subq_index,
        "positive_item_id": positive.item_id,
        "negative_item_id": negative.item_id if negative is not None else None,
        "pair_id": f"{record['id']}:s{subq_index}:pair",
        "replaced_support_paragraph_idx": record["question_decomposition"][subq_index][
            "paragraph_support_idx"
        ],
        "replacement_paragraph_idx": (
            negative.gold.get("replaced_with_paragraph_idx") if negative is not None else None
        ),
        "question_resolved": positive.question,
        "reference_answers": positive.reference_answers,
        "alias_source": positive.alias_source,
        "syntax": positive.syntax,
        "is_terminal": positive.is_terminal,
        "checks": checks,
        "proposed_label": "unanswerable" if negative is not None else None,
        "excluded_reason": exclusion_reason,
        "review_status": "proposed" if negative is not None else "excluded",
        "human_reviewed": False,
        "reviewed_by": None,
        "review_note": None,
        "provenance": {
            "builder_version": BUILDER_VERSION,
            "source": MUSIQUE_ANS_RELEASE,
            "parent_record_id": record["id"],
        },
    }
    return entry


def _negative_checks(gold_answers: List[str], negative: MusiqueItem) -> Dict[str, bool]:
    return {
        "negative_answer_visible_in_title": any(
            _literal_in_text(gold_answers, p.title) for p in negative.visible
        ),
        "negative_answer_visible_in_paragraph": any(
            _literal_in_text(gold_answers, p.text) for p in negative.visible
        ),
        "negative_answer_visible_in_prerequisites": _literal_in_text(
            gold_answers, negative.question
        ),
    }


def write_review_file(
    path: Path,
    parents: List[Dict[str, Any]],
    positives: Dict[str, MusiqueItem],   # by parent id
    negatives: Dict[str, Optional[MusiqueItem]],  # by parent id; None if excluded
    exclusion_reasons: Dict[str, str],   # by parent id
) -> int:
    """Persist the negative-pair review file. Returns number of pair candidates.

    Human decisions survive rebuilds: for parents whose existing entry is
    human-reviewed (human_reviewed=true), the decision fields (review_status,
    human_reviewed, reviewed_by, review_note, excluded_reason) are carried
    over; everything else (checks, provenance) is regenerated from the fresh
    build. Build-derived entries (human_reviewed=false) are regenerated.
    """
    existing: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        try:
            existing = {e["parent_id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}
        except Exception:  # noqa: BLE001 - a corrupt review file must not kill the build
            existing = {}
    entries: List[Dict[str, Any]] = []
    n_candidates = 0
    for record in parents:
        pid = record["id"]
        pos = positives.get(pid)
        if pos is None:
            continue
        neg = negatives.get(pid)
        entry = review_entry(record, pos.subq_index, pos, neg, exclusion_reasons.get(pid))
        old = existing.get(pid)
        if old is not None and old.get("human_reviewed"):
            entry["review_status"] = old.get("review_status", entry["review_status"])
            entry["human_reviewed"] = True
            entry["reviewed_by"] = old.get("reviewed_by")
            entry["review_note"] = old.get("review_note")
            entry["excluded_reason"] = old.get("excluded_reason")
        entries.append(entry)
        if neg is not None:
            n_candidates += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    return n_candidates


def load_musique_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_review_map(review_path: Path) -> Dict[str, Dict[str, Any]]:
    """ALL review entries keyed by negative_item_id (proposed/approved/
    excluded). build_musique applies the status rules: approved pairs re-enter
    manifests, human-excluded pairs stay out everywhere and are reported as
    review: exclusions. Entries without a negative item (build-excluded
    parents) are not keyed here - the build itself reports those."""
    if not review_path.exists():
        return {}
    with open(review_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["negative_item_id"]: e for e in entries if e.get("negative_item_id")}


def load_approved_pairs(review_path: Path) -> Dict[str, Dict[str, Any]]:
    """Approved negative pairs from a review file (never auto-approved)."""
    return {
        k: e
        for k, e in load_review_map(review_path).items()
        if e.get("review_status") == "approved" and e.get("human_reviewed")
    }


def split_parents(records: Dict[str, List[Dict[str, Any]]], cfg: ConfigBundle) -> Dict[str, List[str]]:
    """Deterministic parent-id assignment to debug/selection/holdout pools.

    MuSiQue labeled training data -> debug pool (smoke/pilot).
    MuSiQue public development data -> selection (first N eligible parents)
    and holdout (next N), disjoint by parent id. Only eligible parents count
    toward the quotas; shortfalls are reported, never fabricated.
    """
    train_ids = sorted(r["id"] for r in records.get("train", []))
    dev_ids = sorted(r["id"] for r in records.get("dev", []))
    sel_n = cfg.splits["stages"]["selection"]["musique_parents"]
    hold_n = cfg.splits["stages"]["holdout"]["musique_parents"]
    return {
        "debug": train_ids,
        "selection": dev_ids[:sel_n],
        "holdout": dev_ids[sel_n : sel_n + hold_n],
    }
