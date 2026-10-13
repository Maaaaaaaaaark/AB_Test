"""Paired negatives: construction, screening, exclusions, approval re-entry.

Extends tests/test_musique_builder.py (which covers reference resolution,
leakage into the built question, aliases, strict distractors, review entry
shape and split disjointness). Here: the negative/positive pair contract
itself (same question, same slot, one replaced paragraph), the literal
screening checks and the yes/no + numeric policy exclusions, the exclusion
accounting for insufficient distractors, and the full human-review lifecycle
(approve / exclude / reset) with its re-entry into approved-only builds.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from weval.datasets.musique import (
    build_musique,
    build_pair,
    build_positive,
    load_approved_pairs,
    non_supporting_anywhere,
    write_review_file,
)
from weval.prompts.renderers import render_item
from weval.review import ReviewError, apply_decision, list_entries, review_summary
from tests.conftest import make_cfg, musique_record

SPLIT = "selection"
SEED = 20261013


def _paragraph_blocks(user_message: str) -> List[str]:
    """The five 'Paragraph N: title: text' blocks of a rendered musique prompt."""
    return [b for b in user_message.split("\n\n") if b.startswith("Paragraph ")]


def _non_paragraph_lines(user_message: str) -> List[str]:
    return [ln for ln in user_message.splitlines() if not ln.startswith("Paragraph ")]


def _paragraphs(replacement: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Default 8-paragraph layout with a custom fifth non-supporting paragraph."""
    return [
        {"idx": 0, "title": "Paris", "paragraph_text": "Paris is the capital of France.", "is_supporting": True},
        {"idx": 1, "title": "Paris population", "paragraph_text": "Paris has a population of 2.1 million.", "is_supporting": True},
        {"idx": 2, "title": "London", "paragraph_text": "London is the capital of England.", "is_supporting": False},
        {"idx": 3, "title": "Rome", "paragraph_text": "Rome is the capital of Italy.", "is_supporting": False},
        {"idx": 4, "title": "Berlin", "paragraph_text": "Berlin is the capital of Germany.", "is_supporting": False},
        {"idx": 5, "title": "Madrid", "paragraph_text": "Madrid is the capital of Spain.", "is_supporting": False},
        replacement,
        {"idx": 7, "title": "Beijing", "paragraph_text": "Beijing is the capital of China.", "is_supporting": False},
    ]


def _four_distractor_paragraphs() -> List[Dict[str, Any]]:
    """Two supporting + four non-supporting: positive buildable, pair is not."""
    return [
        {"idx": 0, "title": "A", "paragraph_text": "A text.", "is_supporting": True},
        {"idx": 1, "title": "B", "paragraph_text": "B text.", "is_supporting": True},
        {"idx": 2, "title": "C", "paragraph_text": "C text.", "is_supporting": False},
        {"idx": 3, "title": "D", "paragraph_text": "D text.", "is_supporting": False},
        {"idx": 4, "title": "E", "paragraph_text": "E text.", "is_supporting": False},
        {"idx": 5, "title": "F", "paragraph_text": "F text.", "is_supporting": False},
    ]


class TestPairConstruction:
    def test_negative_keeps_question_and_instruction_rendering(self):
        cfg = make_cfg()
        record = musique_record()
        pos = build_positive(record, 1, SPLIT)
        neg, reason = build_pair(record, 1, pos, SPLIT)
        assert neg is not None and reason is None

        # Same resolved question text as its positive sibling.
        assert neg.question == pos.question == "Paris >> population"

        pos_msg = render_item(pos.to_dict(), cfg).user_message
        neg_msg = render_item(neg.to_dict(), cfg).user_message
        # Instruction, relation note, question and answer stub are identical.
        assert _non_paragraph_lines(pos_msg) == _non_paragraph_lines(neg_msg)

        pos_blocks = _paragraph_blocks(pos_msg)
        neg_blocks = _paragraph_blocks(neg_msg)
        assert len(pos_blocks) == len(neg_blocks) == 5
        supp_slot = next(p.slot for p in pos.visible if p.role == "support")
        diff = [i for i, (a, b) in enumerate(zip(pos_blocks, neg_blocks)) if a != b]
        # Exactly one paragraph block differs, and it is the support slot.
        assert diff == [supp_slot - 1]

        assert render_item(pos.to_dict(), cfg).kind == "positive"
        assert render_item(neg.to_dict(), cfg).kind == "negative"

    def test_support_replaced_by_fifth_non_supporting_at_same_slot(self):
        cfg = make_cfg()
        record = musique_record()
        pos = build_positive(record, 1, SPLIT)
        neg, reason = build_pair(record, 1, pos, SPLIT)
        assert neg is not None and reason is None

        pool = non_supporting_anywhere(record)
        fifth = sorted(pool, key=lambda p: p["idx"])[4]
        supp_idx = record["question_decomposition"][1]["paragraph_support_idx"]
        supp = next(p for p in record["paragraphs"] if p["idx"] == supp_idx)

        supp_slot = next(p.slot for p in pos.visible if p.role == "support")
        neg_at_slot = next(p for p in neg.visible if p.slot == supp_slot)
        assert neg_at_slot.source_idx == fifth["idx"]
        assert neg_at_slot.role == "distractor"
        assert neg_at_slot.title == fifth["title"]
        assert neg_at_slot.text == fifth["paragraph_text"]

        # Every other slot is byte-identical to the positive.
        others_pos = {p.slot: p for p in pos.visible if p.slot != supp_slot}
        others_neg = {p.slot: p for p in neg.visible if p.slot != supp_slot}
        assert others_pos.keys() == others_neg.keys()
        for slot in others_pos:
            a, b = others_pos[slot], others_neg[slot]
            assert (a.source_idx, a.title, a.text, a.role) == (
                b.source_idx, b.title, b.text, b.role,
            )

        # The rendered prompts differ only inside that block.
        pos_block = _paragraph_blocks(render_item(pos.to_dict(), cfg).user_message)[supp_slot - 1]
        neg_block = _paragraph_blocks(render_item(neg.to_dict(), cfg).user_message)[supp_slot - 1]
        assert pos_block == f"Paragraph {supp_slot}: {supp['title']}: {supp['paragraph_text']}"
        assert neg_block == f"Paragraph {supp_slot}: {fifth['title']}: {fifth['paragraph_text']}"

    def test_negative_item_fields(self):
        record = musique_record()
        pos = build_positive(record, 1, SPLIT)
        neg, reason = build_pair(record, 1, pos, SPLIT)
        assert neg is not None and reason is None

        assert neg.kind == "negative" and pos.kind == "positive"
        assert neg.pair_id == pos.pair_id == f"{record['id']}:s1:pair"
        assert neg.parent_id == pos.parent_id == record["id"]
        assert neg.expected == "not_found" and pos.expected == "answer"
        assert neg.review_status == "proposed" and not neg.human_reviewed
        assert neg.excluded_reason is None
        assert neg.reference_answers == pos.reference_answers
        assert neg.syntax == pos.syntax
        assert neg.gold["replaced_with_paragraph_idx"] == sorted(
            p["idx"] for p in non_supporting_anywhere(record)
        )[4]
        assert neg.gold["support_paragraph_idx"] == record["question_decomposition"][1][
            "paragraph_support_idx"
        ]
        assert sorted(p.slot for p in neg.visible) == [1, 2, 3, 4, 5]


class TestScreening:
    def _pair(self, record: Dict[str, Any], subq: int = 1):
        pos = build_positive(record, subq, SPLIT)
        return build_pair(record, subq, pos, SPLIT)

    def test_gold_answer_in_replacement_title_excluded(self):
        record = musique_record(paragraphs=_paragraphs(
            {"idx": 6, "title": "2.1 million residents", "paragraph_text": "Many people live in the metro area.", "is_supporting": False},
        ))
        neg, reason = self._pair(record)
        assert neg is None
        assert reason == "answer_visible_in_title"

    def test_subanswer_in_replacement_text_excluded(self):
        # Terminal normalization mismatch -> gold is the canonical subanswer.
        record = musique_record(paragraphs=_paragraphs(
            {"idx": 6, "title": "Demographics", "paragraph_text": "The city has 2.1 million residents.", "is_supporting": False},
        ))
        neg, reason = self._pair(record)
        assert neg is None
        assert reason == "answer_visible_in_paragraph"

    def test_parent_alias_in_replacement_text_excluded(self):
        record = musique_record(
            pid="2hop__alias",
            answer="Paris",
            aliases=["City of Light"],
            subquestions=[
                {"id": 1, "question": "Which city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "Which city is the capital of France?", "answer": "Paris", "paragraph_support_idx": 1},
            ],
            paragraphs=_paragraphs(
                {"idx": 6, "title": "Tourism", "paragraph_text": "The City of Light attracts many visitors.", "is_supporting": False},
            ),
        )
        neg, reason = self._pair(record)
        assert neg is None
        assert reason == "answer_visible_in_paragraph"

    def test_answer_in_resolved_prerequisites_excluded(self):
        record = musique_record(
            pid="2hop__prereq",
            answer="Paris",
            subquestions=[
                {"id": 1, "question": "Which city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "What is #1?", "answer": "Paris", "paragraph_support_idx": 1},
            ],
        )
        neg, reason = self._pair(record)
        assert neg is None
        assert reason == "answer_visible_in_prerequisites"

    def test_ambiguous_yes_no_negative_excluded_by_policy(self):
        record = musique_record(
            pid="2hop__yesno",
            answer="yes",
            subquestions=[
                {"id": 1, "question": "Is the sky blue?", "answer": "yes", "paragraph_support_idx": 0},
            ],
        )
        neg, reason = self._pair(record, subq=0)
        assert neg is None
        assert reason == "ambiguous_yes_no"

    def test_numeric_negative_excluded_by_policy(self):
        record = musique_record(
            pid="2hop__num",
            answer="42",
            subquestions=[
                {"id": 1, "question": "What is six times seven?", "answer": "42", "paragraph_support_idx": 0},
            ],
        )
        neg, reason = self._pair(record, subq=0)
        assert neg is None
        assert reason == "numeric_answer_policy"

    def test_screening_failure_recorded_in_build_report(self):
        record = musique_record(paragraphs=_paragraphs(
            {"idx": 6, "title": "2.1 million residents", "paragraph_text": "Many people live in the metro area.", "is_supporting": False},
        ))
        items, report, pair_exclusions = build_musique([record], SPLIT, SEED)
        assert pair_exclusions[record["id"]] == "answer_visible_in_title"
        assert report.exclusions.get("answer_visible_in_title") == 1
        assert report.n_pair_candidates == 0
        assert report.n_positives_built == 1
        assert all(i.kind == "positive" for i in items)


class TestExclusionAccounting:
    def test_four_distractor_pool_builds_positive_but_never_a_negative(self):
        record = musique_record(paragraphs=_four_distractor_paragraphs())
        items, report, pair_exclusions = build_musique([record], SPLIT, SEED)
        assert report.n_parents_eligible == 1
        assert report.n_positives_built == 1
        assert report.n_pair_candidates == 0
        assert pair_exclusions[record["id"]] == "insufficient_non_supporting_paragraphs"
        assert report.exclusions == {"insufficient_non_supporting_paragraphs": 1}
        assert [i.kind for i in items] == ["positive"]  # never yields a negative item

    def test_direct_pair_build_returns_documented_reason(self):
        record = musique_record(paragraphs=_four_distractor_paragraphs())
        pos = build_positive(record, 1, SPLIT)
        neg, reason = build_pair(record, 1, pos, SPLIT)
        assert neg is None
        assert reason == "insufficient_non_supporting_paragraphs"


class TestApprovalReentry:
    def _parents(self) -> List[Dict[str, Any]]:
        return [
            musique_record(pid="2hop__pair_a"),
            musique_record(pid="2hop__pair_b"),
            musique_record(pid="2hop__pair_c"),
            musique_record(pid="2hop__pair_d", paragraphs=_four_distractor_paragraphs()),
        ]

    @staticmethod
    def _collect(records: List[Dict[str, Any]]):
        positives: Dict[str, Any] = {}
        negatives: Dict[str, Optional[Any]] = {}
        exclusions: Dict[str, str] = {}
        for rec in records:
            items, _report, pair_exclusions = build_musique([rec], SPLIT, SEED)
            positives[rec["id"]] = next(i for i in items if i.kind == "positive")
            negatives[rec["id"]] = next((i for i in items if i.kind == "negative"), None)
            exclusions.update(pair_exclusions)
        return positives, negatives, exclusions

    def test_review_flow_and_reentry(self, tmp_path):
        parents = self._parents()
        positives, negatives, exclusions = self._collect(parents)
        path = tmp_path / "manifests" / "musique_negative_review.json"
        n_candidates = write_review_file(path, parents, positives, negatives, exclusions)
        assert n_candidates == 3

        entries = json.loads(path.read_text(encoding="utf-8"))
        by_pid = {e["parent_id"]: e for e in entries}
        assert by_pid["2hop__pair_d"]["review_status"] == "excluded"
        assert by_pid["2hop__pair_d"]["negative_item_id"] is None
        assert by_pid["2hop__pair_d"]["excluded_reason"] == "insufficient_non_supporting_paragraphs"

        # Human decisions: approve a candidate, exclude another with a note.
        apply_decision(path, "2hop__pair_a", "approve", note="negative is clearly unanswerable")
        apply_decision(path, "2hop__pair_b", "exclude", note="candidate is a trick question")

        # Approve requires a pair candidate; exclude requires a note;
        # decisions must be from the fixed vocabulary; parents must exist.
        with pytest.raises(ReviewError):
            apply_decision(path, "2hop__pair_d", "approve")
        with pytest.raises(ReviewError):
            apply_decision(path, "2hop__pair_c", "exclude")
        with pytest.raises(ReviewError):
            apply_decision(path, "2hop__pair_c", "banish")
        with pytest.raises(ReviewError):
            apply_decision(path, "2hop__nope", "approve")

        assert review_summary(path) == {
            "total": 4, "proposed": 1, "approved": 1, "excluded": 2,
            "excluded_at_build": 1, "human_excluded": 1,
        }

        # Only approved + human_reviewed + with a negative_item_id re-enter.
        approved = load_approved_pairs(path)
        assert list(approved) == ["2hop__pair_a:s1:neg"]

        items, report, pair_exclusions = build_musique(
            parents, SPLIT, SEED, require_approved=True, approved_pairs=approved,
        )
        neg_a = next(i for i in items if i.kind == "negative")
        assert neg_a.review_status == "approved" and neg_a.human_reviewed
        assert report.n_pairs_approved == 1
        # Human-excluded (B) and unreviewed (C) candidates stay out of an
        # approved-only build; via load_approved_pairs they are not tracked
        # in pair_exclusions (see the dedicated branch test below).
        assert "2hop__pair_b" not in pair_exclusions
        assert "2hop__pair_c" not in pair_exclusions
        assert [i for i in items if i.parent_id in ("2hop__pair_b", "2hop__pair_c") and i.kind == "negative"] == []
        # Build-excluded parent still fails with its documented build reason.
        assert pair_exclusions["2hop__pair_d"] == "insufficient_non_supporting_paragraphs"
        assert report.exclusions.get("insufficient_non_supporting_paragraphs") == 1

        # Development mode (smoke) includes unreviewed candidates, clearly labeled.
        rec_c = next(p for p in parents if p["id"] == "2hop__pair_c")
        items_dev, report_dev, _ = build_musique([rec_c], "smoke", SEED)
        neg_c = next(i for i in items_dev if i.kind == "negative")
        assert neg_c.review_status == "proposed" and not neg_c.human_reviewed

        # Reset undoes the approval; the pair leaves the approved map again.
        apply_decision(path, "2hop__pair_a", "reset")
        assert load_approved_pairs(path) == {}
        assert list_entries(path, only="approved") == []
        proposed = list_entries(path, only="proposed")
        assert {e["parent_id"] for e in proposed} == {"2hop__pair_a", "2hop__pair_c"}

    def test_build_musique_records_review_excluded_entries_when_supplied(self):
        """The review-excluded branch: reason is recorded as 'review:<reason>'.

        Note: load_approved_pairs returns ONLY approved entries, so the real
        pipeline never reaches this branch for human-excluded pairs (they stay
        out silently, per weval/review.py). The branch is exercised here with
        a full entry map to pin its accounting behavior.
        """
        rec = musique_record(pid="2hop__pair_x")
        items, report, pair_exclusions = build_musique(
            [rec], SPLIT, SEED, require_approved=True,
            approved_pairs={
                f"{rec['id']}:s1:neg": {"review_status": "excluded", "excluded_reason": "reviewer said no"},
            },
        )
        assert [i for i in items if i.kind == "negative"] == []
        assert pair_exclusions[rec["id"]] == "review:reviewer said no"
        assert report.exclusions.get("review:reviewer said no") == 1

    def test_load_approved_pairs_requires_all_three_conditions(self, tmp_path):
        manual = [
            {"parent_id": "p1", "review_status": "approved", "human_reviewed": True, "negative_item_id": "p1:s1:neg"},
            # Approved as a positive-only entry: must NOT enter the approved map.
            {"parent_id": "p2", "review_status": "approved", "human_reviewed": True, "negative_item_id": None},
            {"parent_id": "p3", "review_status": "approved", "human_reviewed": False, "negative_item_id": "p3:s1:neg"},
            {"parent_id": "p4", "review_status": "excluded", "human_reviewed": True, "negative_item_id": "p4:s1:neg"},
        ]
        path = tmp_path / "review_manual.json"
        path.write_text(json.dumps(manual), encoding="utf-8")
        approved = load_approved_pairs(path)
        assert list(approved) == ["p1:s1:neg"]
        assert approved["p1:s1:neg"] == manual[0]

    def test_load_approved_pairs_missing_file_is_empty(self, tmp_path):
        assert load_approved_pairs(tmp_path / "does_not_exist.json") == {}
