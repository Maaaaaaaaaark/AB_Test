"""MuSiQue local builder: references, leakage, aliases, distractors, review."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from weval.datasets.musique import (
    UnresolvableReference,
    build_musique,
    build_positive,
    build_pair,
    non_supporting_anywhere,
    resolve_references,
    split_parents,
    write_review_file,
)
from tests.conftest import make_cfg, musique_record


class TestResolveReferences:
    def test_resolves_earlier_prerequisite_inline(self, record):
        q = "#1 >> population"
        resolved, used = resolve_references(q, record["question_decomposition"], 1)
        assert resolved == "Paris >> population"
        assert used == [(1, "Paris")]

    def test_rejects_current_and_later_references(self, record):
        # #2 at subquestion 0 refers to a later subquestion -> unresolved
        with pytest.raises(UnresolvableReference):
            resolve_references("#2 >> something", record["question_decomposition"], 0)
        # self reference
        with pytest.raises(UnresolvableReference):
            resolve_references("#1 >> x", record["question_decomposition"], 0)

    def test_multiple_references(self):
        rec = musique_record(subquestions=[
            {"id": 1, "question": "Who wrote A?", "answer": "Author", "paragraph_support_idx": 0},
            {"id": 2, "question": "Who wrote B?", "answer": "Writer", "paragraph_support_idx": 1},
            {"id": 3, "question": "What is the difference between #1 and #2?", "answer": "Genre", "paragraph_support_idx": 2},
        ])
        resolved, used = resolve_references(rec["question_decomposition"][2]["question"], rec["question_decomposition"], 2)
        assert resolved == "What is the difference between Author and Writer?"
        assert used == [(1, "Author"), (2, "Writer")]


class TestLeakage:
    def test_built_question_never_contains_current_answer(self, record):
        """The question must not contain the CURRENT subanswer or parent answer."""
        pos = build_positive(record, 1, "selection")
        q = pos.question
        assert "2.1 million" not in q  # current subanswer
        assert record["answer"] not in q  # parent answer

    def test_built_question_never_contains_later_answers(self):
        rec = musique_record(subquestions=[
            {"id": 1, "question": "Who wrote A?", "answer": "Author", "paragraph_support_idx": 0},
            {"id": 2, "question": "Who wrote B?", "answer": "Writer", "paragraph_support_idx": 1},
            {"id": 3, "question": "#1 and #2 >> genre", "answer": "Genre", "paragraph_support_idx": 2},
        ])
        # build the terminal subquestion (index 2), whose #1/#2 refs resolve
        # to the earlier canonical answers inline
        pos = build_positive(rec, 2, "selection")
        # current subanswer "Genre" must be absent
        assert "Genre" not in pos.question
        # prerequisites "Author" and "Writer" appear (resolved inline)
        assert "Author" in pos.question
        assert "Writer" in pos.question


class TestAliases:
    def test_terminal_step_uses_parent_answer_and_aliases(self):
        # parent answer == canonical subanswer under official normalization
        # -> the parent answer AND its aliases become the gold set
        rec = musique_record(
            answer="2.1 million",
            aliases=["2.1M", "2,100,000"],
            subquestions=[
                {"id": 1, "question": "What city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "#1 >> population", "answer": "2.1 million", "paragraph_support_idx": 1},
            ],
        )
        pos = build_positive(rec, 1, "selection")
        assert pos.is_terminal
        assert pos.reference_answers == ["2.1 million", "2.1M", "2,100,000"]
        assert pos.alias_source == "parent_answer_aliases"

    def test_terminal_step_canonical_mismatch_falls_back_to_subanswer(self):
        rec = musique_record(
            answer="Paris, France",
            subquestions=[
                {"id": 1, "question": "What city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "#1 >> country", "answer": "France", "paragraph_support_idx": 1},
            ],
        )
        # canonical subanswer "France" != parent answer "Paris, France" under
        # official normalization -> subanswer used, alias source recorded
        pos = build_positive(rec, 1, "selection")
        assert pos.reference_answers == ["France"]
        assert pos.alias_source == "subquestion_canonical_only_terminal_mismatch"

    def test_terminal_step_parent_answer_under_normalization(self):
        rec = musique_record(
            answer="2.1 million people",
            subquestions=[
                {"id": 1, "question": "What city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "#1 >> population", "answer": "2.1 million", "paragraph_support_idx": 1},
            ],
        )
        pos = build_positive(rec, 1, "selection")
        # normalize("2.1 million") != normalize("2.1 million people") -> the
        # subanswer is the gold and the mismatch is recorded as an anomaly
        assert pos.reference_answers == ["2.1 million"]
        assert pos.alias_source == "subquestion_canonical_only_terminal_mismatch"


class TestStrictDistractors:
    def test_distractors_are_non_supporting_anywhere(self, record):
        pos = build_positive(record, 1, "selection")
        src = [p.source_idx for p in pos.visible]
        assert 1 in src  # the support paragraph for subq 1
        pool = {p["idx"] for p in non_supporting_anywhere(record)}
        for p in pos.visible:
            if p.role == "distractor":
                assert p.source_idx in pool
        # the OTHER supporting paragraph (idx 0) must never be a distractor
        for p in pos.visible:
            if p.role == "distractor":
                assert p.source_idx != 0

    def test_insufficient_distractors_excludes_parent(self):
        # only 3 non-supporting paragraphs -> subq1 unbuildable
        rec = musique_record(paragraphs=[
            {"idx": 0, "title": "A", "paragraph_text": "A text.", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "B text.", "is_supporting": True},
            {"idx": 2, "title": "C", "paragraph_text": "C text.", "is_supporting": False},
            {"idx": 3, "title": "D", "paragraph_text": "D text.", "is_supporting": False},
            {"idx": 4, "title": "E", "paragraph_text": "E text.", "is_supporting": False},
        ])
        items, report, exclusions = build_musique([rec], "selection", 0)
        assert report.n_parents_eligible == 0
        assert report.parent_eligibility_reasons.get("insufficient_non_supporting_paragraphs", 0) >= 1

    def test_five_visible_paragraphs(self, record):
        pos = build_positive(record, 1, "selection")
        assert len(pos.visible) == 5
        slots = sorted(p.slot for p in pos.visible)
        assert slots == [1, 2, 3, 4, 5]


class TestReviewFile:
    def test_review_entry_shape(self, record, tmp_path):
        pos = build_positive(record, 1, "selection")
        neg, reason = build_pair(record, 1, pos, "selection")
        assert neg is not None and reason is None
        n = write_review_file(
            tmp_path / "review.json",
            [record],
            {record["id"]: pos},
            {record["id"]: neg},
            {},
        )
        assert n == 1
        entries = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        e = entries[0]
        assert e["parent_id"] == record["id"]
        assert e["review_status"] == "proposed"
        assert e["human_reviewed"] is False
        assert e["negative_item_id"] == neg.item_id
        assert "checks" in e

    def test_review_file_preserves_human_decisions_on_rebuild(self, record, tmp_path):
        """A rebuild (which regenerates the review file) must not wipe human
        decisions: approved stays approved with its reviewer/note fields."""
        pos = build_positive(record, 1, "selection")
        neg, reason = build_pair(record, 1, pos, "selection")
        assert neg is not None and reason is None
        path = tmp_path / "review.json"
        write_review_file(path, [record], {record["id"]: pos}, {record["id"]: neg}, {})
        entries = json.loads(path.read_text(encoding="utf-8"))
        entries[0].update(
            review_status="approved", human_reviewed=True,
            reviewed_by="tester", review_note="safe negative",
        )
        path.write_text(json.dumps(entries), encoding="utf-8")
        # rebuild: decision fields survive, build-derived fields regenerate
        write_review_file(path, [record], {record["id"]: pos}, {record["id"]: neg}, {})
        e = json.loads(path.read_text(encoding="utf-8"))[0]
        assert e["review_status"] == "approved"
        assert e["human_reviewed"] is True
        assert e["reviewed_by"] == "tester"
        assert e["review_note"] == "safe negative"
        assert "checks" in e

    def test_review_file_preserves_human_exclusion_on_rebuild(self, record, tmp_path):
        """After a human exclusion the pair no longer builds (the rebuild
        passes negative=None); the human decision must survive instead of
        being overwritten by a build-derived entry."""
        pos = build_positive(record, 1, "selection")
        neg, _ = build_pair(record, 1, pos, "selection")
        path = tmp_path / "review.json"
        write_review_file(path, [record], {record["id"]: pos}, {record["id"]: neg}, {})
        entries = json.loads(path.read_text(encoding="utf-8"))
        entries[0].update(
            review_status="excluded", human_reviewed=True,
            reviewed_by="tester", excluded_reason="answer in title",
        )
        path.write_text(json.dumps(entries), encoding="utf-8")
        # rebuild after the exclusion: the pair is gone from the build output
        write_review_file(path, [record], {record["id"]: pos}, {record["id"]: None}, {})
        e = json.loads(path.read_text(encoding="utf-8"))[0]
        assert e["review_status"] == "excluded"
        assert e["human_reviewed"] is True
        assert e["reviewed_by"] == "tester"
        assert e["excluded_reason"] == "answer in title"
        assert e["negative_item_id"] is None

    def test_split_parents_disjoint(self, record):
        recs = {
            "dev": [musique_record(pid=f"2hop__{i}_{i+1}") for i in range(20)],
            "train": [],
        }
        cfg = make_cfg()
        pools = split_parents(recs, cfg)
        sel = set(pools["selection"])
        hold = set(pools["holdout"])
        assert len(sel) == 6 and len(hold) == 6
        assert sel & hold == set()
