"""MuSiQue-local scoring of missing/abstained answers.

The musique_local path is parse (weval/parsing/musique_answer.py) -> score
(weval/scoring/musique_em_f1.py). Encoded here:

  * NOT FOUND strictness: after whitespace/eos trimming the ENTIRE final
    region must equal "not found" case-insensitively. On a NEGATIVE item
    that is a correct abstention (1.0); anything else (empty output, an
    answer attempt, "not found" plus trailing text, punctuation) is a false
    answer (0.0).
  * On a POSITIVE item NOT FOUND is a false abstention: value 0.0 with the
    flag set, and the item is excluded from answerable-F1.
  * An open reasoning region at EOS (or an empty final region) parses to
    missing_final: value 0.0 on a positive, NOT an abstention, and the
    missing_final status propagates to the ScoreRecord.
  * finish_reason "length" (token-limit truncation) is an evaluated outcome:
    whatever text exists is parsed and scored, never unscorable and never an
    infra-failure.
  * Gold alias matching: the terminal step's reference answers are the parent
    answer plus its aliases (parent_answer_aliases), and a prediction matching
    an alias scores against it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from weval.datasets.musique import build_positive
from weval.parsing.musique_answer import parse_musique_answer
from weval.scoring.musique_em_f1 import score_musique, score_musique_batch
from tests.conftest import musique_record

MODE = "no_thinking_channel"


def _item(
    kind: str = "positive",
    golds: Optional[List[str]] = None,
    item_id: str = "it1",
    group_id: str = "g1",
    split: str = "smoke",
) -> Dict[str, Any]:
    golds = list(golds) if golds is not None else []
    return {
        "item_id": item_id,
        "group_id": group_id,
        "split": split,
        "task": "musique_local",
        "kind": kind,
        "reference_answers": golds,
        "gold": {"reference_answers": golds},
    }


def _score(text: str, item: Dict[str, Any], mode: str = MODE):
    return score_musique(item, parse_musique_answer(text, mode))


class TestNegativeNotFoundStrictness:
    NEG = _item(kind="negative")

    @pytest.mark.parametrize("text", ["NOT FOUND", "not found", "Not Found"])
    def test_not_found_scores_correct_abstention(self, text):
        # exact "not found" (case-insensitive) is the ONLY correct behavior
        rec = _score(text, self.NEG)
        assert rec.value == 1.0
        assert rec.status == "scored"
        assert rec.detail["correct_abstention"] is True
        assert rec.detail["abstained"] is True
        assert rec.detail["false_answer"] is False

    @pytest.mark.parametrize(
        "text",
        [
            "",  # empty output
            "an actual answer",
            "not found, sorry",  # trailing text: strict full match
            "NOT FOUND.",  # punctuation not stripped
            "I think NOT FOUND",  # substring match not allowed
            "notfound",  # no space
        ],
    )
    def test_anything_else_scores_zero(self, text):
        rec = _score(text, self.NEG)
        assert rec.value == 0.0
        assert rec.detail["correct_abstention"] is False
        assert rec.detail["abstained"] is False
        assert rec.detail["false_answer"] is True

    def test_empty_output_propagates_missing_final(self):
        rec = _score("", self.NEG)
        assert rec.value == 0.0
        assert rec.status == "missing_final"


class TestPositiveFalseAbstention:
    POS = _item(kind="positive", golds=["2.1 million"])

    @pytest.mark.parametrize("text", ["NOT FOUND", "not found"])
    def test_not_found_on_positive_is_false_abstention(self, text):
        rec = _score(text, self.POS)
        assert rec.value == 0.0
        assert rec.status == "scored"
        assert rec.detail["false_abstention"] is True
        assert rec.detail["abstained"] is True
        assert rec.detail["answerable"] is False
        assert rec.detail["em"] == 0
        assert rec.detail["f1"] == 0.0


class TestPositiveMissingFinal:
    POS = _item(kind="positive", golds=["2.1 million"])

    def test_open_reasoning_region_scores_zero_not_abstention(self):
        rec = _score("<think>unfinished", self.POS, mode="prefilled_thinking")
        assert rec.value == 0.0
        assert rec.status == "missing_final"  # status propagates to the record
        assert rec.detail["abstained"] is False
        assert rec.detail["false_abstention"] is False
        assert rec.detail["answerable"] is True
        assert rec.detail["predicted_answer"] is None
        assert rec.detail["em"] == 0
        assert rec.detail["f1"] == 0.0

    def test_empty_output_scores_zero(self):
        rec = _score("", self.POS)
        assert rec.value == 0.0
        assert rec.status == "missing_final"
        assert rec.detail["abstained"] is False
        assert rec.detail["false_abstention"] is False

    def test_closed_thinking_without_answer_scores_zero(self):
        rec = _score("<think>only reasoning</think>", self.POS, mode="optional_thinking")
        assert rec.value == 0.0
        assert rec.status == "missing_final"


class TestTruncationIsEvaluatedOutcome:
    """finish_reason "length" is a normal evaluated outcome, never a crash."""

    POS = _item(kind="positive", golds=["Paris"])

    @staticmethod
    def _gen_record(output_text: str, eos: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "record_version": "prediction-record-v1",
            "status": "ok",
            "item_id": "it1",
            "group_id": "g1",
            "task": "musique_local",
            "kind": "positive",
            "finish_reason": "length",
            "output_text": output_text,
            "eos_token_strings": eos or [],
        }

    def _parse_and_score(self, record: Dict[str, Any], mode: str):
        parsed = parse_musique_answer(record["output_text"], mode, record.get("eos_token_strings"))
        return parsed, score_musique(self.POS, parsed)

    def test_truncated_mid_reasoning_is_scored_missing_final(self):
        rec = self._gen_record("Long chain-of-thought that got cut off")
        parsed, score = self._parse_and_score(rec, "prefilled_thinking")
        assert parsed.status == "missing_final"
        assert score.value == 0.0
        assert score.status == "missing_final"
        assert score.status != "unscorable"

    def test_truncated_with_complete_answer_scores_actual(self):
        rec = self._gen_record("Paris")
        parsed, score = self._parse_and_score(rec, MODE)
        assert parsed.status == "ok"
        assert score.value == 1.0
        assert score.status == "scored"

    def test_truncated_with_wrong_answer_scores_zero_not_unscorable(self):
        rec = self._gen_record("Par")
        _, score = self._parse_and_score(rec, MODE)
        assert score.value == 0.0
        assert score.status == "scored"
        assert score.status != "unscorable"

    def test_truncated_output_with_eos_tokens_still_scored(self):
        rec = self._gen_record("Paris<|im_end|>", eos=["<|im_end|>"])
        _, score = self._parse_and_score(rec, MODE)
        assert score.value == 1.0
        assert score.status == "scored"


class TestGoldAliasMatching:
    def test_parent_answer_aliases_are_reference_answers(self):
        rec = musique_record(
            answer="2.1 million",
            aliases=["2,100,000 people"],
            subquestions=[
                {"id": 1, "question": "What city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "#1 >> population", "answer": "2.1 million", "paragraph_support_idx": 1},
            ],
        )
        item = build_positive(rec, 1, "selection")
        assert item.alias_source == "parent_answer_aliases"
        d = item.to_dict()
        assert d["reference_answers"] == ["2.1 million", "2,100,000 people"]
        assert d["gold"]["parent_answer_aliases"] == ["2,100,000 people"]

        # a prediction matching an ALIAS scores against the alias
        alias_rec = score_musique(d, parse_musique_answer("2,100,000 people", MODE))
        assert alias_rec.value == 1.0
        assert alias_rec.detail["em"] == 1
        assert alias_rec.detail["is_correct"] is True
        assert alias_rec.detail["predicted_answer"] == "2,100,000 people"

        # the parent answer itself still scores
        parent_rec = score_musique(d, parse_musique_answer("2.1 million", MODE))
        assert parent_rec.value == 1.0
        assert parent_rec.detail["em"] == 1

    def test_an_unmatched_answer_scores_zero_against_alias_set(self):
        rec = musique_record(
            answer="2.1 million",
            aliases=["2,100,000 people"],
            subquestions=[
                {"id": 1, "question": "What city?", "answer": "Paris", "paragraph_support_idx": 0},
                {"id": 2, "question": "#1 >> population", "answer": "2.1 million", "paragraph_support_idx": 1},
            ],
        )
        d = build_positive(rec, 1, "selection").to_dict()
        bad = score_musique(d, parse_musique_answer("London", MODE))
        assert bad.value == 0.0
        assert bad.status == "scored"
        assert bad.detail["em"] == 0


class TestUnscorableOnlyForMissingGold:
    def test_positive_without_gold_answers_is_unscorable(self):
        rec = _score("Paris", _item(kind="positive", golds=None))
        assert rec.status == "unscorable"
        assert rec.value is None
        assert rec.detail["reason"] == "no_gold_answers"

    def test_negative_items_are_never_unscorable(self):
        # negatives are scored on abstention behavior alone; golds irrelevant
        rec = _score("NOT FOUND", _item(kind="negative", golds=None))
        assert rec.status == "scored"
        assert rec.value == 1.0

    def test_batch_scores_only_items_with_parsed_answers(self):
        items = [
            _item(kind="positive", golds=["Paris"], item_id="a"),
            _item(kind="positive", golds=["Rome"], item_id="b"),
            _item(kind="positive", golds=["Berlin"], item_id="c"),
        ]
        parsed_by_id = {
            "a": parse_musique_answer("Paris", MODE),
            "b": parse_musique_answer("NOT FOUND", MODE),
        }
        records = score_musique_batch(items, parsed_by_id)
        assert [r.item_id for r in records] == ["a", "b"]
        assert records[0].value == 1.0
        assert records[1].value == 0.0
