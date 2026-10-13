"""Parity between weval scorers and the VENDORED official evaluation code.

No network, no torch. Vendored files are read from vendor/ (as text where the
module imports torch at top level) or imported with the same sys.path dance
weval/scoring uses. The vendor copies are verbatim per vendor/VENDORED.md.

Covered:
  * MuSiQue  - our scorer's EM/F1 equal vendored compute_exact/compute_f1
               (articles, punctuation, numbers, partial overlap, max over
               golds) and share normalize_answer semantics.
  * GSM8K    - the ANS_RE pattern used by our parser is character-for-
               character identical to the vendored source's regex literal
               (byte-level source comparison), and extraction has official
               FIRST-match semantics (commas stripped, INVALID_ANS when no
               match).
  * ProcessBench - last \\boxed{...} on the FULL text -> int (vendored
               extract_answer semantics); scorer value 1.0 iff pred == gold
               label, including the -1 (correct solution) label.
  * HotpotQA - our scorer's value/em equal vendored exact_match_score and
               f1_score on several cases (including the yes/no zero rule).
  * MATH     - math_score.is_equiv IS the vendored heuristic (parity on
               several LaTeX/numeric pairs); missing predicted boxed ->
               0.0 + missing_final; missing GOLD boxed -> unscorable
               (gold_answer_has_no_boxed); two missing answers never match.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.conftest import PROJECT_ROOT
from weval.datasets.gsm8k import ANS_RE, INVALID_ANS, extract_answer
from weval.parsing.gsm8k_answer import parse_gsm8k_answer
from weval.parsing.hotpotqa_answer import parse_hotpotqa_answer
from weval.parsing.math_answer import parse_math_answer
from weval.parsing.processbench_answer import (
    BOXED_RE,
    official_extract_index,
    parse_processbench_answer,
)
from weval.scoring.gsm8k_score import score_gsm8k
from weval.scoring.hotpotqa_score import score_hotpotqa
from weval.scoring.math_score import is_equiv as weval_is_equiv
from weval.scoring.math_score import score_math
from weval.scoring.musique_em_f1 import score_musique
from weval.scoring.processbench_score import score_processbench

VENDOR = PROJECT_ROOT / "vendor"


def _musique_item(pred_text: str, golds, mode: str = "no_thinking_channel"):
    from weval.parsing.musique_answer import parse_musique_answer

    item = {
        "item_id": "it1",
        "group_id": "g1",
        "split": "smoke",
        "task": "musique_local",
        "kind": "positive",
        "reference_answers": list(golds),
        "gold": {"reference_answers": list(golds)},
    }
    return item, score_musique(item, parse_musique_answer(pred_text, mode))


# ---------------------------------------------------------------------------
# MuSiQue: import the vendored metric exactly like weval/scoring/musique_em_f1.py
# ---------------------------------------------------------------------------
for _p in (str(VENDOR / "musique"), str(VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from musique.metrics.answer import compute_exact, compute_f1, normalize_answer  # noqa: E402


class TestMusiqueOfficialParity:
    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("The Eiffel Tower", "Eiffel Tower"),  # article stripped
            ("Paris, France", "Paris France"),  # punctuation stripped
            ("2.1 million", "2.1 Million"),  # case
            ("New York", "New York City"),  # partial overlap
            ("answer with  extra   spaces", "answer with extra spaces"),
        ],
    )
    def test_f1_and_em_equal_vendored(self, pred, gold):
        _, rec = _musique_item(pred, [gold])
        assert rec.detail["em"] == compute_exact(pred, gold)
        assert rec.detail["f1"] == compute_f1(pred, gold)
        assert rec.value == compute_f1(pred, gold)

    def test_value_is_max_f1_over_golds(self):
        golds = ["Berlin, Germany", "Paris"]
        pred = "Paris"
        _, rec = _musique_item(pred, golds)
        assert rec.value == max(compute_f1(pred, g) for g in golds)
        assert rec.value == 1.0  # exact match against the second gold
        assert rec.detail["em"] == max(compute_exact(pred, g) for g in golds)
        assert rec.detail["em"] == 1

    def test_partial_overlap_scores_f1_not_em(self):
        golds = ["Berlin, Germany", "Paris, France"]
        pred = "Paris"
        _, rec = _musique_item(pred, golds)
        assert rec.value == max(compute_f1(pred, g) for g in golds)
        assert rec.value == pytest.approx(2 / 3, abs=1e-9)
        assert rec.detail["em"] == 0  # normalization makes "Paris" != "Paris, France"

    def test_normalization_matches_vendored(self):
        # the vendored normalize_answer drives both; spot-check its semantics
        assert normalize_answer("The  Eiffel   Tower!!!") == "eiffel tower"
        assert normalize_answer("2.1 million") == "21 million"
        assert compute_exact("New York City", "new york city") == 1
        assert compute_f1("", "") == 1  # official no-answer agreement rule
        assert compute_f1("", "Paris") == 0


# ---------------------------------------------------------------------------
# GSM8K: byte-level pattern parity + official first-match extraction
# ---------------------------------------------------------------------------
VENDOR_GSM8K_SRC = (VENDOR / "gsm8k" / "grade_school_math_dataset.py").read_text(encoding="utf-8")


class TestGsm8kOfficialPatternParity:
    def test_ans_re_literal_is_byte_identical_to_vendored_source(self):
        # the exact source line, character for character
        assert 'ANS_RE = re.compile(r"#### (\\-?[0-9\\.\\,]+)")' in VENDOR_GSM8K_SRC
        # compile the literal extracted from the vendored source and compare
        # the resulting pattern to the one our parser uses
        m = re.search(r'ANS_RE = re\.compile\(r"(?P<pat>[^"]*)"\)', VENDOR_GSM8K_SRC)
        assert m is not None
        vendored_pattern = m.group("pat")
        assert vendored_pattern == r"#### (\-?[0-9\.\,]+)"
        assert re.compile(vendored_pattern).pattern == ANS_RE.pattern

    def test_invalid_ans_literal_is_byte_identical(self):
        assert 'INVALID_ANS = "[invalid]"' in VENDOR_GSM8K_SRC
        assert INVALID_ANS == "[invalid]"

    def test_first_match_semantics(self):
        # official extraction: FIRST "#### <number>" match on the full text
        assert extract_answer("#### 1\n#### 2") == "1"
        assert extract_answer("#### -3") == "-3"
        assert extract_answer("#### 1,000") == "1000"  # commas stripped
        assert extract_answer("#### 1,000.50") == "1000.50"
        assert extract_answer("no answer here") == INVALID_ANS
        assert extract_answer("#### abc") == INVALID_ANS

    def test_parser_uses_first_match_and_scorer_uses_it(self):
        parsed = parse_gsm8k_answer("#### 1\n#### 2", "no_thinking_channel")
        assert parsed.value == "1"  # official extraction, NOT the last number
        assert parsed.status == "ok"
        item = {"item_id": "g1", "group_id": "g1", "split": "smoke", "task": "gsm8k",
                "gold": {"extracted_answer": "1"}}
        rec = score_gsm8k(item, parsed)
        assert rec.value == 1.0
        assert rec.status == "scored"

    def test_extraction_failure_is_invalid_format_not_a_crash(self):
        parsed = parse_gsm8k_answer("no #### marker", "no_thinking_channel")
        assert parsed.status == "invalid_format"
        assert parsed.value is None
        item = {"item_id": "g1", "group_id": "g1", "split": "smoke", "task": "gsm8k",
                "gold": {"extracted_answer": "42"}}
        rec = score_gsm8k(item, parsed)
        assert rec.value == 0.0
        assert rec.detail["extraction_failed"] is True
        assert rec.detail["correct"] is False


# ---------------------------------------------------------------------------
# ProcessBench: vendored extract_answer semantics (last boxed -> int, else None)
# ---------------------------------------------------------------------------
class TestProcessbenchOfficialParity:
    def test_boxed_pattern_matches_vendored_literal(self):
        assert 'boxed_pattern = r\'\\\\boxed\\{([^}]*)\\}\'' in (
            VENDOR / "processbench" / "run_eval.py").read_text(encoding="utf-8")
        assert BOXED_RE.pattern == r"\\boxed\{([^}]*)\}"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("x=3. Then \\boxed{5}", 5),
            ("\\boxed{2} and \\boxed{4}", 4),  # LAST match wins
            ("\\boxed{abc}", None),  # non-numeric
            ("\\boxed{1.5}", None),  # not an int
            ("no boxed here", None),
        ],
    )
    def test_official_extraction(self, text, expected):
        assert official_extract_index(text) == expected

    def test_extraction_applies_to_full_text(self):
        # the official evaluator extracts from the FULL generated text, even
        # when the boxed index sits inside the reasoning region
        parsed = parse_processbench_answer(
            "<think>\\boxed{2}</think>no boxed in the answer channel", "optional_thinking"
        )
        assert parsed.value == 2
        assert parsed.status == "ok"
        assert parsed.diagnostics["index_in_final_region"] is None

    @pytest.mark.parametrize(
        "label,pred_text,expected,kind",
        [
            (4, "\\boxed{4}", 1.0, "error_solution"),
            (4, "\\boxed{2}", 0.0, "error_solution"),
            (4, "\\boxed{abc}", 0.0, "error_solution"),
            (-1, "\\boxed{-1}", 1.0, "correct_solution"),
            (-1, "\\boxed{0}", 0.0, "correct_solution"),
            (-1, "no boxed", 0.0, "correct_solution"),
        ],
    )
    def test_value_is_one_iff_pred_equals_gold_label(self, label, pred_text, expected, kind):
        item = {"item_id": "p1", "group_id": "g1", "split": "smoke", "task": "processbench",
                "subset": "GSM8K", "gold": {"label": label}}
        parsed = parse_processbench_answer(pred_text, "no_thinking_channel")
        rec = score_processbench(item, parsed)
        assert rec.value == expected
        assert rec.detail["kind"] == kind
        assert rec.detail["match"] == (expected == 1.0)


# ---------------------------------------------------------------------------
# HotpotQA: vendored exact_match_score / f1_score parity
# ---------------------------------------------------------------------------
for _p in (str(VENDOR / "hotpotqa"), str(VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hotpotqa.hotpot_evaluate_v1 import exact_match_score, f1_score  # noqa: E402


def _hotpot_item(gold: str) -> Dict[str, Any]:
    return {"item_id": "h1", "group_id": "g1", "split": "smoke", "task": "hotpotqa",
            "gold": {"answer": gold}}


class TestHotpotqaParity:
    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("New York City", "new york city"),  # exact after normalization
            ("Paris, France", "Paris"),  # partial overlap
            ("yes", "no"),  # official yes/no zero rule
            ("The Grand Budapest Hotel", "grand budapest hotel"),  # article
            ("completely different", "answer"),
        ],
    )
    def test_value_and_em_equal_vendored(self, pred, gold):
        parsed = parse_hotpotqa_answer(pred, "no_thinking_channel")
        rec = score_hotpotqa(_hotpot_item(gold), parsed)
        assert rec.value == f1_score(pred, gold)[0]
        assert rec.detail["em"] == int(exact_match_score(pred, gold))
        assert rec.detail["f1"] == f1_score(pred, gold)[0]

    def test_empty_prediction_scores_zero(self):
        parsed = parse_hotpotqa_answer("", "no_thinking_channel")
        rec = score_hotpotqa(_hotpot_item("Paris"), parsed)
        assert rec.value == 0.0
        assert rec.status == "missing_final"
        assert rec.value == f1_score("", "Paris")[0]


# ---------------------------------------------------------------------------
# MATH: vendored is_equiv parity + missing-answer guards
# ---------------------------------------------------------------------------
_MATH_SPEC = importlib.util.spec_from_file_location(
    "vendored_math_equivalence_parity", VENDOR / "math" / "math_equivalence.py"
)
_VENDORED_MATH = importlib.util.module_from_spec(_MATH_SPEC)
_MATH_SPEC.loader.exec_module(_VENDORED_MATH)
vendored_is_equiv = _VENDORED_MATH.is_equiv


def _math_item(gold_boxed):
    return {"item_id": "m1", "group_id": "g1", "split": "smoke", "task": "math",
            "gold": {"boxed_answer": gold_boxed}}


class TestMathParity:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("1/2", "0.5", True),
            (r"\frac{a}{b}", "a/b", False),  # letters are not ints for a/b fixing
            ("x^2", "x^{2}", False),  # braces are NOT stripped by the heuristic
            (r"\frac{1}{2}", "1/2", True),
            (r"\frac{1}{2}", "0.5", True),
            (r"\sqrt{4}", "2", False),
            ("3", "3", True),
        ],
    )
    def test_is_equiv_equals_vendored(self, a, b, expected):
        assert weval_is_equiv(a, b) == vendored_is_equiv(a, b) == expected

    def test_end_to_end_equivalence_scoring(self):
        parsed = parse_math_answer(r"The answer is \boxed{1/2}", "no_thinking_channel")
        assert parsed.value == "1/2"
        rec = score_math(_math_item(r"\frac{1}{2}"), parsed)
        assert rec.value == 1.0
        assert rec.status == "scored"
        assert rec.detail["correct"] is True

    def test_missing_predicted_boxed_scores_zero_missing_final(self):
        # open reasoning region at EOS: extraction is missing
        parsed = parse_math_answer("<think>open", "prefilled_thinking")
        assert parsed.status == "missing_final"
        rec = score_math(_math_item("5"), parsed)
        assert rec.value == 0.0
        assert rec.status == "missing_final"
        assert rec.detail["reason"] == "no_boxed_in_answer"

    def test_missing_predicted_boxed_closed_channel_is_invalid_format(self):
        parsed = parse_math_answer("no boxed answer here", "no_thinking_channel")
        assert parsed.status == "invalid_format"
        rec = score_math(_math_item("5"), parsed)
        assert rec.value == 0.0
        assert rec.status == "invalid_format"

    @pytest.mark.parametrize("gold_boxed", [None, "", "   "])
    def test_missing_gold_boxed_is_unscorable(self, gold_boxed):
        parsed = parse_math_answer(r"\boxed{5}", "no_thinking_channel")
        rec = score_math(_math_item(gold_boxed), parsed)
        assert rec.status == "unscorable"
        assert rec.value is None
        assert rec.detail["reason"] == "gold_answer_has_no_boxed"

    def test_two_missing_answers_never_match(self):
        # predicted boxed missing AND gold boxed missing: never a 1.0 match
        parsed = parse_math_answer("no boxed", "no_thinking_channel")
        rec_both_missing = score_math(_math_item(None), parsed)
        assert rec_both_missing.value is None
        assert rec_both_missing.status == "unscorable"

        # gold present, prediction missing: 0.0, never a match
        rec_pred_missing = score_math(_math_item("5"), parsed)
        assert rec_pred_missing.value == 0.0
        assert rec_pred_missing.detail["correct"] is False

        # prediction present, gold missing: unscorable, never a match
        parsed_present = parse_math_answer(r"\boxed{5}", "no_thinking_channel")
        rec_gold_missing = score_math(_math_item(None), parsed_present)
        assert rec_gold_missing.value is None
