"""Regression tests for failures that can silently invalidate comparisons."""

from __future__ import annotations

from weval.datasets.hotpotqa import _context_paragraphs
from weval.metrics.report import _aggregate_musique
from weval.models.scripted import ScriptedBackend
from weval.parsing.musique_answer import parse_musique_answer
from weval.scoring.musique_em_f1 import score_musique
from weval.metrics.stats import macro_group_bootstrap, paired_difference
from weval.run.store import record_fingerprints


def test_hotpot_context_struct_of_lists_is_not_split_into_key_characters() -> None:
    context = {
        "title": ["Doc A", "Doc B"],
        "sentences": [["A one.", "A two."], ["B one."]],
    }
    assert _context_paragraphs(context) == [
        {"title": "Doc A", "text": "A one. A two."},
        {"title": "Doc B", "text": "B one."},
    ]


def test_generation_budget_changes_cache_fingerprint() -> None:
    common = dict(
        item={"item_id": "x"}, rendered_text="prompt", model_revision="rev",
        template_sha256="abc", parser_version="p", scorer_version="s",
    )
    a = record_fingerprints(
        **common, generation_settings={"max_new_tokens": 128, "temperature": 0.6}
    )
    b = record_fingerprints(
        **common, generation_settings={"max_new_tokens": 256, "temperature": 0.6}
    )
    assert a["generation_settings"] != b["generation_settings"]


def test_macro_bootstrap_preserves_repeated_group_draws() -> None:
    # The point of this regression is that the implementation accepts and
    # resamples unequal groups without collapsing duplicate draws.  A fixed
    # seed makes the resulting interval deterministic.
    values = {
        "s": {
            "large": [1.0, 1.0, 1.0],
            "small": [0.0],
        }
    }
    ci = macro_group_bootstrap(values, n_boot=100, seed=7)
    assert ci.lo is not None and ci.hi is not None
    assert ci.lo < ci.hi


def test_paired_difference_aligns_exact_observation_keys() -> None:
    a = {"g": {"i1::seed=0": 1.0, "i2::seed=0": 0.0}}
    b = {"g": {"i2::seed=0": 1.0}}
    ci, missing_a, missing_b = paired_difference(a, b, n_boot=10)
    assert ci.estimate == -1.0
    assert missing_a == 0
    assert missing_b == 1


def test_false_abstention_stays_in_answerable_f1_population() -> None:
    item = {
        "item_id": "positive",
        "group_id": "parent",
        "task": "musique_local",
        "split": "smoke",
        "kind": "positive",
        "reference_answers": ["Paris"],
    }
    parsed = parse_musique_answer("NOT FOUND", "no_thinking_channel")
    scored = score_musique(item, parsed)
    assert scored.value == 0.0
    assert scored.detail["false_abstention"] is True
    assert scored.detail["answerable"] is True

    aggregate = _aggregate_musique(
        None,  # cfg is intentionally unused by the aggregator
        [
            {
                "item_id": item["item_id"],
                "group_id": item["group_id"],
                "value": scored.value,
                "detail": scored.detail,
            }
        ],
        {item["item_id"]: item},
        n_boot=10,
    )
    assert aggregate["answerable_f1"]["estimate"] == 0.0
    assert aggregate["relation_vs_natural"]["natural"]["n"] == 1


def test_scripted_default_closes_prefilled_reasoning_channel() -> None:
    backend = ScriptedBackend(
        default_text="NOT FOUND", reasoning_mode="prefilled_thinking"
    )
    result = backend.generate("x", 0, "prompt", 16, 1.0)
    assert result.output_text == "</think>\nNOT FOUND"
    parsed = parse_musique_answer(result.output_text, "prefilled_thinking")
    assert parsed.diagnostics["is_not_found"] is True
