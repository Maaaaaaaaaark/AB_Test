"""Regression tests for failures that can silently invalidate comparisons."""

from __future__ import annotations

from weval.datasets.hotpotqa import _context_paragraphs
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
