"""Compute-cost metrics: Cproxy / n_text / prices / efficiency.

Synthetic safetensors checkpoints are written with the numpy writer (no torch);
n_text is derived from safetensors HEADERS only. The record shape mirrors
weval/metrics/report.py: Tin/Tout come from the prediction record fields
`n_prompt_tokens` / `n_generated_tokens`, and Cproxy = 2 * n_text * (Tin+Tout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from safetensors.numpy import save_file

from weval.metrics.compute import (
    COMPUTE_VERSION,
    PriceSource,
    aggregate_cproxy,
    cproxy_per_item,
    cweighted,
    efficiency_compute,
    load_prices,
)
from weval.models.params import count_text_parameters


@pytest.fixture
def torchless_safetensors(monkeypatch):
    """Header access via the 'np' framework instead of 'pt'.

    safetensors 0.8's rust safe_open(framework="pt") imports torch eagerly at
    open time; count_text_parameters only reads headers (keys and shapes,
    never tensor data), so the identical 'np' header path is substituted to
    keep the tests torch-free. params.py does `from safetensors import
    safe_open` inside the function, so patching the module attribute works.
    """
    import safetensors as st

    real = st.safe_open

    def np_safe_open(filename, framework="pt", device="cpu"):
        return real(filename, framework="np", device=device)

    monkeypatch.setattr(st, "safe_open", np_safe_open)


def _write_snapshot(
    snapshot_dir: Path,
    tensors: Dict[str, Tuple[int, ...]],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Single-file checkpoint: model.safetensors (+ optional config.json)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: np.ones(shape, dtype=np.float32) for k, shape in tensors.items()},
        str(snapshot_dir / "model.safetensors"),
    )
    if config is not None:
        (snapshot_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot_dir


def _cproxy_from_record(n_text: int, record: Dict[str, Any]) -> float:
    """The per-item formula report.py applies to prediction records."""
    return cproxy_per_item(n_text, record["n_prompt_tokens"], record["n_generated_tokens"])


class TestNTextFromSafetensors:
    def test_plain_weights_counted_once_each(self, torchless_safetensors, tmp_path):
        snap = _write_snapshot(
            tmp_path / "snap",
            {"w1": (2, 3), "w2": (2, 2), "w3": (5,)},
            config={"tie_word_embeddings": True},
        )
        audit = count_text_parameters(snap)
        assert audit.n_text == 6 + 4 + 5
        assert audit.n_total_in_files == 15
        assert audit.n_excluded == 0
        assert audit.tie_word_embeddings is True
        # w1/w2/w3 are not embedding keys; the tie rule only concerns the
        # embedding / LM-head pair
        assert audit.tied_dedup_applied is False
        assert audit.per_file["model.safetensors"]["numel_counted"] == 15

    def test_tie_true_embed_and_lm_head_counted_once(self, torchless_safetensors, tmp_path):
        """tie_word_embeddings=true with equal embed/LM-head shapes: the pair
        is the same underlying parameter and counts ONCE (transformers'
        get_num_parameters convention)."""
        snap = _write_snapshot(
            tmp_path / "snap",
            {
                "model.embed_tokens.weight": (2, 3),
                "lm_head.weight": (2, 3),
                "model.norm.weight": (3,),
            },
            config={"tie_word_embeddings": True},
        )
        audit = count_text_parameters(snap)
        assert audit.n_text == 9  # 6 (tied pair, once) + 3
        assert audit.n_total_in_files == 15  # raw file contents
        assert audit.tied_dedup_applied is True
        assert audit.tied_keys == ["model.embed_tokens.weight", "lm_head.weight"]
        assert audit.tie_word_embeddings is True

    def test_tie_true_shape_mismatch_counts_both(self, torchless_safetensors, tmp_path):
        """tie=true but embed/LM-head shapes differ: the checkpoint is
        actually untied despite the flag, so both tensors are counted."""
        snap = _write_snapshot(
            tmp_path / "snap",
            {
                "model.embed_tokens.weight": (2, 3),
                "lm_head.weight": (6,),
                "model.norm.weight": (3,),
            },
            config={"tie_word_embeddings": True},
        )
        audit = count_text_parameters(snap)
        assert audit.n_text == 15
        assert audit.tied_dedup_applied is False
        assert audit.tie_word_embeddings is True

    def test_tie_false_counts_embed_and_head_separately(self, torchless_safetensors, tmp_path):
        snap = _write_snapshot(
            tmp_path / "snap",
            {
                "model.embed_tokens.weight": (2, 3),
                "lm_head.weight": (2, 3),
                "model.norm.weight": (3,),
            },
            config={"tie_word_embeddings": False},
        )
        audit = count_text_parameters(snap)
        assert audit.n_text == 15
        assert audit.tie_word_embeddings is False
        assert audit.tied_dedup_applied is False

    def test_no_tie_info_in_config(self, torchless_safetensors, tmp_path):
        snap = _write_snapshot(
            tmp_path / "snap",
            {"model.embed_tokens.weight": (2, 3), "lm_head.weight": (2, 3)},
        )
        audit = count_text_parameters(snap)
        assert audit.tie_word_embeddings is None
        assert audit.tied_dedup_applied is False
        assert audit.n_text == 12

    def test_repeated_embed_key_name_across_shards_deduped(self, torchless_safetensors, tmp_path):
        """The dedup branch the implementation actually covers: the SAME embed
        key name appearing in two shard files is counted only once."""
        snap = tmp_path / "snap"
        snap.mkdir()
        save_file({"model.embed_tokens.weight": np.ones((2, 3), dtype=np.float32)}, str(snap / "s1.safetensors"))
        save_file(
            {
                "model.embed_tokens.weight": np.ones((2, 3), dtype=np.float32),
                "model.norm.weight": np.ones((3,), dtype=np.float32),
            },
            str(snap / "s2.safetensors"),
        )
        (snap / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "s1.safetensors",
                        "model.norm.weight": "s2.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        (snap / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")
        audit = count_text_parameters(snap, detail=True)
        assert audit.n_text == 9  # 6 (embed, once) + 3 (norm)
        assert audit.tied_dedup_applied is True
        assert audit.n_total_in_files == 15
        assert audit.per_file["s1.safetensors"]["numel_counted"] == 6
        assert audit.per_file["s2.safetensors"]["numel_counted"] == 3

    def test_multi_shard_checkpoint_summed(self, torchless_safetensors, tmp_path):
        snap = tmp_path / "snap"
        snap.mkdir()
        save_file(
            {
                "a": np.ones((2, 2), dtype=np.float32),
                "b": np.ones((3,), dtype=np.float32),
            },
            str(snap / "model-00001-of-00002.safetensors"),
        )
        save_file({"c": np.ones((2, 3), dtype=np.float32)}, str(snap / "model-00002-of-00002.safetensors"))
        (snap / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        (snap / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")
        audit = count_text_parameters(snap)
        assert audit.n_text == 4 + 3 + 6
        assert set(audit.per_file) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
        assert audit.per_file["model-00001-of-00002.safetensors"]["numel_counted"] == 7
        assert audit.per_file["model-00002-of-00002.safetensors"]["numel_counted"] == 6

    def test_single_file_preferred_over_index(self, torchless_safetensors, tmp_path):
        snap = tmp_path / "snap"
        snap.mkdir()
        save_file({"a": np.ones((2,), dtype=np.float32)}, str(snap / "model.safetensors"))
        save_file({"huge": np.ones((1000,), dtype=np.float32)}, str(snap / "model-00001-of-00002.safetensors"))
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"huge": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        audit = count_text_parameters(snap)
        assert audit.n_text == 2  # the single file, not the indexed shard

    def test_gemma_vision_tensors_excluded(self, torchless_safetensors, tmp_path):
        snap = _write_snapshot(
            tmp_path / "snap",
            {
                "vision_tower.encoder.layer0.weight": (10,),
                "vision_tower.decoder.weight": (10,),
                "multi_modal_projector.linear.weight": (10,),
                "model.embed_tokens.weight": (2, 3),
                "lm_head.weight": (2, 3),
            },
            config={"tie_word_embeddings": True},
        )
        audit = count_text_parameters(
            snap,
            param_exclusions=["vision_tower.*", "multi_modal_projector.*"],
            detail=True,
        )
        assert audit.n_excluded == 3
        assert sorted(audit.excluded_keys) == [
            "multi_modal_projector.linear.weight",
            "vision_tower.decoder.weight",
            "vision_tower.encoder.layer0.weight",
        ]
        # excluded tensors never enter the text count or the file total
        assert audit.n_total_in_files == 12  # 6 (embed) + 6 (lm_head)
        # tied pair (equal shapes) counts once: 6, not 12
        assert audit.n_text == 6
        assert audit.tied_dedup_applied is True
        assert audit.tied_keys == ["model.embed_tokens.weight", "lm_head.weight"]

    def test_missing_shard_file_raises(self, torchless_safetensors, tmp_path):
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "missing-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError):
            count_text_parameters(snap)

    def test_no_checkpoint_files_raises(self, torchless_safetensors, tmp_path):
        with pytest.raises(FileNotFoundError):
            count_text_parameters(tmp_path / "empty")


class TestCproxyFormula:
    def test_cproxy_per_item_formula(self):
        n_text, tin, tout = 7, 5, 3
        assert cproxy_per_item(n_text, tin, tout) == 2 * n_text * (tin + tout)
        assert cproxy_per_item(7, 5, 3) == 112

    def test_record_fields_feed_tin_and_tout(self):
        """The prediction-record fields compute.py's pipeline reads: Tin =
        n_prompt_tokens, Tout = n_generated_tokens (see report.py)."""
        n_text = 10
        records = [
            {"item_id": "i1", "kind": "positive", "n_prompt_tokens": 5, "n_generated_tokens": 3},
            {"item_id": "i2", "kind": "positive", "n_prompt_tokens": 2, "n_generated_tokens": 1},
            {"item_id": "i3", "kind": "negative", "n_prompt_tokens": 4, "n_generated_tokens": 2},
            {"item_id": "i4", "kind": "negative", "n_prompt_tokens": 1, "n_generated_tokens": 1},
        ]
        per_item = [_cproxy_from_record(n_text, r) for r in records]
        assert per_item == [2 * n_text * 8, 2 * n_text * 3, 2 * n_text * 6, 2 * n_text * 2]
        assert per_item == [160.0, 60.0, 120.0, 40.0]


class TestAggregateCproxy:
    def test_sum_mean_and_by_kind(self):
        per_item = [
            {"kind": "positive", "cproxy": 160.0},
            {"kind": "positive", "cproxy": 60.0},
            {"kind": "negative", "cproxy": 120.0},
            {"kind": "negative", "cproxy": 40.0},
        ]
        agg = aggregate_cproxy(per_item)
        assert agg["n_items"] == 4
        assert agg["n_with_cproxy"] == 4
        assert agg["sum"] == 380.0
        assert agg["mean"] == 95.0
        assert agg["by_kind"]["positive"] == {"sum": 220.0, "n": 2, "mean": 110.0}
        assert agg["by_kind"]["negative"] == {"sum": 160.0, "n": 2, "mean": 80.0}

    def test_missing_cproxy_is_skipped_not_zeroed(self):
        per_item = [
            {"kind": "positive", "cproxy": 100.0},
            {"kind": "positive", "cproxy": None},
            {"kind": "negative", "cproxy": 50.0},
        ]
        agg = aggregate_cproxy(per_item)
        assert agg["n_items"] == 3
        assert agg["n_with_cproxy"] == 2
        assert agg["sum"] == 150.0
        assert agg["mean"] == 75.0
        assert agg["by_kind"]["positive"] == {"sum": 100.0, "n": 1, "mean": 100.0}
        assert agg["by_kind"]["negative"] == {"sum": 50.0, "n": 1, "mean": 50.0}

    def test_all_missing_cproxy(self):
        agg = aggregate_cproxy([{"kind": "positive", "cproxy": None}])
        assert agg["n_items"] == 1
        assert agg["n_with_cproxy"] == 0
        assert agg["sum"] == 0.0
        assert agg["mean"] is None
        assert agg["by_kind"] == {}

    def test_kind_defaults_to_na(self):
        agg = aggregate_cproxy([{"cproxy": 10.0}])
        assert agg["by_kind"]["na"] == {"sum": 10.0, "n": 1, "mean": 10.0}


class TestCweighted:
    def test_null_when_prices_absent(self, tmp_path):
        assert cweighted(123, 456, None) is None
        assert load_prices(tmp_path) is None  # no configs/prices.yaml

    def test_formula_from_price_source(self):
        price = PriceSource(
            usd_per_1m_input=1.5,
            usd_per_1m_output=0.5,
            source_url="https://example.com/prices",
            retrieved_at="2026-10-13",
        )
        assert cweighted(1_000_000, 2_000_000, price) == 2.5  # 1.5 + 1.0

    def test_load_prices_parses_yaml(self, tmp_path):
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "prices.yaml").write_text(
            "\n".join(
                [
                    "usd_per_1m_input: 1.25",
                    "usd_per_1m_output: 0.75",
                    'source_url: "https://example.com/prices"',
                    'retrieved_at: "2026-10-13"',
                ]
            ),
            encoding="utf-8",
        )
        price = load_prices(tmp_path)
        assert price is not None
        assert price.usd_per_1m_input == 1.25
        assert price.usd_per_1m_output == 0.75
        assert price.source_url == "https://example.com/prices"
        assert price.retrieved_at == "2026-10-13"

    def test_load_prices_missing_fields_refuses_to_guess(self, tmp_path):
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "prices.yaml").write_text(
            "usd_per_1m_input: 1.25\nusd_per_1m_output: 0.75\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_prices(tmp_path)


class TestEfficiencyCompute:
    def test_formula_mean_score_over_mean_cproxy(self):
        scores = [0.5, 1.0]
        cproxies = [1e12, 3e12]
        expected = 0.75 / 2.0  # mean(score) / mean(cproxy/1e12)
        assert efficiency_compute(scores, cproxies) == expected == 0.375

    def test_length_mismatch_returns_none(self):
        assert efficiency_compute([0.5, 1.0], [1e12]) is None

    def test_empty_inputs_return_none(self):
        assert efficiency_compute([], []) is None

    def test_zero_mean_cproxy_returns_none(self):
        assert efficiency_compute([0.5], [0.0]) is None

    def test_same_task_alignment_mirrors_report_pipeline(self):
        """report.py computes efficiency only within one task: it aligns
        scored records (by item_id) with their cproxy items and drops
        unscored records before calling efficiency_compute."""
        n_text = 10
        records = [
            {"item_id": "a", "kind": "positive", "n_prompt_tokens": 5, "n_generated_tokens": 3},
            {"item_id": "b", "kind": "positive", "n_prompt_tokens": 2, "n_generated_tokens": 1},
            {"item_id": "c", "kind": "positive", "n_prompt_tokens": 4, "n_generated_tokens": 2},
        ]
        scored = {"a": 0.5, "c": 1.0}  # b failed to score -> excluded
        c_items = [
            {"item_id": r["item_id"], "kind": r["kind"], "cproxy": _cproxy_from_record(n_text, r)}
            for r in records
        ]
        eff_scores: List[float] = []
        eff_cs: List[float] = []
        for r, ci in zip(records, c_items):
            s = scored.get(r["item_id"])
            if s is not None and ci["cproxy"] is not None:
                eff_scores.append(float(s))
                eff_cs.append(ci["cproxy"])
        assert efficiency_compute(eff_scores, eff_cs) == efficiency_compute(
            [0.5, 1.0], [160.0, 120.0]
        )
        assert len(eff_scores) == len(eff_cs) == 2
