"""Manifest resolution: stage sizes, per-parent counting, per-subset caps,
split disjointness, and the ProcessBench group-whole behavior.

Everything here drives the pure functions directly (no downloads, no torch);
manifests are hand-written JSON dicts in tmp dirs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from weval.config import ConfigBundle
from weval.datasets.musique import build_musique, split_parents
from weval.datasets.processbench import partition_records
from weval.run.manifests import (
    _manifest_paths,
    _select_items,
    manifest_items_for_task,
)
from tests.conftest import make_cfg, musique_record


def _musique_item(item_id: str, parent_id: str, kind: str = "positive") -> Dict[str, Any]:
    return {"item_id": item_id, "parent_id": parent_id, "kind": kind, "task": "musique_local"}


def _pb_item(item_id: str, subset: str, group_id: str) -> Dict[str, Any]:
    return {"item_id": item_id, "subset": subset, "group_id": group_id, "task": "processbench"}


def _cfg_with_stage(stage: str, **sizes: int) -> ConfigBundle:
    splits = {
        "stages": {stage: sizes},
        "seeds": {stage: [0]},
        "split_seeds": {},
    }
    return make_cfg(splits=splits)


# --------------------------------------------------------------------------
# _select_items per-stage sizes
# --------------------------------------------------------------------------

def test_musique_parents_size_counts_parents_not_items() -> None:
    """One parent contributes 2 positive subquestion items, the others 1 each.

    musique_parents=2 selects the parents of the first two POSITIVE items in
    manifest order, then returns ALL items of those parents (positives and
    negatives together) - the items of the first 2 parents, not 2 items."""
    cfg = _cfg_with_stage("smoke", musique_parents=2)
    items = [
        _musique_item("p1a", "P1"),                                  # parent 1: one positive
        _musique_item("p2a", "P2"), _musique_item("p2b", "P2"),      # parent 2: two positives
        _musique_item("p2n", "P2", kind="negative"),
        _musique_item("p3a", "P3"),
    ]
    chosen = _select_items(items, "smoke", "musique_local", cfg)
    assert [it["item_id"] for it in chosen] == ["p1a", "p2a", "p2b", "p2n"]
    assert {it["parent_id"] for it in chosen} == {"P1", "P2"}
    assert len(chosen) == 4  # whole parents, not a bare 2-item slice


def test_musique_parents_quota_counts_distinct_parents_in_order() -> None:
    """The quota counts the first k DISTINCT parents in manifest order (all of
    their items are kept), so a parent whose positive was human-excluded
    still occupies its slot through its negative and review decisions never
    shift which parents a stage covers."""
    cfg = _cfg_with_stage("smoke", musique_parents=2)
    items = [
        _musique_item("a1", "PA"), _musique_item("a2", "PA"),
        _musique_item("b1", "PB"), _musique_item("c1", "PC"),
    ]
    chosen = _select_items(items, "smoke", "musique_local", cfg)
    assert [it["item_id"] for it in chosen] == ["a1", "a2", "b1"]
    assert {it["parent_id"] for it in chosen} == {"PA", "PB"}


def test_processbench_per_subset_caps_each_subset_independently() -> None:
    cfg = _cfg_with_stage("smoke", processbench_per_subset=2)
    items = (
        [_pb_item(f"g{i}", "gsm8k", "G") for i in range(5)]
        + [_pb_item(f"m{i}", "math", "M") for i in range(3)]
        + [_pb_item(f"o{i}", "olympiadbench", "O") for i in range(2)]
    )
    chosen = _select_items(items, "smoke", "processbench", cfg)
    by_subset: Dict[str, List[str]] = {}
    for it in chosen:
        by_subset.setdefault(it["subset"], []).append(it["item_id"])
    assert by_subset == {
        "gsm8k": ["g0", "g1"],
        "math": ["m0", "m1"],
        "olympiadbench": ["o0", "o1"],
    }
    # gsm8k's overflow does not consume math's budget: each subset is capped
    # on its own.
    assert len(chosen) == 6


def test_reference_per_dataset_truncates_items() -> None:
    cfg = _cfg_with_stage("reference", reference_per_dataset=3)
    items = [{"item_id": f"r{i}"} for i in range(6)]
    chosen = _select_items(items, "reference", "gsm8k", cfg)
    assert [it["item_id"] for it in chosen] == ["r0", "r1", "r2"]
    assert len(chosen) == 3


# --------------------------------------------------------------------------
# ProcessBench group-whole behavior (the REAL behavior of _select_items)
# --------------------------------------------------------------------------

def test_processbench_select_keeps_groups_whole_when_cap_lands_on_boundary() -> None:
    """With contiguous groups and a cap that lands on a group boundary, no
    group is split."""
    cfg = _cfg_with_stage("smoke", processbench_per_subset=3)
    items = [
        _pb_item("g1a", "gsm8k", "G1"), _pb_item("g1b", "gsm8k", "G1"),
        _pb_item("g2a", "gsm8k", "G2"),
        _pb_item("g3a", "gsm8k", "G3"), _pb_item("g3b", "gsm8k", "G3"),
    ]
    chosen = _select_items(items, "smoke", "processbench", cfg)
    # cap 3 = G1 (2 items) + G2 (1 item): groups stay whole
    assert [it["item_id"] for it in chosen] == ["g1a", "g1b", "g2a"]
    assert {it["group_id"] for it in chosen} == {"G1", "G2"}
    assert len([it for it in chosen if it["group_id"] == "G1"]) == 2  # complete


def test_processbench_smoke_cuts_at_the_cap_even_mid_group() -> None:
    """Smoke is the documented exception: its flat per-subset cap may cut a
    group mid-group. Every other stage rounds up to whole groups (next test).
    Smoke outputs never support qualification claims."""
    cfg = _cfg_with_stage("smoke", processbench_per_subset=3)
    items = [
        _pb_item("g1a", "gsm8k", "G1"), _pb_item("g1b", "gsm8k", "G1"),
        _pb_item("g2a", "gsm8k", "G2"), _pb_item("g2b", "gsm8k", "G2"),
    ]
    chosen = _select_items(items, "smoke", "processbench", cfg)
    assert [it["item_id"] for it in chosen] == ["g1a", "g1b", "g2a"]
    assert len(chosen) == 3  # exactly the cap, G2 cut mid-group
    assert {it["group_id"] for it in chosen} == {"G1", "G2"}
    # the cap applies per subset: other subsets are unaffected
    assert all(it["subset"] == "gsm8k" for it in chosen)


def test_processbench_select_rounds_up_to_whole_groups() -> None:
    """Non-smoke stages never split a group: the group crossing the per-subset
    cap is included whole, matching the build-time whole-group partition."""
    cfg = _cfg_with_stage("pilot", processbench_per_subset=3)
    items = [
        _pb_item("g1a", "gsm8k", "G1"), _pb_item("g1b", "gsm8k", "G1"),
        _pb_item("g2a", "gsm8k", "G2"), _pb_item("g2b", "gsm8k", "G2"),
        _pb_item("g3a", "gsm8k", "G3"), _pb_item("g3b", "gsm8k", "G3"),
        _pb_item("m1a", "math", "M1"), _pb_item("m1b", "math", "M1"),
    ]
    chosen = _select_items(items, "pilot", "processbench", cfg)
    # gsm8k: G1 (2) + crossing group G2 (4 >= cap 3) included whole; G3 out.
    # math: M1 (2 < cap 3) included whole.
    assert [it["item_id"] for it in chosen] == ["g1a", "g1b", "g2a", "g2b", "m1a", "m1b"]
    assert {it["group_id"] for it in chosen if it["subset"] == "gsm8k"} == {"G1", "G2"}


def test_partition_records_keeps_whole_groups_and_splits_are_disjoint() -> None:
    """Build-time partition: groups of records sharing a problem always move
    together, and debug/selection/holdout are disjoint by group."""
    records: List[Dict[str, Any]] = []
    for g in range(10):
        problem = f"unique problem number {g}"
        records.append({"problem": problem, "steps": ["s1"], "label": -1, "final_answer_correct": None})
        records.append({"problem": problem, "steps": ["s2"], "label": 1, "final_answer_correct": None})
    quotas = {"debug": 3, "selection": 5, "holdout": 3}
    parts, info = partition_records(records, quotas, order_seed=12345)

    # every record of a group lands in exactly one part
    group_parts: Dict[str, set] = {}
    for part in ("debug", "selection", "holdout"):
        for r in parts[part]:
            group_parts.setdefault(r["problem"], set()).add(part)
    assert all(len(v) == 1 for v in group_parts.values())

    # parts are disjoint by group and take whole groups only
    sets = {p: {r["problem"] for r in parts[p]} for p in ("debug", "selection", "holdout")}
    assert sets["debug"] & sets["selection"] == set()
    assert sets["debug"] & sets["holdout"] == set()
    assert sets["selection"] & sets["holdout"] == set()
    for part in ("debug", "selection", "holdout"):
        assert len(parts[part]) % 2 == 0  # both records of each group present
        for r in parts[part]:
            assert sum(1 for r2 in parts[part] if r2["problem"] == r["problem"]) == 2

    # quotas are met by whole groups and exceeded only by one partial group
    for part, quota in (("debug", 3), ("selection", 5), ("holdout", 3)):
        assert len(parts[part]) >= quota
        assert len(parts[part]) < quota + 2
    assert len(info["group_order"]) == 10


# --------------------------------------------------------------------------
# MuSiQue split disjointness (pure builder path, no downloads)
# --------------------------------------------------------------------------

def test_musique_selection_and_holdout_do_not_share_parents() -> None:
    dev = [musique_record(pid=f"2hop__{i}_{i + 1}") for i in range(30)]
    train = [musique_record(pid=f"3hop__{i}_{i + 1}_{i + 2}") for i in range(5)]
    cfg = make_cfg()

    pools = split_parents({"dev": dev, "train": train}, cfg)
    assert set(pools["debug"]) == {r["id"] for r in train}
    sel, hold = pools["selection"], pools["holdout"]
    assert len(sel) == 6 and len(hold) == 6
    assert set(sel) | set(hold) <= {r["id"] for r in dev}
    assert set(sel) & set(hold) == set()
    assert set(sel) & set(pools["debug"]) == set()

    # Build the actual item sets from each pool (pure; deterministic) and
    # check no parent id appears in two splits.
    seed = cfg.splits["split_seeds"]["musique_subquestion_choice"]
    by_id = {r["id"]: r for r in dev}
    sel_items, _, _ = build_musique(
        [by_id[p] for p in sel], "selection", seed, parent_ids=sel
    )
    hold_items, _, _ = build_musique(
        [by_id[p] for p in hold], "holdout", seed, parent_ids=hold
    )
    sel_parents = {i.parent_id for i in sel_items}
    hold_parents = {i.parent_id for i in hold_items}
    assert sel_parents == set(sel)
    assert hold_parents == set(hold)
    assert sel_parents & hold_parents == set()


# --------------------------------------------------------------------------
# manifest paths + end-to-end manifest reading
# --------------------------------------------------------------------------

def test_manifest_paths_map_stage_to_files(tmp_path: Path) -> None:
    mdir = tmp_path / "manifests"
    mdir.mkdir(parents=True)
    # _manifest_paths validates existence, so touch every file it may name.
    for name in (
        "musique_debug.json", "musique_selection.json", "musique_holdout.json",
        "processbench_gsm8k_debug.json", "processbench_gsm8k_selection.json",
        "processbench_math_selection.json", "processbench_olympiadbench_selection.json",
        "processbench_omnimath_selection.json",
        "gsm8k_reference.json", "gsm8k_reference_pilot.json",
    ):
        (mdir / name).write_text("{}", encoding="utf-8")

    assert _manifest_paths(tmp_path, "musique_local", "smoke") == [mdir / "musique_debug.json"]
    assert _manifest_paths(tmp_path, "musique_local", "pilot") == [mdir / "musique_debug.json"]
    assert _manifest_paths(tmp_path, "musique_local", "selection") == [mdir / "musique_selection.json"]
    assert _manifest_paths(tmp_path, "musique_local", "holdout") == [mdir / "musique_holdout.json"]

    pb = _manifest_paths(tmp_path, "processbench", "selection")
    assert [p.name for p in pb] == [
        "processbench_gsm8k_selection.json",
        "processbench_math_selection.json",
        "processbench_olympiadbench_selection.json",
        "processbench_omnimath_selection.json",
    ]
    assert _manifest_paths(tmp_path, "gsm8k", "selection") == [mdir / "gsm8k_reference.json"]
    assert _manifest_paths(tmp_path, "gsm8k", "pilot") == [mdir / "gsm8k_reference_pilot.json"]


def test_manifest_items_for_task_reads_manifest_and_applies_sizes(tmp_path: Path) -> None:
    items = [
        _musique_item("a1", "PA"),
        _musique_item("a1n", "PA", kind="negative"),
        _musique_item("b1", "PB"),
        _musique_item("c1", "PC"),
    ]
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "manifests" / "musique_debug.json").write_text(
        json.dumps({"items": items}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cfg = _cfg_with_stage("smoke", musique_parents=2)
    chosen = manifest_items_for_task(cfg, tmp_path, "smoke", "musique_local")
    assert [it["item_id"] for it in chosen] == ["a1", "a1n", "b1"]


def test_processbench_manifests_merge_across_subset_files(tmp_path: Path) -> None:
    mdir = tmp_path / "manifests"
    mdir.mkdir(parents=True)
    for subset, ids in (
        ("gsm8k", ["g0", "g1", "g2"]),
        ("math", ["m0", "m1", "m2"]),
        ("olympiadbench", []),
        ("omnimath", []),
    ):
        (mdir / f"processbench_{subset}_debug.json").write_text(
            json.dumps({"items": [_pb_item(i, subset, f"grp-{i}") for i in ids]}),
            encoding="utf-8",
        )
    cfg = _cfg_with_stage("smoke", processbench_per_subset=2)
    chosen = manifest_items_for_task(cfg, tmp_path, "smoke", "processbench")
    # all four subset manifest files are read and merged, then capped per subset
    assert [it["item_id"] for it in chosen] == ["g0", "g1", "m0", "m1"]


def test_manifest_items_for_task_raises_when_any_manifest_missing(tmp_path: Path) -> None:
    cfg = _cfg_with_stage("smoke", musique_parents=2)
    with pytest.raises(FileNotFoundError):
        manifest_items_for_task(cfg, tmp_path, "smoke", "musique_local")
