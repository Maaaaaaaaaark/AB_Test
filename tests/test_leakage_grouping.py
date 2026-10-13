"""Cross-dataset leakage detection and whole-group split construction.

Covers the pure helpers behind the overlap controls: the shared
normalization (digits/operators preserved, no invented token boundaries),
Jaccard near-duplicate flagging at >= 0.95, exact-match detection between
ProcessBench and the reference datasets (restricted to local-used splits and
the gsm8k/math subsets), ProcessBench whole-problem-group partitioning with
reported per-split error ratios, disjoint parent-id pools for MuSiQue, and
the reference-subset exclusion of problems used in local debug/selection.

IMPORTANT NOTE (module bug, not fixed here per instructions): the module
docstring of weval/datasets/normalization.py claims "x+2=5" and "x + 2 = 5"
match, but the implementation preserves collapsed whitespace, so spaced
variants do NOT match under normalize_for_matching. The tests below pin the
ACTUAL behavior; the docstring/implementation contradiction is reported.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from weval.datasets.build import _frozen_reference, _normalized_q
from weval.datasets.leakage import detect_overlap
from weval.datasets.musique import split_parents
from weval.datasets.normalization import (
    NEAR_DUP_JACCARD,
    jaccard,
    normalize_for_matching,
    token_set,
)
from weval.datasets.processbench import (
    build_items,
    partition_records,
    problem_group_key,
    processbench_quotas,
)
from tests.conftest import make_cfg, musique_record


class TestNormalization:
    def test_digits_and_operators_preserved(self):
        # Digits and operators survive exactly as written.
        assert normalize_for_matching("a+b=3") == "a+b=3"
        assert normalize_for_matching("3+5") == "3+5"
        assert normalize_for_matching("x^2 + y/3") == "x^2+y/3"
        # ...and remain discriminating: a different digit never matches.
        assert normalize_for_matching("x+2=5") != normalize_for_matching("x+3=5")

    def test_no_invented_token_boundaries(self):
        # "x2" vs "x 2" differ: normalization must not insert or merge spaces
        # between alphanumeric tokens.
        assert normalize_for_matching("x2") != normalize_for_matching("x 2")

    def test_operator_adjacent_whitespace_dropped(self):
        # Whitespace adjacent to a math operator is insignificant (the module
        # docstring's guarantee): "x+2=5" and "x + 2 = 5" match.
        assert normalize_for_matching("a   b\tc") == "a b c"
        assert normalize_for_matching("a + b = 3") == "a+b=3"
        assert normalize_for_matching("a + b = 3") == normalize_for_matching("a+b=3")
        assert normalize_for_matching("3 + 5") == "3+5"
        # whitespace NOT adjacent to an operator is kept (collapsed)
        assert normalize_for_matching("x 2") != normalize_for_matching("x2")

    def test_case_and_punctuation(self):
        assert normalize_for_matching("What's  A+B?") == "what's a+b"
        assert normalize_for_matching("JANET has ducks.") == "janet has ducks."
        assert normalize_for_matching("!@#$&?") == ""


class TestJaccard:
    def test_known_overlaps(self):
        assert jaccard("a b c", "a b c") == 1.0
        assert jaccard("a b c d", "a b c") == 0.75
        assert jaccard("a b", "c d") == 0.0

    def test_empty_token_sets(self):
        assert jaccard("", "") == 1.0
        assert jaccard("x", "") == 0.0
        assert jaccard("", "x") == 0.0

    def test_token_set_deduplicates_and_normalizes(self):
        assert jaccard("a a b", "a b") == 1.0
        assert token_set("A! B?") == {"a", "b"}

    def test_near_dup_threshold_behavior(self):
        base = " ".join(f"tok{i}" for i in range(20))
        flagged = jaccard(base, base + " extra")
        assert flagged == 20 / 21
        assert flagged >= NEAR_DUP_JACCARD  # review flag
        below = jaccard(base, " ".join(f"tok{i}" for i in range(19)) + " other")
        assert below == 19 / 21
        assert below < NEAR_DUP_JACCARD


class TestDetectOverlap:
    def test_exact_normalized_match_across_datasets(self):
        pb = {"gsm8k": [{"item_id": "pb-1", "problem": "JANET has 3 + 5 ducks.", "split": "selection"}]}
        gsm = [{"id": "g-1", "question": "Janet has 3 + 5 ducks."}]
        rep = detect_overlap(pb, gsm, None)
        assert rep.counts == {"exact_matches": 1, "near_duplicates": 0}
        assert rep.exact_matches[0] == {
            "processbench_item": "pb-1",
            "processbench_subset": "gsm8k",
            "processbench_split": "selection",
            "reference_dataset": "gsm8k",
            "reference_id": "g-1",
        }
        assert rep.near_duplicates == []

    def test_math_target_uses_problem_field(self):
        pb = {"math": [{"item_id": "pb-m", "problem": "Solve x + 2 = 5.", "split": "debug"}]}
        math_recs = [{"id": "m-1", "problem": "Solve x + 2 = 5."}]
        rep = detect_overlap(pb, None, math_recs)
        assert rep.counts["exact_matches"] == 1
        assert rep.exact_matches[0]["reference_dataset"] == "math"
        assert rep.exact_matches[0]["reference_id"] == "m-1"

    def test_near_duplicate_flagged_not_excluded(self):
        base = " ".join(f"tok{i}" for i in range(20))
        pb = {"math": [{"item_id": "pb-m", "problem": base, "split": "selection"}]}
        math_recs = [{"id": "m-1", "problem": base + " extra"}]
        rep = detect_overlap(pb, None, math_recs)
        assert rep.counts == {"exact_matches": 0, "near_duplicates": 1}
        nd = rep.near_duplicates[0]
        assert nd["jaccard"] == round(20 / 21, 4)
        assert nd["note"] == "review required; not auto-excluded"
        assert nd["processbench_item"] == "pb-m"
        assert nd["reference_id"] == "m-1"
        assert rep.exact_matches == []

    def test_only_used_local_splits_are_checked(self):
        pb = {"gsm8k": [{"item_id": "h1", "problem": "Janet has 3 + 5 ducks.", "split": "holdout"}]}
        gsm = [{"id": "g-1", "question": "Janet has 3 + 5 ducks."}]
        # Default used-set (debug/selection) does not include holdout items.
        rep = detect_overlap(pb, gsm, None)
        assert rep.counts == {"exact_matches": 0, "near_duplicates": 0}
        rep2 = detect_overlap(pb, gsm, None, local_splits_used=("holdout",))
        assert rep2.counts["exact_matches"] == 1

    def test_only_gsm8k_and_math_subsets_scanned(self):
        pb = {"olympiadbench": [{"item_id": "o1", "problem": "Janet has 3 + 5 ducks.", "split": "selection"}]}
        gsm = [{"id": "g-1", "question": "Janet has 3 + 5 ducks."}]
        rep = detect_overlap(pb, gsm, None)
        assert rep.counts == {"exact_matches": 0, "near_duplicates": 0}


def _pb(pid: str, problem: str, label: int = -1, fac: bool = True) -> Dict[str, Any]:
    return {
        "id": pid,
        "problem": problem,
        "steps": [f"{pid}-s1", f"{pid}-s2"],
        "label": label,
        "final_answer_correct": fac,
    }


class TestProcessBenchPartition:
    QUOTAS = {"debug": 1, "selection": 2, "holdout": 2}

    def _records(self) -> List[Dict[str, Any]]:
        return [
            _pb("a1", "Alpha + Beta = 3", -1),
            _pb("a2", "Alpha + Beta = 3", 4),  # same normalized problem as a1
            _pb("b1", "Bravo problem", -1),
            _pb("b2", "Bravo problem", 2),
            _pb("b3", "Bravo problem", 5),
            _pb("c1", "Charlie problem", 7),
        ]

    def _expected_order(self, seed: int) -> List[str]:
        """The documented mechanism: first-occurrence group order, then a
        deterministic seeded shuffle."""
        order: List[str] = []
        for r in self._records():
            k = problem_group_key(r["problem"])
            if k not in order:
                order.append(k)
        rng = random.Random(seed)
        rng.shuffle(order)
        return order

    def _expected_assignment(self, seed: int) -> Dict[str, set]:
        """The documented fill rule: debug/selection/holdout consume
        consecutive WHOLE groups of the seeded order until their quota."""
        by_group: Dict[str, List[str]] = {}
        for r in self._records():
            by_group.setdefault(problem_group_key(r["problem"]), []).append(r["id"])
        order = self._expected_order(seed)
        out: Dict[str, set] = {}
        taken = 0
        for part, quota in [("debug", 1), ("selection", 2), ("holdout", 2)]:
            ids: List[str] = []
            while taken < len(order) and len(ids) < quota:
                ids.extend(by_group[order[taken]])
                taken += 1
            out[part] = set(ids)
        return out

    def test_groups_never_split_across_splits(self):
        seed = 20261013
        parts, info = partition_records(self._records(), self.QUOTAS, seed)
        assert info["group_order"] == self._expected_order(seed)
        assert info["order_seed"] == seed
        expected = self._expected_assignment(seed)
        for part in ("debug", "selection", "holdout"):
            assert {r["id"] for r in parts[part]} == expected[part]
        assert info["n_groups_used"] == 3
        # No problem group has members in more than one split.
        part_of: Dict[str, str] = {}
        for part_name, recs in parts.items():
            for r in recs:
                key = problem_group_key(r["problem"])
                if key in part_of:
                    assert part_of[key] == part_name
                else:
                    part_of[key] = part_name

    def test_parts_are_disjoint_by_id(self):
        parts, _ = partition_records(self._records(), self.QUOTAS, 0)
        ids = [r["id"] for recs in parts.values() for r in recs]
        assert len(ids) == len(set(ids)) == 6

    def test_quotas_filled_by_whole_groups_and_shortfall_not_fabricated(self):
        seed = 0
        parts, info = partition_records(self._records(), self.QUOTAS, seed)
        expected = self._expected_assignment(seed)
        # Quotas are met by taking whole groups: a part holds the complete
        # group that crossed its quota, never a fraction of it.
        for part, quota in (("debug", 1), ("selection", 2)):
            assert {r["id"] for r in parts[part]} == expected[part]
            assert len(parts[part]) >= quota
            # every id in the part comes from whole groups of the order
            for r in parts[part]:
                g = problem_group_key(r["problem"])
                group_ids = {x["id"] for x in self._records() if problem_group_key(x["problem"]) == g}
                assert group_ids <= {x["id"] for x in parts[part]}
        # Holdout gets what remains (one group here); reported, not padded.
        assert {r["id"] for r in parts["holdout"]} == expected["holdout"]
        assert info["n_groups_used"] == 3

    def test_error_ratio_reported_per_split(self):
        seed = 0
        parts, info = partition_records(self._records(), self.QUOTAS, seed)
        expected = self._expected_assignment(seed)
        ratios = info["error_ratio"]
        assert ratios["dataset"] == round(4 / 6, 4)  # a2, b2, b3, c1 errored
        for part in ("debug", "selection", "holdout"):
            recs = [r for r in self._records() if r["id"] in expected[part]]
            if not recs:
                # a part can legitimately be empty (all groups consumed by
                # earlier parts); reported as None, never padded
                assert ratios[part] is None
                continue
            want = sum(1 for r in recs if r["label"] != -1) / len(recs)
            assert ratios[part] == round(want, 4)
        assert ratios["leftover"] is None  # everything was consumed
        # Ratios reflect the records actually assigned to each split.
        assert len(parts["selection"]) == len(expected["selection"])

    def test_order_seed_drives_partition_reproducibly(self):
        # same seed -> identical partition and info (reproducibility)
        parts1, info1 = partition_records(self._records(), self.QUOTAS, 20261013)
        parts2, info2 = partition_records(self._records(), self.QUOTAS, 20261013)
        assert parts1 == parts2
        assert info1 == info2
        # the seed is actually used: every sampled seed's group order equals
        # the deterministic shuffle of first-occurrence order with that seed
        for s in (0, 1, 7, 20261013):
            _, info = partition_records(self._records(), self.QUOTAS, s)
            assert info["group_order"] == self._expected_order(s)
            assert info["order_seed"] == s

    def test_build_items_group_id_and_fields(self):
        recs = [r for r in self._records() if r["id"].startswith("b")]
        items = build_items(recs, "gsm8k", "selection")
        assert [i.item_id for i in items] == ["b1", "b2", "b3"]
        assert {i.group_id for i in items} == {"gsm8k:bravo problem"}
        assert all(i.subset == "gsm8k" and i.split == "selection" for i in items)
        assert [i.label for i in items] == [-1, 2, 5]
        assert all(i.final_answer_correct is True for i in items)

    def test_build_items_auto_id_and_none_final_answer(self):
        rec = {"problem": "P", "steps": [], "label": -1, "final_answer_correct": None}
        items = build_items([rec], "gsm8k", "debug")
        assert items[0].item_id == "gsm8k_0"
        assert items[0].final_answer_correct is None
        assert items[0].gold == {"label": -1, "final_answer_correct": None, "subset": "gsm8k"}

    def test_problem_group_key_uses_normalized_text(self):
        assert problem_group_key("Alpha + Beta = 3") == "alpha+beta=3"
        assert problem_group_key("Alpha + Beta = 3") == normalize_for_matching("Alpha + Beta = 3")
        assert problem_group_key("ALPHA + BETA = 3") == "alpha+beta=3"  # case-insensitive
        assert problem_group_key("Alpha + Beta = 4") != "alpha + beta = 3"  # digits preserved

    def test_processbench_quotas_read_config(self):
        assert processbench_quotas(make_cfg()) == {"debug": 5, "selection": 25, "holdout": 25}


class TestSplitParentsDisjoint:
    def test_pools_exact_and_pairwise_disjoint(self):
        recs = {
            "train": [musique_record(pid=f"2hop__t{i}_{i}") for i in range(3)],
            "dev": [musique_record(pid=f"2hop__d{i}_{i+1}") for i in range(20)],
        }
        pools = split_parents(recs, make_cfg())
        assert pools["debug"] == sorted(r["id"] for r in recs["train"])
        dev_ids = sorted(r["id"] for r in recs["dev"])
        assert pools["selection"] == dev_ids[:6]
        assert pools["holdout"] == dev_ids[6:12]
        for a, b in itertools.combinations(pools.values(), 2):
            assert set(a) & set(b) == set()

    def test_dev_shortfall_reported_as_shortfall_not_fabricated(self):
        recs = {"dev": [musique_record(pid=f"2hop__{i}") for i in range(5)], "train": []}
        pools = split_parents(recs, make_cfg())
        assert len(pools["selection"]) == 5  # quota is 6, only 5 exist
        assert pools["holdout"] == []
        assert pools["debug"] == []


def _ref_records(n: int = 10) -> List[Dict[str, Any]]:
    return [{"id": str(i), "q": f"problem {i}"} for i in range(n)]


@dataclass
class _FakeRefItem:
    """Stands in for the adapter item dataclasses (Gsm8kItem etc.)."""

    q: str
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return {"split": self.split, "q": self.q}


def _ref_build(recs: List[Dict[str, Any]], split: str) -> List[_FakeRefItem]:
    return [_FakeRefItem(q=r["q"], split=split) for r in recs]


class TestReferenceExclusion:
    def test_overlap_indices_excluded_with_documented_reason(self, tmp_path):
        cfg = make_cfg()
        out = tmp_path / "out"
        res = _frozen_reference(
            "gsm8k", cfg, out, None, _ref_records(), _ref_build, 10,
            {1, 3}, lambda r, i: str(r.get("id") or i),
        )
        assert res["n_chosen"] == 8
        assert res["n_excluded_overlap"] == 2
        assert res["n_total_available"] == 10
        assert res["excluded"] == [
            {"id": "1", "reason": "overlap_with_local_debug_or_selection"},
            {"id": "3", "reason": "overlap_with_local_debug_or_selection"},
        ]
        manifest = json.loads((out / "manifests" / "gsm8k_reference.json").read_text(encoding="utf-8"))
        qs = {item["q"] for item in manifest["items"]}
        assert manifest["n_items"] == 8
        assert "problem 1" not in qs and "problem 3" not in qs
        assert "problem 0" in qs and "problem 9" in qs
        pilot = json.loads((out / "manifests" / "gsm8k_reference_pilot.json").read_text(encoding="utf-8"))
        assert pilot["n_items"] == 8  # pilot quota (10) exceeds what remains

    def test_quota_cap_stops_scanning_and_pilot_is_limited(self, tmp_path):
        splits = {
            "stages": {
                "reference": {"reference_per_dataset": 5},
                "pilot": {"reference_per_dataset": 2},
            }
        }
        cfg = make_cfg(splits=splits)
        out = tmp_path / "out2"
        res = _frozen_reference(
            "math", cfg, out, None, _ref_records(), _ref_build, 10,
            {1, 3}, lambda r, i: str(i),
        )
        assert res["n_chosen"] == 5
        assert res["n_excluded_overlap"] == 2  # exclusions seen before the cap
        pilot = json.loads((out / "manifests" / "math_reference_pilot.json").read_text(encoding="utf-8"))
        assert pilot["n_items"] == 2  # pilot quota applies

    def test_normalized_q_helper(self):
        assert _normalized_q("Solve A + B = 3") == "solve a+b=3"
