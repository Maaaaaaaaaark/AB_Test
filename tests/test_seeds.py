"""Item-seed derivation stability (weval/models/backend.py::make_item_seed).

item_seed = (run_seed * 2654435761 + first 8 bytes of sha256(item_id) as
little-endian) mod 2^32. The same (item_id, run_seed) always yields the same
seed regardless of which other items ran first, and the result is a uint32.
The identical formula is duplicated in weval/datasets/musique.py::_item_seed
(for split-seed subquestion choice); both are cross-checked here.
"""

from __future__ import annotations

import hashlib

from weval.datasets.musique import _item_seed as musique_item_seed
from weval.models.backend import make_item_seed

MULTIPLIER = 2654435761
MODULUS = 2**32

ITEMS = ["2hop__1_2", "3hop__7_11_3", "item-a", "item-b", "ünicode-ïd", ""]
RUN_SEEDS = [0, 1, 2, 42, 20261013, 2**31 - 1, 2**31, 2**31 + 7]


def _expected(run_seed: int, item_id: str) -> int:
    h = hashlib.sha256(item_id.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "little")
    return (run_seed * MULTIPLIER + x) % MODULUS


class TestStability:
    def test_same_item_and_seed_same_item_seed(self):
        for seed in RUN_SEEDS:
            for item_id in ITEMS:
                assert make_item_seed(seed, item_id) == make_item_seed(seed, item_id)

    def test_different_run_seeds_differ(self):
        # MULTIPLIER is odd, so (r1 - r2) * MULTIPLIER != 0 mod 2^32 for any
        # two distinct run seeds in [0, 2^32): the seeds are guaranteed to
        # differ, not merely likely to.
        for item_id in ITEMS:
            for r1, r2 in zip(RUN_SEEDS, RUN_SEEDS[1:]):
                assert make_item_seed(r1, item_id) != make_item_seed(r2, item_id)

    def test_different_items_under_same_run_seed(self):
        seeds = [make_item_seed(42, i) for i in ITEMS]
        assert len(set(seeds)) == len(ITEMS)


class TestOrderIndependence:
    def test_derivation_is_independent_of_order(self):
        ids = ["alpha", "beta", "gamma", "delta"]
        order_a = {i: make_item_seed(7, i) for i in ids}
        order_b = {i: make_item_seed(7, i) for i in reversed(ids)}
        # per-item equality regardless of the order the items were derived in
        assert order_a == order_b
        # and deriving with a different prefix/other items interleaved
        for i in ids:
            make_item_seed(7, "unrelated-item")
            assert make_item_seed(7, i) == order_a[i]


class TestFormula:
    def test_matches_documented_constants(self):
        for seed in RUN_SEEDS:
            for item_id in ITEMS:
                assert make_item_seed(seed, item_id) == _expected(seed, item_id)

    def test_run_seed_zero_is_the_hash_term(self):
        for item_id in ITEMS:
            h = hashlib.sha256(item_id.encode("utf-8")).digest()
            x = int.from_bytes(h[:8], "little")
            assert make_item_seed(0, item_id) == x % MODULUS

    def test_large_run_seed_wraps_modulo_2_32(self):
        # run_seed 2**32 + 7 must behave exactly like run_seed 7 (mod 2^32)
        for item_id in ITEMS:
            assert make_item_seed(2**32 + 7, item_id) == make_item_seed(7, item_id)

    def test_item_seed_is_a_stable_uint32(self):
        for seed in RUN_SEEDS:
            for item_id in ITEMS:
                s = make_item_seed(seed, item_id)
                assert isinstance(s, int)
                assert 0 <= s < MODULUS


class TestConsistencyWithMusiqueSplitSeed:
    def test_same_formula_as_musique_item_seed(self):
        """weval/datasets/musique.py duplicates the derivation for the
        split-seed subquestion choice; both must stay in sync."""
        for seed in RUN_SEEDS:
            for item_id in ITEMS:
                assert make_item_seed(seed, item_id) == musique_item_seed(seed, item_id)

    def test_choose_subquestion_is_deterministic_across_calls(self, record):
        from weval.datasets.musique import choose_subquestion

        assert choose_subquestion(record, 20261013) == choose_subquestion(record, 20261013)
