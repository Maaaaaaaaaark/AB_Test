"""Standalone single-model math solving: frozen GSM8K / MATH samples.

Descriptive comparison of how well each configuration solves math problems
ON ITS OWN (one model, one problem, no other agent). It never nominates a
worker and has no holdout.

Sampling (fixed BEFORE any model output is seen, seeded, written to the
manifest and never redrawn):

  * GSM8K  - official main test split (1,319 problems): ``gsm8k_items``
             problems drawn uniformly at random;
  * MATH   - original test split (5,000 problems): stratified by difficulty,
             ``math_per_level`` problems from each of Level 1-5 (equal
             allocation). Because equal allocation over-represents the easy
             levels relative to the eligible post-exclusion pool, the report
             gives BOTH the plain mean over the sample and the mean re-weighted
             to that pool's level proportions (the latter is the headline MATH
             accuracy).
             Records whose level is not one of Level 1-5 are excluded and
             counted.

Exclusions (both criteria of the existing overlap module: exact match after
``normalize_for_matching`` and near-duplicate with token Jaccard >= 0.95):

  * every ProcessBench problem (all subsets, all records - not only the
    partitions used so far);
  * every problem of the GSM8K / MATH reference subsets (which contain the
    pilot items);
  * the GSM8K prompt's worked example (a TRAIN question).

Within-sample duplicates are also skipped (a candidate that duplicates an
already chosen problem is passed over). Item ids are the record's position
in the pinned test split (``test-<index>``), so they are stable and never
collide with the reference ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .normalization import NEAR_DUP_JACCARD, normalize_for_matching

MATH_SOLVE_SAMPLER_VERSION = "math-solve-sampler-v1"
MATH_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]


class InsufficientPoolError(ValueError):
    """Not enough eligible problems to draw the configured sample."""


# ---------------------------------------------------------------------------
# exact / near-duplicate matching
# ---------------------------------------------------------------------------

@dataclass
class _Ref:
    source: str
    ref_id: str
    norm: str
    tokens: frozenset


class DuplicateIndex:
    """Exact (normalized string) and near-duplicate (token Jaccard >=
    threshold) lookup against a set of reference problems.

    Near-duplicate search uses the size bound J(A, B) <= min/max(|A|, |B|), so
    only references whose token-set size is within the feasible window are
    compared. Results are identical to a brute-force scan (tested)."""

    def __init__(self, threshold: float = NEAR_DUP_JACCARD) -> None:
        self.threshold = threshold
        self._exact: Dict[str, _Ref] = {}
        self._by_size: Dict[int, List[_Ref]] = {}

    def add(self, source: str, ref_id: str, text: str) -> None:
        norm = normalize_for_matching(text or "")
        ref = _Ref(source, str(ref_id), norm, frozenset(norm.split()))
        self._exact.setdefault(norm, ref)
        self._by_size.setdefault(len(ref.tokens), []).append(ref)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_size.values())

    def match(self, text: str) -> Optional[Dict[str, Any]]:
        norm = normalize_for_matching(text or "")
        hit = self._exact.get(norm)
        if hit is not None:
            return {"kind": "exact", "source": hit.source, "ref_id": hit.ref_id}
        toks = frozenset(norm.split())
        n = len(toks)
        t = self.threshold
        if n == 0:
            return None
        lo = int(n * t)  # sizes below t*n cannot reach the threshold
        hi = int(n / t) + 1
        best: Optional[Tuple[float, _Ref]] = None
        for size in range(max(lo, 0), hi + 1):
            for ref in self._by_size.get(size, ()):
                if not ref.tokens:
                    continue
                inter = len(toks & ref.tokens)
                union = len(toks | ref.tokens)
                j = inter / union if union else 0.0
                if j >= t and (best is None or j > best[0]):
                    best = (j, ref)
        if best is None:
            return None
        return {"kind": "near_duplicate", "source": best[1].source,
                "ref_id": best[1].ref_id, "jaccard": round(best[0], 4)}


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    chosen: List[int] = field(default_factory=list)       # record indices, in order
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)


def _draw(
    candidates: Sequence[int],
    text_of: Callable[[int], str],
    n: int,
    rng_seed: int,
    exclusions: DuplicateIndex,
    already: DuplicateIndex,
    label: str,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    order = list(candidates)
    random.Random(rng_seed).shuffle(order)
    chosen: List[int] = []
    excluded: List[Dict[str, Any]] = []
    for idx in order:
        if len(chosen) >= n:
            break
        text = text_of(idx)
        hit = exclusions.match(text)
        if hit is not None:
            excluded.append({"index": idx, "stratum": label, "reason": f"overlap_{hit['source']}", **hit})
            continue
        dup = already.match(text)
        if dup is not None:
            excluded.append({"index": idx, "stratum": label, "reason": "duplicate_within_sample", **dup})
            continue
        chosen.append(idx)
        already.add("sample", f"test-{idx}", text)
    if len(chosen) < n:
        raise InsufficientPoolError(
            f"{label}: only {len(chosen)} eligible problems after exclusions, need {n}"
        )
    return chosen, excluded


def sample_gsm8k(
    records: List[Dict[str, Any]], n: int, seed: int, exclusions: DuplicateIndex,
) -> SampleResult:
    already = DuplicateIndex(exclusions.threshold)
    chosen, excluded = _draw(
        range(len(records)), lambda i: records[i]["question"], n, seed, exclusions, already, "gsm8k",
    )
    res = SampleResult(chosen=chosen, excluded=excluded)
    res.counts = {
        "population": len(records),
        "n_sampled": len(chosen),
        "n_excluded_examined": len(excluded),
        "excluded_by_reason": _count_by(excluded, "reason"),
    }
    return res


def sample_math_stratified(
    records: List[Dict[str, Any]], per_level: int, seed: int, exclusions: DuplicateIndex,
) -> SampleResult:
    by_level: Dict[str, List[int]] = {lvl: [] for lvl in MATH_LEVELS}
    invalid_level: List[int] = []
    for i, r in enumerate(records):
        lvl = str(r.get("level"))
        if lvl in by_level:
            by_level[lvl].append(i)
        else:
            invalid_level.append(i)
    already = DuplicateIndex(exclusions.threshold)
    chosen: List[int] = []
    excluded: List[Dict[str, Any]] = []
    for k, lvl in enumerate(MATH_LEVELS):
        # a distinct, deterministic stream per level
        c, e = _draw(by_level[lvl], lambda i: records[i]["problem"], per_level,
                     seed * 10 + k + 1, exclusions, already, lvl)
        chosen.extend(c)
        excluded.extend(e)
    res = SampleResult(chosen=chosen, excluded=excluded)
    population_levels = {lvl: len(by_level[lvl]) for lvl in MATH_LEVELS}
    # Excluded overlaps have zero inclusion probability in this experiment,
    # so the headline estimand is the eligible post-exclusion pool. Scan the
    # entire pool here (rather than only candidates seen before a quota fills)
    # so the post-stratification weights cannot depend on shuffle order.
    eligible_levels = {
        lvl: sum(1 for i in by_level[lvl] if exclusions.match(records[i]["problem"]) is None)
        for lvl in MATH_LEVELS
    }
    res.counts = {
        "population": len(records),
        "population_level_counts": population_levels,
        "eligible_level_counts": eligible_levels,
        "n_externally_excluded_in_population": sum(population_levels.values()) - sum(eligible_levels.values()),
        "n_invalid_level_excluded": len(invalid_level),
        "per_level": per_level,
        "n_sampled": len(chosen),
        "n_excluded_examined": len(excluded),
        "excluded_by_reason": _count_by(excluded, "reason"),
        "sample_subject_counts": _count_by([{"type": str(records[i].get("type"))} for i in chosen], "type"),
        "sample_level_counts": _count_by([{"level": str(records[i].get("level"))} for i in chosen], "level"),
    }
    return res


def level_weights(population_level_counts: Dict[str, int]) -> Dict[str, float]:
    """Declared pool's level proportions for re-weighted MATH accuracy."""
    total = sum(population_level_counts.get(l, 0) for l in MATH_LEVELS)
    if total <= 0:
        return {}
    return {l: population_level_counts.get(l, 0) / total for l in MATH_LEVELS}


def _count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
