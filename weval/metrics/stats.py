"""Statistics: per-seed means, cross-seed means, clustered bootstrap CIs,
paired config differences.

Clustering: the bootstrap resamples PARENT PROBLEM GROUPS, so pairs, solution
variants and seeds of the same problem are always resampled together. The
macro-average preserves subset structure (per-subset statistic computed inside
each resample, then averaged across subsets).

All CIs are 95% percentile intervals (2.5 / 97.5) from n_boot resamples with
a fixed RNG seed for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

STATS_VERSION = "stats-v1"
N_BOOT_DEFAULT = 1000
BOOTSTRAP_SEED_DEFAULT = 0


@dataclass
class BootstrapCI:
    estimate: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    n_boot: int
    n_groups: int
    n_values: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci95_lo": self.lo,
            "ci95_hi": self.hi,
            "n_boot": self.n_boot,
            "n_groups": self.n_groups,
            "n_values": self.n_values,
            "seed": self.seed,
            "version": STATS_VERSION,
        }


def _pct(values: List[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo_i = int(k)
    hi_i = min(lo_i + 1, len(values) - 1)
    frac = k - lo_i
    return values[lo_i] * (1 - frac) + values[hi_i] * frac


def group_bootstrap(
    values_by_group: Dict[str, List[float]],
    stat_fn: Callable[[List[float]], Optional[float]] = lambda v: sum(v) / len(v) if v else None,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> BootstrapCI:
    """Clustered bootstrap: resample groups, apply stat_fn to pooled values."""
    rng = random.Random(seed)
    groups = list(values_by_group)
    all_values = [v for vs in values_by_group.values() for v in vs]
    estimate = stat_fn(all_values)
    samples: List[float] = []
    for _ in range(n_boot):
        picked = [rng.choice(groups) for _ in range(len(groups))]
        vals = [v for g in picked for v in values_by_group[g]]
        s = stat_fn(vals)
        if s is not None:
            samples.append(s)
    if not samples:
        return BootstrapCI(estimate, None, None, n_boot, len(groups), len(all_values), seed)
    return BootstrapCI(estimate, _pct(samples, 0.025), _pct(samples, 0.975), n_boot, len(groups), len(all_values), seed)


def macro_group_bootstrap(
    values_by_subset_group: Dict[str, Dict[str, List[float]]],
    subset_stat: Callable[[List[float]], Optional[float]] = lambda v: sum(v) / len(v) if v else None,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> BootstrapCI:
    """Subset-structured macro average with clustered bootstrap.

    Within each resample, every subset resamples ITS OWN groups, the subset
    statistic is computed, and the macro average is the mean across subsets.
    Subset structure is preserved inside every resample.
    """
    rng = random.Random(seed)

    def macro(vals_by_sg: Dict[str, Dict[str, List[float]]]) -> Optional[float]:
        stats = []
        for sg in vals_by_sg.values():
            pooled = [v for vs in sg.values() for v in vs]
            s = subset_stat(pooled)
            if s is not None:
                stats.append(s)
        if not stats:
            return None
        return sum(stats) / len(stats)

    estimate = macro(values_by_subset_group)
    n_groups = sum(len(sg) for sg in values_by_subset_group.values())
    n_values = sum(len(vs) for sg in values_by_subset_group.values() for vs in sg.values())
    samples: List[float] = []
    for _ in range(n_boot):
        resampled: Dict[str, Dict[str, List[float]]] = {}
        for subset, sg in values_by_subset_group.items():
            gs = list(sg)
            picked = [rng.choice(gs) for _ in range(len(gs))]
            # Preserve bootstrap multiplicity.  A dict keyed by the original
            # group id collapses repeated draws and is not a bootstrap sample.
            resampled[subset] = {
                f"{draw_idx}:{g}": sg[g]
                for draw_idx, g in enumerate(picked)
            }
        s = macro(resampled)
        if s is not None:
            samples.append(s)
    if not samples:
        return BootstrapCI(estimate, None, None, n_boot, n_groups, n_values, seed)
    return BootstrapCI(estimate, _pct(samples, 0.025), _pct(samples, 0.975), n_boot, n_groups, n_values, seed)


def paired_difference(
    values_by_group_a: Dict[str, Dict[str, float]],
    values_by_group_b: Dict[str, Dict[str, float]],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> Tuple[BootstrapCI, int, int]:
    """Paired (same-item) difference A - B, clustered by group.

    Only groups present in BOTH configs are compared (comparison incomplete
    otherwise); missing groups are counted and reported.
    """
    rng = random.Random(seed)
    common = sorted(set(values_by_group_a) & set(values_by_group_b))
    missing_a = sum(
        len(set(values_by_group_b[g]) - set(values_by_group_a.get(g, {})))
        for g in values_by_group_b
    )
    missing_b = sum(
        len(set(values_by_group_a[g]) - set(values_by_group_b.get(g, {})))
        for g in values_by_group_a
    )

    deltas_by_group: Dict[str, List[float]] = {}
    for g in common:
        # Exact observation alignment prevents a missing item/seed from
        # shifting every subsequent comparison within the group.
        keys = sorted(set(values_by_group_a[g]) & set(values_by_group_b[g]))
        deltas_by_group[g] = [
            values_by_group_a[g][k] - values_by_group_b[g][k]
            for k in keys
        ]

    all_deltas = [d for ds in deltas_by_group.values() for d in ds]
    estimate = sum(all_deltas) / len(all_deltas) if all_deltas else None
    samples: List[float] = []
    for _ in range(n_boot):
        picked = [rng.choice(common) for _ in range(len(common))] if common else []
        vals = [d for g in picked for d in deltas_by_group[g]]
        if vals:
            samples.append(sum(vals) / len(vals))
    if not samples:
        return BootstrapCI(estimate, None, None, n_boot, len(common), len(all_deltas), seed), missing_a, missing_b
    return BootstrapCI(estimate, _pct(samples, 0.025), _pct(samples, 0.975), n_boot, len(common), len(all_deltas), seed), missing_a, missing_b


def per_seed_means(values_by_group_seed: Dict[str, List[Tuple[int, float]]]) -> Dict[int, float]:
    """Mean per seed over all (group, seed) values."""
    by_seed: Dict[int, List[float]] = {}
    for vals in values_by_group_seed.values():
        for s, v in vals:
            by_seed.setdefault(s, []).append(v)
    return {s: sum(vs) / len(vs) for s, vs in sorted(by_seed.items())}


def weighted_strata_bootstrap(
    values_by_stratum_group: Dict[str, Dict[str, List[float]]],
    weights: Dict[str, float],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> BootstrapCI:
    """Post-stratified mean with a stratified clustered bootstrap.

    estimate = sum_s w_s * mean(stratum s), with the weights renormalized over
    the strata that have data. Inside every resample each stratum resamples
    ITS OWN groups (seeds of a problem move together), so the CI reflects the
    stratified design (e.g. MATH sampled with an equal number of problems per
    difficulty level, re-weighted to the eligible post-exclusion pool's level
    proportions).
    """
    rng = random.Random(seed)
    strata = [s for s, sg in values_by_stratum_group.items()
              if sg and weights.get(s, 0) > 0 and any(sg.values())]
    total_w = sum(weights[s] for s in strata)

    def weighted(by_sg: Dict[str, Dict[str, List[float]]]) -> Optional[float]:
        if not strata or total_w <= 0:
            return None
        acc = 0.0
        for s in strata:
            pooled = [v for vs in by_sg[s].values() for v in vs]
            if not pooled:
                return None
            acc += weights[s] / total_w * (sum(pooled) / len(pooled))
        return acc

    estimate = weighted(values_by_stratum_group)
    n_groups = sum(len(values_by_stratum_group[s]) for s in strata)
    n_values = sum(len(vs) for s in strata for vs in values_by_stratum_group[s].values())
    samples: List[float] = []
    for _ in range(n_boot):
        resampled: Dict[str, Dict[str, List[float]]] = {}
        for s in strata:
            sg = values_by_stratum_group[s]
            gs = list(sg)
            picked = [rng.choice(gs) for _ in range(len(gs))]
            resampled[s] = {f"{i}:{g}": sg[g] for i, g in enumerate(picked)}
        v = weighted(resampled)
        if v is not None:
            samples.append(v)
    if not samples:
        return BootstrapCI(estimate, None, None, n_boot, n_groups, n_values, seed)
    return BootstrapCI(estimate, _pct(samples, 0.025), _pct(samples, 0.975), n_boot, n_groups, n_values, seed)
