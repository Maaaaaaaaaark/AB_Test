"""Pilot-based cost/time estimates for the formal plans, with uncertainty.

Reads real prediction records from a completed plan (pilot or smoke) and
extrapolates per-item compute (Cproxy) and wall time to the target plans
(selection, holdout). Uncertainty comes from a group-clustered bootstrap over
the per-item means; no prices are invented (Cweighted stays null).

Output: artifacts/estimates/{source_plan}__to_{target_plan}.json + a plain
text summary. Estimates are clearly marked as estimates.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigBundle
from .metrics.stats import group_bootstrap
from .run.manifests import manifest_items_for_task

ESTIMATE_VERSION = "estimate-v1"


def _records(out_dir: Path, plan: str, config: str, task: str) -> List[Dict[str, Any]]:
    base = out_dir / "predictions" / plan / config / task
    if not base.exists():
        return []
    recs: List[Dict[str, Any]] = []
    for seed_dir in sorted(base.iterdir()):
        if not seed_dir.is_dir():
            continue
        for p in sorted(seed_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if rec.get("status") == "ok":
                recs.append(rec)
    return recs


def _n_text_for(cfg: ConfigBundle, out_dir: Path, config: str) -> Optional[int]:
    p = out_dir / "model_params" / f"{config}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("n_text")


def _target_counts(cfg: ConfigBundle, out_dir: Path, target: str) -> Dict[str, int]:
    """How many items the target plan will generate per task."""
    counts: Dict[str, int] = {}
    for task in cfg.plans[target].tasks:
        try:
            items = manifest_items_for_task(cfg, out_dir, target, task)
        except FileNotFoundError:
            counts[task] = 0
            continue
        counts[task] = len(items)
    return counts


def estimate(
    cfg: ConfigBundle, out_dir: Path, source_plan: str, target_plan: str, n_boot: int = 1000
) -> Dict[str, Any]:
    plan = cfg.plans[source_plan]
    target = cfg.plans[target_plan]
    n_seeds_target = len(cfg.splits["seeds"].get(target_plan, [0]))
    n_seeds_source = len(cfg.splits["seeds"].get(source_plan, [0]))
    target_counts = _target_counts(cfg, out_dir, target_plan)

    per_cell: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    # only configurations and tasks the TARGET plan actually runs
    for config in [c for c in plan.configs if c in target.configs]:
        n_text = _n_text_for(cfg, out_dir, config)
        if n_text is None:
            warnings.append(
                f"{config}: no artifacts/model_params/{config}.json - Cproxy "
                f"excluded from estimates (run `weval inspect-params` on the cluster)"
            )
        for task in [t for t in plan.tasks if t in target.tasks]:
            recs = _records(out_dir, source_plan, config, task)
            key = f"{config}/{task}"
            if not recs:
                per_cell[key] = {"n_records": 0, "note": "no pilot records yet"}
                continue
            n_items = target_counts.get(task, 0)
            # per-item stats over observed records
            cproxy_items = []
            tin, tout, wall = [], [], []
            for r in recs:
                tin.append(r.get("n_prompt_tokens", 0))
                tout.append(r.get("n_generated_tokens", 0))
                wall.append(r.get("wall_seconds", 0.0))
                if n_text:
                    cproxy_items.append(2 * n_text * (r.get("n_prompt_tokens", 0) + r.get("n_generated_tokens", 0)))
            mean_tin, mean_tout = sum(tin) / len(tin), sum(tout) / len(tout)
            mean_wall = sum(wall) / len(wall)
            wall_by_group: Dict[str, List[float]] = {}
            for r, w in zip(recs, wall):
                wall_by_group.setdefault(r.get("group_id") or r.get("item_id"), []).append(w)
            wall_ci = group_bootstrap(wall_by_group, n_boot=n_boot).to_dict() if len(wall) >= 3 else None
            cell = {
                "n_records": len(recs),
                "n_target_items_per_seed": n_items,
                "n_target_seeds": n_seeds_target,
                "n_source_seeds": n_seeds_source,
                "mean_tokens_in": mean_tin,
                "mean_tokens_out": mean_tout,
                "mean_wall_seconds": mean_wall,
                "wall_seconds_ci95": wall_ci,
                "est_total_items": n_items * n_seeds_target,
                "est_total_wall_seconds": mean_wall * n_items * n_seeds_target,
                "est_total_wall_hours": mean_wall * n_items * n_seeds_target / 3600.0,
                "est_total_tokens_in": mean_tin * n_items * n_seeds_target,
                "est_total_tokens_out": mean_tout * n_items * n_seeds_target,
            }
            if n_text:
                mean_cproxy = sum(cproxy_items) / len(cproxy_items)
                cell["n_text"] = n_text
                cell["mean_cproxy_per_item"] = mean_cproxy
                cell["est_total_cproxy"] = mean_cproxy * n_items * n_seeds_target
                cell["est_total_cproxy_tflops_ish"] = mean_cproxy * n_items * n_seeds_target / 1e12
            # source budget vs target budget: records that hit the token limit
            # make the extrapolation a LOWER bound (e.g. pilot MATH at 4096
            # tokens vs math_solve at 8192)
            n_len = sum(1 for r in recs if r.get("finish_reason") == "length")
            cell["n_source_length_stops"] = n_len
            if n_len:
                warnings.append(
                    f"{key}: {n_len}/{len(recs)} source records stopped at the token limit; "
                    "wall time / tokens for this cell are a lower bound"
                )
            per_cell[key] = cell

    return {
        "estimate_version": ESTIMATE_VERSION,
        "source_plan": source_plan,
        "target_plan": target_plan,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "extrapolation from observed per-item means; uncertainty = group-clustered bootstrap CI on wall time",
            "assumes sequential execution on one GPU per config",
            "Cproxy uses checkpoint-derived n_text; Cweighted stays null (no sourced prices)",
            "estimates are NOT guarantees - pilot hardware may differ from the cluster",
        ],
        "warnings": warnings,
        "per_config_task": per_cell,
    }


def print_estimate(est: Dict[str, Any]) -> None:
    print(f"estimate {est['source_plan']} -> {est['target_plan']} ({est['created_at_utc']})")
    for w in est["warnings"]:
        print(f"  WARNING: {w}")
    print()
    print(f"{'config/task':<34} {'items':>7} {'wall h':>8} {'tin':>9} {'tout':>9} {'Cproxy':>12} {'Cproxy/1e12':>11}")
    for key, c in sorted(est["per_config_task"].items()):
        if not c.get("n_records"):
            print(f"{key:<34} {c.get('note', 'no data'):>12}")
            continue
        cproxy = f"{c.get('mean_cproxy_per_item', 0):.3e}" if "mean_cproxy_per_item" in c else "-"
        print(
            f"{key:<34} {c['est_total_items']:>7} {c['est_total_wall_hours']:>8.2f} "
            f"{c['mean_tokens_in']:>9.1f} {c['mean_tokens_out']:>9.1f} {cproxy:>12} "
            f"{c.get('est_total_cproxy_tflops_ish', 0):>11.4f}"
        )
    print()
    for n in est["notes"]:
        print(f"  note: {n}")
