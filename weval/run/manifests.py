"""Manifest resolution shared by the runner and the report layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ConfigBundle

PROCESSBENCH_SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]


def _manifest_paths(out_dir: Path, task: str, stage: str) -> List[Path]:
    mdir = out_dir / "manifests"
    paths: List[Path] = []
    if task == "musique_local":
        split = "debug" if stage in ("smoke", "pilot") else stage
        paths.append(mdir / f"musique_{split}.json")
    elif task == "processbench":
        split = "debug" if stage in ("smoke", "pilot") else stage
        paths.extend(mdir / f"processbench_{s}_{split}.json" for s in PROCESSBENCH_SUBSETS)
    elif task in ("gsm8k", "math", "hotpotqa"):
        if stage in ("smoke", "pilot"):
            paths.append(mdir / f"{task}_reference_pilot.json")
        else:
            paths.append(mdir / f"{task}_reference.json")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing manifests for {task}@{stage}: {missing}; run `weval build` first")
    return paths


def _select_items(items: List[Dict[str, Any]], stage: str, task: str, cfg: ConfigBundle) -> List[Dict[str, Any]]:
    """Apply stage sizes. Musique sizes count PARENTS (positives + their
    negatives); ProcessBench sizes count items PER SUBSET; other tasks count
    items.

    ProcessBench stages keep problem groups whole (a group is never split),
    matching the build-time partition - except smoke, a tiny engineering
    screen whose flat per-subset cap may cut a group mid-group; its outputs
    never support qualification claims."""
    sizes = cfg.splits["stages"][stage]
    if task == "musique_local":
        k = sizes["musique_parents"]
        positives = [it for it in items if it.get("kind") == "positive"]
        chosen_parents = {p["parent_id"] for p in positives[:k]}
        return [it for it in items if it.get("parent_id") in chosen_parents]
    if task == "processbench":
        k = sizes["processbench_per_subset"]
        picked: List[Dict[str, Any]] = []
        per_subset: Dict[str, int] = {}
        if stage == "smoke":
            for it in items:
                subset = it["subset"]
                n = per_subset.get(subset, 0)
                if n < k:
                    picked.append(it)
                    per_subset[subset] = n + 1
            return picked
        # Whole-group selection: include complete groups until each subset's
        # quota is met; the group that crosses the quota is included in full.
        open_group: Dict[str, Optional[str]] = {}
        for it in items:
            subset = it["subset"]
            gid = it.get("group_id") or it["item_id"]
            n = per_subset.get(subset, 0)
            if n < k:
                picked.append(it)
                per_subset[subset] = n + 1
                open_group[subset] = gid
            elif open_group.get(subset) == gid:
                picked.append(it)
                per_subset[subset] += 1
        return picked
    return items[: sizes["reference_per_dataset"]]


def manifest_items_for_task(
    cfg: ConfigBundle, out_dir: Path, plan_name: str, task: str
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for p in _manifest_paths(out_dir, task, plan_name):
        with open(p, encoding="utf-8") as f:
            all_items.extend(json.load(f)["items"])
    return _select_items(all_items, plan_name, task, cfg)
