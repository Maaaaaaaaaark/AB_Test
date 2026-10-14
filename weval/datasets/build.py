"""Build orchestration: prepared data -> manifests, splits, review file,
coverage and leakage reports.

Everything here is deterministic given the prepared data and the pinned
configs. Eligibility (which parents/pairs/records are used) is frozen here,
BEFORE any model output is seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ConfigBundle, canonical_json, fingerprint
from ..provenance import DatasetProvenance
from . import gsm8k as gsm8k_adapter
from . import hotpotqa as hotpotqa_adapter
from . import leakage as leakage_mod
from . import mathbench as math_adapter
from . import musique as musique_adapter
from . import processbench as pb_adapter

BUILD_VERSION = "build-v1"


@dataclass
class BuildResult:
    manifests: Dict[str, Path] = field(default_factory=dict)
    review_file: Optional[Path] = None
    coverage: Dict[str, Any] = field(default_factory=dict)
    leakage: Dict[str, Any] = field(default_factory=dict)
    blocked: Dict[str, str] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_version": BUILD_VERSION,
            "manifests": {k: str(v) for k, v in self.manifests.items()},
            "review_file": str(self.review_file) if self.review_file else None,
            "coverage": self.coverage,
            "leakage": self.leakage,
            "blocked": self.blocked,
        }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _manifest(dataset: str, split: str, items: List[Dict[str, Any]], cfg: ConfigBundle, provenance: Optional[DatasetProvenance]) -> Dict[str, Any]:
    prov_fp = provenance.fingerprint() if provenance else None
    return {
        "manifest_version": 1,
        "build_version": BUILD_VERSION,
        "dataset": dataset,
        "split": split,
        "created_at": None,  # filled by caller to keep determinism (not part of fingerprint)
        "data_provenance_fingerprint": prov_fp,
        "config_fingerprint": fingerprint(
            {
                "instruction_versions": {
                    t: cfg.task(t).instruction_version for t in cfg.tasks
                },
                "split_seeds": cfg.splits.get("split_seeds", {}),
                "stage_sizes": cfg.splits.get("stages", {}),
            }
        ),
        "n_items": len(items),
        "items": items,
    }


def _load_musique_records(data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "dev"):
        p = data_dir / "musique" / f"musique_ans_v1.0_{split}.jsonl"
        if not p.exists():
            continue
        out[split] = musique_adapter.load_musique_jsonl(p)
    return out


def build_musique_splits(
    cfg: ConfigBundle, data_dir: Path, out_dir: Path, provenance: Optional[DatasetProvenance]
) -> BuildResult:
    res = BuildResult()
    records = _load_musique_records(data_dir)
    if not records.get("dev"):
        res.blocked["musique"] = "MuSiQue dev jsonl not prepared; run `weval data-prep` first."
        return res
    pools = musique_adapter.split_parents(records, cfg)
    stages = cfg.splits["stages"]
    seed = cfg.splits["split_seeds"]["musique_subquestion_choice"]
    order_seed = int(cfg.splits["split_seeds"].get(
        "musique_paragraph_order", musique_adapter.DEFAULT_PARAGRAPH_ORDER_SEED
    ))
    review_path = out_dir / "manifests" / "musique_negative_review.json"
    # Full entry maps (not just approved): build_musique applies the status
    # rules itself, so human-excluded pairs/positives are reported as review
    # exclusions instead of silently disappearing.
    review_map = musique_adapter.load_review_map(review_path)  # empty on first build
    positive_review_map = musique_adapter.load_positive_review_map(review_path)

    review_pool_parents: List[Dict[str, Any]] = []
    review_positives: Dict[str, musique_adapter.MusiqueItem] = {}
    review_negatives: Dict[str, Optional[musique_adapter.MusiqueItem]] = {}
    review_exclusions: Dict[str, str] = {}

    def _build(pool_name: str, split: str, n_parents: Optional[int]) -> Dict[str, Any]:
        ids = pools[pool_name]
        if n_parents is not None:
            ids = ids[:n_parents]
        src = records["train"] if pool_name == "debug" else records["dev"]
        by_id = {r["id"]: r for r in src}
        parent_records = [by_id[pid] for pid in ids if pid in by_id]
        items, report, pair_exclusions = musique_adapter.build_musique(
            parent_records, split, seed,
            parent_ids=[r["id"] for r in parent_records],
            approved_pairs=review_map,
            order_seed=order_seed,
            positive_reviews=positive_review_map,
        )
        return {"items": items, "report": report, "pair_exclusions": pair_exclusions, "src": src}

    per_split: Dict[str, Dict[str, Any]] = {}
    per_split["debug"] = _build("debug", "debug", None)
    per_split["selection"] = _build("selection", "selection", stages["selection"]["musique_parents"])
    per_split["holdout"] = _build("holdout", "holdout", stages["holdout"]["musique_parents"])

    for split_name, blob in per_split.items():
        rep = blob["report"].to_dict()
        rep["split"] = split_name
        res.coverage[f"musique_{split_name}"] = rep
        items = [i.to_dict() for i in blob["items"]]
        manifest = _manifest("musique", split_name, items, cfg, provenance)
        path = out_dir / "manifests" / f"musique_{split_name}.json"
        _write_json(path, manifest)
        res.manifests[f"musique_{split_name}"] = path

    # Review file covers: pilot parents (first N eligible debug parents), all
    # eligible selection and holdout parents. It is written from the build
    # CANDIDATES (independent of any decision), so a human exclusion never
    # makes an entry - or its decision - disappear on the next rebuild.
    # Unreviewed smoke pairs stay development-only.
    pilot_n = stages["pilot"]["musique_parents"]

    def _add_candidates(parent_ids: List[str], src: List[Dict[str, Any]], split: str, limit: Optional[int]) -> None:
        by_id = {r["id"]: r for r in src}
        n_added = 0
        for pid in parent_ids:
            if limit is not None and n_added >= limit:
                break
            record = by_id.get(pid)
            if record is None:
                continue
            subq, _eligible = musique_adapter.choose_subquestion(record, seed)
            if subq < 0:
                continue
            pos = musique_adapter.build_positive(record, subq, split, order_seed=order_seed)
            neg, reason = musique_adapter.build_pair(record, subq, pos, split)
            review_pool_parents.append(record)
            review_positives[pid] = pos
            review_negatives[pid] = neg
            review_exclusions[pid] = reason
            n_added += 1

    _add_candidates(pools["debug"], records.get("train", []), "debug", pilot_n)
    _add_candidates(pools["selection"][: stages["selection"]["musique_parents"]], records["dev"], "selection", None)
    _add_candidates(pools["holdout"][: stages["holdout"]["musique_parents"]], records["dev"], "holdout", None)

    n_cand = musique_adapter.write_review_file(
        review_path, review_pool_parents, review_positives, review_negatives, review_exclusions
    )
    res.review_file = review_path
    res.coverage["musique_review"] = {
        "n_review_pool_parents": len(review_pool_parents),
        "n_pair_candidates": n_cand,
        "n_excluded_at_build": len(review_pool_parents) - n_cand,
    }
    return res


def build_processbench_splits(
    cfg: ConfigBundle, data_dir: Path, out_dir: Path, provenance: Optional[DatasetProvenance]
) -> BuildResult:
    res = BuildResult()
    cache_dir = str(data_dir / "hf_cache")
    try:
        data = pb_adapter.load_processbench(cache_dir=cache_dir)
    except Exception as e:  # noqa: BLE001 - surfaced in blocked[]
        res.blocked["processbench"] = f"could not load ProcessBench: {e}"
        return res
    quotas = pb_adapter.processbench_quotas(cfg)
    for subset in pb_adapter.PROCESSBENCH_SUBSETS:
        parts, info = pb_adapter.partition_records(data[subset], quotas, cfg.splits["split_seeds"]["processbench_group_order"])
        for split_name in ("debug", "selection", "holdout"):
            items = pb_adapter.build_items(parts[split_name], subset, split_name)
            manifest = _manifest("processbench", f"{subset}_{split_name}", [i.to_dict() for i in items], cfg, provenance)
            path = out_dir / "manifests" / f"processbench_{subset}_{split_name}.json"
            _write_json(path, manifest)
            res.manifests[f"processbench_{subset}_{split_name}"] = path
            res.coverage[f"processbench_{subset}_{split_name}"] = {
                "n_records": len(items),
                "error_ratio": info["error_ratio"][split_name],
                "dataset_error_ratio": info["error_ratio"]["dataset"],
                "n_groups": len({i.group_id for i in items}),
            }
    return res


def _frozen_reference(
    dataset: str, cfg: ConfigBundle, out_dir: Path, provenance: Optional[DatasetProvenance],
    records: List[Dict[str, Any]], build_fn, n_total: int,
    excluded_idx: set[int], id_of,
) -> Dict[str, Any]:
    n_ref = cfg.splits["stages"]["reference"]["reference_per_dataset"]
    n_pilot = cfg.splits["stages"]["pilot"]["reference_per_dataset"]
    import random

    chosen: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    order = list(range(len(records)))
    seed = int(cfg.splits.get("split_seeds", {}).get("reference_subset_order", 0))
    random.Random(seed).shuffle(order)
    for i in order:
        r = records[i]
        if len(chosen) >= n_ref:
            break
        rid = id_of(r, i)
        if i in excluded_idx:
            excluded.append({"id": rid, "reason": "overlap_with_local_debug_or_selection"})
            continue
        chosen.append(r)
    items = build_fn(chosen, "reference")
    pilot_items = build_fn(chosen[:n_pilot], "reference")
    manifest = _manifest(dataset, "reference", [i.to_dict() for i in items], cfg, provenance)
    path = out_dir / "manifests" / f"{dataset}_reference.json"
    _write_json(path, manifest)
    pilot_manifest = _manifest(dataset, "reference_pilot", [i.to_dict() for i in pilot_items], cfg, provenance)
    pilot_path = out_dir / "manifests" / f"{dataset}_reference_pilot.json"
    _write_json(pilot_path, pilot_manifest)
    return {
        "n_chosen": len(items),
        "n_excluded_overlap": len(excluded),
        "excluded": excluded,
        "manifest": str(path),
        "pilot_manifest": str(pilot_path),
        "n_total_available": n_total,
    }


def build_references(
    cfg: ConfigBundle, data_dir: Path, out_dir: Path,
    provenance: Dict[str, Optional[DatasetProvenance]],
    local_problems: Dict[str, set[str]],  # normalized problem text used in local debug/selection
) -> BuildResult:
    res = BuildResult()
    cache_dir = str(data_dir / "hf_cache")

    # GSM8K
    try:
        gsm_records = gsm8k_adapter.load_gsm8k_main_test(cache_dir=cache_dir)
        gsm_excluded = {
            i for i, r in enumerate(gsm_records)
            if _normalized_q(r["question"]) in local_problems.get("gsm8k", set())
        }
        # The prompt's fixed worked example (a TRAIN question) must never be
        # an evaluation item; identical test questions are excluded too.
        exemplar = getattr(cfg.task("gsm8k"), "exemplar", None) if "gsm8k" in cfg.tasks else None
        n_exemplar_overlap = 0
        if exemplar:
            ex_q = _normalized_q(str(exemplar["question"]))
            overlap = {i for i, r in enumerate(gsm_records) if _normalized_q(r["question"]) == ex_q}
            n_exemplar_overlap = len(overlap)
            gsm_excluded |= overlap
        res.coverage["gsm8k_reference"] = _frozen_reference(
            "gsm8k", cfg, out_dir, provenance.get("gsm8k"), gsm_records,
            lambda recs, split: gsm8k_adapter.build_items(recs, split),
            len(gsm_records), gsm_excluded, lambda r, i: str(r.get("id") or i)
        )
        res.coverage["gsm8k_reference"]["n_exemplar_overlap_excluded"] = n_exemplar_overlap
        res.manifests["gsm8k_reference"] = Path(res.coverage["gsm8k_reference"]["manifest"])
        res.manifests["gsm8k_reference_pilot"] = Path(res.coverage["gsm8k_reference"]["pilot_manifest"])
    except Exception as e:  # noqa: BLE001
        res.blocked["gsm8k"] = f"could not load GSM8K: {e}"

    # HotpotQA
    try:
        hp_records = hotpotqa_adapter.load_hotpotqa_distractor_dev(cache_dir=cache_dir)
        res.coverage["hotpotqa_reference"] = _frozen_reference(
            "hotpotqa", cfg, out_dir, provenance.get("hotpotqa"), hp_records,
            lambda recs, split: hotpotqa_adapter.build_items(recs, split),
            len(hp_records), set(), lambda r, i: str(r["id"])
        )
        res.manifests["hotpotqa_reference"] = Path(res.coverage["hotpotqa_reference"]["manifest"])
        res.manifests["hotpotqa_reference_pilot"] = Path(res.coverage["hotpotqa_reference"]["pilot_manifest"])
    except Exception as e:  # noqa: BLE001
        res.blocked["hotpotqa"] = f"could not load HotpotQA: {e}"

    # MATH
    try:
        math_data = math_adapter.load_math(cache_dir=cache_dir)
        math_records = math_data["test"]
        math_excluded = {
            i for i, r in enumerate(math_records)
            if _normalized_q(r["problem"]) in local_problems.get("math", set())
        }
        res.coverage["math_reference"] = _frozen_reference(
            "math", cfg, out_dir, provenance.get("math"), math_records,
            lambda recs, split: math_adapter.build_items(recs, split),
            len(math_records), math_excluded, lambda r, i: str(i)
        )
        res.manifests["math_reference"] = Path(res.coverage["math_reference"]["manifest"])
        res.manifests["math_reference_pilot"] = Path(res.coverage["math_reference"]["pilot_manifest"])
    except Exception as e:  # noqa: BLE001
        res.blocked["math"] = f"could not load MATH (task blocked, others unaffected): {e}"

    return res


def _normalized_q(text: str) -> str:
    from .normalization import normalize_for_matching

    return normalize_for_matching(text)


def build_all(cfg: ConfigBundle, data_dir: Path, out_dir: Path, provenance_map: Dict[str, Optional[DatasetProvenance]]) -> BuildResult:
    res = BuildResult()

    musique_res = build_musique_splits(cfg, data_dir, out_dir, provenance_map.get("musique"))
    res.manifests.update(musique_res.manifests)
    res.review_file = musique_res.review_file
    res.coverage.update(musique_res.coverage)
    res.blocked.update(musique_res.blocked)

    pb_res = build_processbench_splits(cfg, data_dir, out_dir, provenance_map.get("processbench"))
    res.manifests.update(pb_res.manifests)
    res.coverage.update(pb_res.coverage)
    res.blocked.update(pb_res.blocked)

    # Local problems used in debug/selection (for reference exclusions).
    local_problems: Dict[str, set[str]] = {"gsm8k": set(), "math": set()}
    for key, path in pb_res.manifests.items():
        if not key.startswith("processbench_"):
            continue
        subset, split_name = key[len("processbench_"):].rsplit("_", 1)
        if split_name not in ("debug", "selection"):
            continue
        if subset not in ("gsm8k", "math"):
            continue
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest["items"]:
            local_problems[subset].add(_normalized_q(item["problem"]))

    ref_res = build_references(cfg, data_dir, out_dir, provenance_map, local_problems)
    res.manifests.update(ref_res.manifests)
    res.coverage.update(ref_res.coverage)
    res.blocked.update(ref_res.blocked)

    _write_json(out_dir / "build_result.json", res.to_dict())
    return res
