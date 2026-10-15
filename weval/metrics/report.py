"""Report building: parse -> score -> aggregate -> statistics -> recommendation.

Reads raw prediction records (never regenerating anything), re-derives parsed
answers and scores with the pinned parser/scorer versions, then computes:

  * per-seed metrics and cross-seed means,
  * 95% clustered bootstrap CIs (parent problem groups; subset structure
    preserved for ProcessBench macro metrics),
  * paired config differences on identical items,
  * Cproxy / Cweighted / Efficiency_compute,
  * role qualification against thresholds.yaml - a nomination is produced
    ONLY by the selection stage with thresholds.status == "confirmed";
    holdout only checks the selection nominee (pass/fail, never re-picks);
    smoke/pilot/reference show metrics next to the thresholds for
    information and never nominate.

Unresolved infrastructure failures mark the comparison incomplete and are
listed explicitly (report_incomplete_on_unresolved).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ConfigBundle, PlanCfg
from ..parsing import PARSERS, PARSER_VERSIONS
from ..parsing.reasoning import merge_stop_strings
from ..run.manifests import manifest_items_for_task, manifest_sampling
from ..run.store import PredictionStore
from ..scoring import SCORERS, SCORER_VERSIONS
from ..scoring.processbench_score import subset_error_correct_f1
from .compute import COMPUTE_VERSION, aggregate_cproxy, cweighted, efficiency_compute, load_prices
from .stats import (
    BOOTSTRAP_SEED_DEFAULT,
    N_BOOT_DEFAULT,
    BootstrapCI,
    group_bootstrap,
    macro_group_bootstrap,
    paired_difference,
    weighted_strata_bootstrap,
)

REPORT_VERSION = "report-v5"

# Only this plan stage nominates role workers; holdout only checks the
# selection nominee; every other stage is informational.
NOMINATING_STAGE = "selection"
HOLDOUT_STAGE = "holdout"


def _answer_format_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """How the short answer was obtained (free-text QA tasks): an explicit
    "Answer:" line, a single-line reply, a single line with an inline
    "Answer:" marker (format deviation, answer still extracted), or a failure
    (invalid format = no extractable answer; missing final = reasoning never
    closed / empty)."""
    out = {"answer_line": 0, "single_line": 0, "inline_marker": 0, "invalid_format": 0, "missing_final": 0}
    for r in records:
        st = r.get("status")
        if st in ("invalid_format", "missing_final"):
            out[st] += 1
            continue
        src = (r.get("detail") or {}).get("answer_source")
        if src in out:
            out[src] += 1
    return out


@dataclass
class ReportData:
    plan: str
    created_at_utc: str
    thresholds_status: str
    per_config: Dict[str, Any] = field(default_factory=dict)
    compute: Dict[str, Any] = field(default_factory=dict)
    paired: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": REPORT_VERSION,
            "plan": self.plan,
            "created_at_utc": self.created_at_utc,
            "thresholds_status": self.thresholds_status,
            "per_config": self.per_config,
            "compute": self.compute,
            "paired": self.paired,
            "recommendation": self.recommendation,
            "unresolved": self.unresolved,
            "notes": self.notes,
        }


def _load_ok_records(store_dir: Path, config: str, task: str, seeds: List[int]) -> List[Dict[str, Any]]:
    store = PredictionStore(store_dir)
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        d = store_dir / config / task / str(seed)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name == "status.json":
                continue
            r = store.load(p)
            if r is not None and r.get("status") == "ok":
                out.append(r)
    return out


def _load_all_records(store_dir: Path, config: str, task: str, seeds: List[int]) -> List[Dict[str, Any]]:
    store = PredictionStore(store_dir)
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        d = store_dir / config / task / str(seed)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name != "status.json":
                r = store.load(p)
                if r is not None:
                    out.append(r)
    return out


def _load_unresolved(store_dir: Path, config: str, task: str, seeds: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        d = store_dir / config / task / str(seed)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name == "status.json":
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("status") == "infra_failed":
                out.append(r)
    return out


def _parse_and_score(
    task_cfg,
    model_reasoning_mode: str,
    item: Dict[str, Any],
    record: Dict[str, Any],
    declared_stop_strings: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    parser = PARSERS[task_cfg.parser]
    # Stop strings recorded at generation time MERGED with the model
    # configuration's declared stop strings: records produced before a stop
    # string was known (e.g. Gemma's <end_of_turn>) are re-scored correctly.
    stops = merge_stop_strings(record.get("eos_token_strings"), declared_stop_strings)
    parsed = parser(
        record["output_text"],
        model_reasoning_mode,
        stops,
    )
    scorer = SCORERS[task_cfg.scorer]
    score = scorer(item, parsed)
    return parsed, score


def _aggregate_musique(cfg: ConfigBundle, records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]], n_boot: int = N_BOOT_DEFAULT) -> Dict[str, Any]:
    """Per (config, task): musique metrics with clustered CIs."""
    pos_by_group: Dict[str, List[float]] = {}      # group -> answerable F1s
    pos_all_f1: Dict[str, List[float]] = {}        # group -> all positive F1s
    fa_by_group: Dict[str, List[float]] = {}       # group -> false-abstention flags
    em_by_group: Dict[str, List[float]] = {}
    neg_rev_by_group: Dict[str, List[float]] = {}  # reviewed negatives -> correct-abstention
    neg_dev_by_group: Dict[str, List[float]] = {}  # unreviewed negatives (dev only)
    relation_f1: List[float] = []
    natural_f1: List[float] = []
    n_pos = n_neg_reviewed = n_neg_unreviewed = 0
    n_pos_reviewed = n_pos_unreviewed = 0

    for r in records:
        item = items_by_id.get(r["item_id"])
        if item is None or item.get("kind") != "positive":
            continue
        n_pos += 1
        if item.get("review_status") == "approved":
            n_pos_reviewed += 1
        else:
            n_pos_unreviewed += 1
        g = r["group_id"]
        d = r["detail"]
        pos_all_f1.setdefault(g, []).append(float(r["value"]))
        em_by_group.setdefault(g, []).append(float(d.get("em", 0)))
        if d.get("answerable"):
            pos_by_group.setdefault(g, []).append(float(r["value"]))
            if item.get("syntax", {}).get("has_relation"):
                relation_f1.append(float(r["value"]))
            else:
                natural_f1.append(float(r["value"]))
        fa_by_group.setdefault(g, []).append(1.0 if d.get("false_abstention") else 0.0)

    for r in records:
        item = items_by_id.get(r["item_id"])
        if item is None or item.get("kind") != "negative":
            continue
        g = r["group_id"]
        if item.get("review_status") == "approved":
            n_neg_reviewed += 1
            neg_rev_by_group.setdefault(g, []).append(float(r["value"]))
        else:
            n_neg_unreviewed += 1
            neg_dev_by_group.setdefault(g, []).append(float(r["value"]))

    def _ci(values_by_group: Dict[str, List[float]]) -> BootstrapCI:
        return group_bootstrap(values_by_group, n_boot=n_boot)

    known = [r for r in records if items_by_id.get(r["item_id"]) is not None]
    return {
        "answer_format": _answer_format_counts(known),
        "n_positives": n_pos,
        # Positives are human-reviewed (support paragraph really answers the
        # question). Formal stages run approved positives only; smoke may
        # include unreviewed ones, which support no qualification claim.
        "n_positives_reviewed": n_pos_reviewed,
        "n_positives_unreviewed_dev_only": n_pos_unreviewed,
        "n_negatives_reviewed": n_neg_reviewed,
        "n_negatives_unreviewed_dev_only": n_neg_unreviewed,
        "answerable_f1": _ci(pos_by_group).to_dict(),
        "em_over_answerable": _ci(em_by_group).to_dict(),
        "false_abstention_rate": _ci(fa_by_group).to_dict(),
        "reviewed_negative_correct_abstention": _ci(neg_rev_by_group).to_dict(),
        "unreviewed_negative_correct_abstention_dev_only": _ci(neg_dev_by_group).to_dict(),
        "all_positive_f1_including_abstentions": _ci(pos_all_f1).to_dict(),
        "relation_vs_natural": {
            "relation": {"n": len(relation_f1), "mean_f1": sum(relation_f1) / len(relation_f1) if relation_f1 else None},
            "natural": {"n": len(natural_f1), "mean_f1": sum(natural_f1) / len(natural_f1) if natural_f1 else None},
        },
    }


def _aggregate_processbench(records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]], n_boot: int = N_BOOT_DEFAULT) -> Dict[str, Any]:
    """Official per-subset metrics + subset-structured macro CIs."""
    by_subset: Dict[str, List[Any]] = {}
    for r in records:
        item = items_by_id.get(r["item_id"])
        subset = (item or {}).get("subset") or r.get("detail", {}).get("subset") or "unknown"
        by_subset.setdefault(subset, []).append(r)

    per_subset: Dict[str, Any] = {}
    macro_inputs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
        "f1": {}, "correct_solution_acc": {}, "error_solution_acc": {}
    }
    for subset, rs in sorted(by_subset.items()):
        scored = [r for r in rs if r.get("value") is not None]
        agg = subset_error_correct_f1(scored)
        per_subset[subset] = {
            **agg,
            "n_records": len(rs),
            "n_format_errors": sum(r.get("status") == "invalid_format" for r in rs),
            "n_missing_final": sum(r.get("status") == "missing_final" for r in rs),
            "n_length_terminated": sum(r.get("finish_reason") == "length" for r in rs),
        }
        # macro inputs: per-group values tagged with kind
        for metric in macro_inputs:
            macro_inputs[metric][subset] = {}
        for r in scored:
            g = r["group_id"]
            kind = r["detail"]["kind"]
            if kind not in ("error_solution", "correct_solution"):
                continue
            code = "e" if kind == "error_solution" else "c"
            macro_inputs["f1"][subset].setdefault(g, []).append((code, float(r["value"])))
            macro_inputs["correct_solution_acc"][subset].setdefault(g, []).append((code, float(r["value"])))
            macro_inputs["error_solution_acc"][subset].setdefault(g, []).append((code, float(r["value"])))

    def subset_f1_stat(pairs: List[Tuple[str, float]]) -> Optional[float]:
        e = [v for c, v in pairs if c == "e"]
        c = [v for c, v in pairs if c == "c"]
        if not e or not c:
            return None
        a = sum(e) / len(e)
        b = sum(c) / len(c)
        return 2 * a * b / (a + b) if (a + b) > 0 else 0.0

    def subset_acc_stat(code: str):
        def fn(pairs: List[Tuple[str, float]]) -> Optional[float]:
            vals = [v for c, v in pairs if c == code]
            return sum(vals) / len(vals) if vals else None
        return fn

    return {
        "per_subset": per_subset,
        "macro_f1": macro_group_bootstrap(macro_inputs["f1"], subset_stat=subset_f1_stat, n_boot=n_boot).to_dict(),
        "macro_correct_solution_acc": macro_group_bootstrap(
            macro_inputs["correct_solution_acc"], subset_stat=subset_acc_stat("c"), n_boot=n_boot
        ).to_dict(),
        "macro_error_solution_acc": macro_group_bootstrap(
            macro_inputs["error_solution_acc"], subset_stat=subset_acc_stat("e"), n_boot=n_boot
        ).to_dict(),
    }


MATH_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
N_LOOSE_EXAMPLES = 20


def _aggregate_math(
    records: List[Dict[str, Any]],
    items_by_id: Dict[str, Dict[str, Any]],
    sampling: Optional[Dict[str, Any]] = None,
    n_boot: int = N_BOOT_DEFAULT,
) -> Dict[str, Any]:
    """MATH: official accuracy (primary), per-level accuracy, the accuracy
    re-weighted to the eligible post-exclusion pool's level proportions
    (headline for stratified samples), subject counts, answer-failure counts and the loose-equivalence
    DIAGNOSTIC (disagreements listed for spot checks; never the score)."""
    base = _aggregate_simple("math", records, n_boot=n_boot)
    sampling = sampling or {}
    by_level_group: Dict[str, Dict[str, List[float]]] = {}
    loose_by_group: Dict[str, List[float]] = {}
    subjects: Dict[str, int] = {}
    seen_items: set = set()
    disagreements: List[Dict[str, Any]] = []
    n_loose_only = n_official_only = 0
    for r in records:
        if r.get("value") is None:
            continue
        item = items_by_id.get(r["item_id"]) or {}
        level = str(item.get("level") or (item.get("gold") or {}).get("level") or "unknown")
        by_level_group.setdefault(level, {}).setdefault(r["group_id"], []).append(float(r["value"]))
        if r["item_id"] not in seen_items:
            seen_items.add(r["item_id"])
            subj = str(item.get("type") or "unknown")
            subjects[subj] = subjects.get(subj, 0) + 1
        d = r.get("detail") or {}
        loose = bool(d.get("loose_correct"))
        loose_by_group.setdefault(r["group_id"], []).append(1.0 if loose else 0.0)
        official = float(r["value"]) >= 1.0
        if loose and not official:
            n_loose_only += 1
            if len(disagreements) < N_LOOSE_EXAMPLES:
                disagreements.append({
                    "item_id": r["item_id"], "seed": r.get("seed"), "level": level,
                    "predicted": d.get("predicted_boxed"), "gold": d.get("gold_boxed"),
                })
        elif official and not loose:
            n_official_only += 1  # should not happen (loose includes official)
    per_level: Dict[str, Any] = {}
    for level in sorted(by_level_group):
        sg = by_level_group[level]
        ci = group_bootstrap(sg, n_boot=n_boot).to_dict()
        ci["n_items"] = len(sg)
        per_level[level] = ci
    weights = sampling.get("level_weights") or {}
    weighted = None
    if weights and all(l in by_level_group for l in MATH_LEVELS):
        weighted = weighted_strata_bootstrap(
            {l: by_level_group[l] for l in MATH_LEVELS}, weights, n_boot=n_boot
        ).to_dict()
    n = len(records)
    base.update({
        "accuracy_unweighted_note": (
            "plain mean over the sample; for a sample stratified with an equal number "
            "of problems per level use accuracy_level_weighted as the MATH accuracy"
        ) if weights else None,
        "accuracy_level_weighted": weighted,
        "level_weights": weights or None,
        "per_level": per_level,
        "subject_counts": dict(sorted(subjects.items())),
        "n_invalid_format": sum(1 for r in records if r.get("status") == "invalid_format"),
        "n_length_terminated": sum(1 for r in records if r.get("finish_reason") == "length"),
        "loose_accuracy_diagnostic": group_bootstrap(loose_by_group, n_boot=n_boot).to_dict(),
        "loose_vs_official": {
            "n_observations": n,
            "n_loose_correct_official_wrong": n_loose_only,
            "n_official_correct_loose_wrong": n_official_only,
            "examples": disagreements,
            "note": "diagnostic only; spot-check the examples - the official score is the MATH accuracy",
        },
    })
    return base


def _aggregate_simple(task: str, records: List[Dict[str, Any]], n_boot: int = N_BOOT_DEFAULT) -> Dict[str, Any]:
    by_group: Dict[str, List[float]] = {}
    extra: Dict[str, Any] = {}
    if task == "gsm8k":
        n = len(records)
        failures = sum(1 for r in records if r.get("status") == "invalid_format")
        relaxed_hits = sum(
            1 for r in records
            if r.get("status") == "invalid_format" and r.get("detail", {}).get("relaxed_matches_gold")
        )
        relaxed_by_group: Dict[str, List[float]] = {}
        for r in records:
            relaxed_by_group.setdefault(r["group_id"], []).append(
                1.0 if r.get("detail", {}).get("relaxed_matches_gold") else 0.0
            )
        extra = {
            "n_missing_final": sum(1 for r in records if r.get("status") == "missing_final"),
            "n_length_terminated": sum(1 for r in records if r.get("finish_reason") == "length"),
            "extraction_failure_rate": failures / n if n else None,
            "format_compliance_rate": (n - failures) / n if n else None,
            "n_extraction_failures": failures,
            "relaxed_would_rescue": relaxed_hits,
            "relaxed_accuracy": group_bootstrap(
                relaxed_by_group, n_boot=n_boot
            ).to_dict(),
            "n": n,
        }
    elif task == "math":
        extra = {
            "n": len(records),
            "n_unscorable_gold_missing": sum(1 for r in records if r.get("status") == "unscorable"),
            "n_missing_final": sum(1 for r in records if r.get("status") == "missing_final"),
        }
    elif task == "hotpotqa":
        extra = {"n": len(records), "answer_format": _answer_format_counts(records)}
    else:
        extra = {"n": len(records)}
    for r in records:
        # value is None ONLY for unscorable items; every other status
        # (missing_final, invalid_format) is an incorrect answer with value 0.
        if r.get("value") is not None:
            by_group.setdefault(r["group_id"], []).append(float(r["value"]))
    ci = group_bootstrap(by_group, n_boot=n_boot).to_dict()
    em = None
    if task == "hotpotqa":
        em_by_group: Dict[str, List[float]] = {}
        for r in records:
            if r.get("value") is not None:
                em_by_group.setdefault(r["group_id"], []).append(float(r["detail"].get("em", 0)))
        em = group_bootstrap(em_by_group, n_boot=n_boot).to_dict()
    return {"accuracy": ci, "em": em, **extra}


def build_report(
    cfg: ConfigBundle,
    plan_name: str,
    out_dir: Path,
    n_boot: int = N_BOOT_DEFAULT,
) -> ReportData:
    plan = cfg.plan(plan_name)
    seeds = cfg.splits["seeds"].get(plan_name, [0])
    store_dir = out_dir / "predictions" / plan_name
    prices = load_prices(cfg.root)
    report = ReportData(
        plan=plan_name,
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        thresholds_status=cfg.thresholds.status,
    )
    paired_inputs: Dict[
        Tuple[str, str], Dict[str, Dict[str, Dict[str, float]]]
    ] = {}
    # task -> (group_id -> stratum, stratum weights) for stratified samples
    paired_strata: Dict[str, Tuple[Dict[str, str], Dict[str, float]]] = {}

    for config_name in plan.configs:
        model_cfg = cfg.model(config_name)
        report.per_config[config_name] = {}
        report.compute[config_name] = {}
        for task in plan.tasks:
            task_cfg = cfg.task(task)
            items = manifest_items_for_task(cfg, out_dir, plan_name, task)
            runnable_items = [
                it for it in items
                if not (
                    plan.require_approved_negatives
                    and it.get("kind") == "negative"
                    and it.get("review_status") != "approved"
                )
                and not (
                    plan.require_approved_positives
                    and task == "musique_local"
                    and it.get("kind") == "positive"
                    and it.get("review_status") != "approved"
                )
            ]
            items_by_id = {it["item_id"]: it for it in items}
            all_records = _load_all_records(store_dir, config_name, task, seeds)
            records = _load_ok_records(store_dir, config_name, task, seeds)
            unresolved = _load_unresolved(store_dir, config_name, task, seeds)
            for r in unresolved:
                report.unresolved.append({**r, "config": config_name, "task": task})
            if not records:
                report.per_config[config_name][task] = {
                    "n_records": 0, "note": "no prediction records found (run `weval run --plan ...` first)"
                }
                continue
            # parse + score (deterministic, cheap; re-derived on every report)
            scored: List[Dict[str, Any]] = []
            n_parse_errors = 0
            for r in records:
                item = items_by_id.get(r["item_id"])
                if item is None:
                    continue
                try:
                    _, score = _parse_and_score(
                        task_cfg, model_cfg.reasoning_mode, item, r, model_cfg.stop_strings
                    )
                except Exception as e:  # noqa: BLE001 - a broken parser must not kill the report
                    report.notes.append(f"parse/score error for {config_name}/{task}/{r['item_id']}: {e}")
                    n_parse_errors += 1
                    continue
                scored.append({
                    **score.to_dict(),
                    "seed": r["seed"],
                    "finish_reason": r.get("finish_reason"),
                })

            if task == "musique_local":
                metrics = _aggregate_musique(cfg, scored, items_by_id, n_boot=n_boot)
            elif task == "processbench":
                metrics = _aggregate_processbench(scored, items_by_id, n_boot=n_boot)
            elif task == "math":
                sampling = manifest_sampling(out_dir, plan_name, task)
                metrics = _aggregate_math(scored, items_by_id, sampling, n_boot=n_boot)
                if sampling:
                    metrics["sampling"] = {k: v for k, v in sampling.items() if k != "excluded"}
                if sampling.get("level_weights"):
                    paired_strata["math"] = (
                        {it["group_id"]: str(it.get("level")) for it in items},
                        sampling["level_weights"],
                    )
            else:
                metrics = _aggregate_simple(task, scored, n_boot=n_boot)
            per_seed: Dict[str, Any] = {}
            for seed in seeds:
                seed_scored = [s for s in scored if s["seed"] == seed]
                if task == "musique_local":
                    per_seed[str(seed)] = _aggregate_musique(
                        cfg, seed_scored, items_by_id, n_boot=0
                    )
                elif task == "processbench":
                    per_seed[str(seed)] = _aggregate_processbench(
                        seed_scored, items_by_id, n_boot=0
                    )
                elif task == "math":
                    per_seed[str(seed)] = _aggregate_math(
                        seed_scored, items_by_id, manifest_sampling(out_dir, plan_name, task), n_boot=0
                    )
                else:
                    per_seed[str(seed)] = _aggregate_simple(task, seed_scored, n_boot=0)
            metrics["n_records"] = len(records)
            metrics["seeds"] = seeds
            metrics["per_seed"] = per_seed
            gen_tokens = [int(r.get("n_generated_tokens") or 0) for r in records]
            metrics["mean_generated_tokens"] = (sum(gen_tokens) / len(gen_tokens)) if gen_tokens else None
            expected_observations = len(runnable_items) * len(seeds)
            observed_keys = {
                (str(r.get("item_id")), int(r.get("seed", -1)))
                for r in all_records
            }
            metrics["completion"] = {
                "expected_observations": expected_observations,
                "recorded_observations": len(observed_keys),
                "ok_observations": len(records),
                "non_ok_observations": len(all_records) - len(records),
                "parse_score_errors": n_parse_errors,
                "complete": (
                    len(observed_keys) == expected_observations
                    and len(records) == expected_observations
                    and n_parse_errors == 0
                ),
            }
            report.per_config[config_name][task] = metrics

            # paired-difference inputs: group -> scored values (any nonzero status
            # with a value is an evaluated outcome; None = unscorable only)
            by_group: Dict[str, Dict[str, float]] = {}
            for s in scored:
                if s.get("value") is not None:
                    observation = f"{s['item_id']}::seed={s['seed']}"
                    by_group.setdefault(s["group_id"], {})[observation] = float(s["value"])
            if task == "musique_local":
                metric_groups: Dict[str, Dict[str, Dict[str, float]]] = {
                    "answerable_f1": {},
                    "false_abstention_rate": {},
                    "reviewed_negative_correct_abstention": {},
                    "unreviewed_negative_correct_abstention_dev_only": {},
                }
                for s in scored:
                    item = items_by_id.get(s["item_id"])
                    if item is None:
                        continue
                    observation = f"{s['item_id']}::seed={s['seed']}"
                    group = s["group_id"]
                    detail = s.get("detail") or {}
                    if item.get("kind") == "positive":
                        metric_groups["answerable_f1"].setdefault(group, {})[observation] = float(s["value"])
                        metric_groups["false_abstention_rate"].setdefault(group, {})[observation] = (
                            1.0 if detail.get("false_abstention") else 0.0
                        )
                    elif item.get("review_status") == "approved":
                        metric_groups["reviewed_negative_correct_abstention"].setdefault(group, {})[observation] = float(s["value"])
                    else:
                        metric_groups["unreviewed_negative_correct_abstention_dev_only"].setdefault(group, {})[observation] = float(s["value"])
                paired_inputs[(config_name, task)] = metric_groups
            else:
                paired_inputs[(config_name, task)] = {"primary": by_group}

            # ---- compute metrics ----
            c_items: List[Dict[str, Any]] = []
            n_text = _load_n_text(out_dir, config_name)
            for r in records:
                c = None
                if n_text is not None:
                    c = 2.0 * n_text * (r["n_prompt_tokens"] + r["n_generated_tokens"])
                c_items.append({
                    "kind": r.get("kind", "na"),
                    "cproxy": c,
                    "tin": r["n_prompt_tokens"],
                    "tout": r["n_generated_tokens"],
                    "value": None,
                })
            # align scores for efficiency (same-task, scored records only)
            by_id_seed = {(s["item_id"], s["seed"]): s for s in scored}
            eff_scores: List[float] = []
            eff_cs: List[float] = []
            for r, ci in zip(records, c_items):
                s = by_id_seed.get((r["item_id"], r["seed"]))
                if s is not None and s.get("value") is not None and ci["cproxy"] is not None:
                    eff_scores.append(float(s["value"]))
                    eff_cs.append(ci["cproxy"])
            agg = aggregate_cproxy(c_items)
            agg["n_text"] = n_text
            agg["n_text_source"] = "artifacts/model_params/{config}.json" if n_text is not None else None
            agg["cweighted_usd"] = None
            if prices is not None:
                tin = sum(r["n_prompt_tokens"] for r in records)
                tout = sum(r["n_generated_tokens"] for r in records)
                agg["cweighted_usd"] = cweighted(tin, tout, prices)
                agg["prices"] = prices.to_dict()
            agg["efficiency_compute"] = efficiency_compute(eff_scores, eff_cs)
            report.compute[config_name][task] = agg

    # ---- paired differences (same items across configs) ----
    report.paired = _paired_differences(plan, paired_inputs, n_boot, paired_strata)

    # ---- recommendation ----
    report.recommendation = _recommendation(cfg, plan, report, out_dir)
    if prices is None:
        report.notes.append("Cweighted is null: no sourced prices (configs/prices.yaml absent). Never zeroed or invented.")
    if report.unresolved and plan.report_incomplete_on_unresolved:
        report.notes.append(
            f"{len(report.unresolved)} item(s) have unresolved infrastructure failures; "
            "the affected comparisons are INCOMPLETE, not merely low."
        )
    return report


def _load_n_text(out_dir: Path, config_name: str) -> Optional[int]:
    p = out_dir / "model_params" / f"{config_name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    n = d.get("n_text")
    return int(n) if n is not None else None


def _paired_differences(
    plan: PlanCfg,
    paired_inputs: Dict[
        Tuple[str, str], Dict[str, Dict[str, Dict[str, float]]]
    ],
    n_boot: int,
    strata: Optional[Dict[str, Tuple[Dict[str, str], Dict[str, float]]]] = None,
) -> Dict[str, Any]:
    """Paired A - B differences per task and metric. For a task sampled by
    strata (MATH in math_solve) the primary metric is also given re-weighted to
    the eligible-pool stratum proportions ("accuracy_level_weighted")."""
    configs = list(plan.configs)
    strata = strata or {}
    out: Dict[str, Any] = {}
    for task in plan.tasks:
        metric_names = sorted({
            metric
            for c in configs
            for metric in paired_inputs.get((c, task), {}).keys()
        })
        out[task] = {}
        for metric in metric_names:
            per_config = {
                c: paired_inputs.get((c, task), {}).get(metric, {})
                for c in configs
            }
            pairs: Dict[str, Any] = {}
            for i in range(len(configs)):
                for j in range(i + 1, len(configs)):
                    a, b = configs[i], configs[j]
                    ci, missing_a, missing_b = paired_difference(
                        per_config[a], per_config[b], n_boot=n_boot
                    )
                    pairs[f"{a} - {b}"] = {
                        **ci.to_dict(),
                        "n_observations_missing_in_a": missing_a,
                        "n_observations_missing_in_b": missing_b,
                    }
            out[task][metric] = pairs
        if task in strata and "primary" in out[task]:
            stratum_of, weights = strata[task]
            primary = {c: paired_inputs.get((c, task), {}).get("primary", {}) for c in configs}
            wpairs: Dict[str, Any] = {}
            for i in range(len(configs)):
                for j in range(i + 1, len(configs)):
                    a, b = configs[i], configs[j]
                    key = f"{a} - {b}"
                    wpairs[key] = {
                        **_weighted_paired_difference(
                            primary[a], primary[b], stratum_of, weights, n_boot
                        ).to_dict(),
                        "n_observations_missing_in_a": out[task]["primary"][key]["n_observations_missing_in_a"],
                        "n_observations_missing_in_b": out[task]["primary"][key]["n_observations_missing_in_b"],
                    }
            out[task]["accuracy_level_weighted"] = wpairs
    return out


def _weighted_paired_difference(
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    stratum_of: Dict[str, str],
    weights: Dict[str, float],
    n_boot: int,
) -> BootstrapCI:
    """Post-stratified paired difference: per-stratum mean of the aligned
    (same item, same seed) A - B deltas, weighted by the declared pool's stratum
    proportions; groups resampled within their stratum."""
    deltas: Dict[str, Dict[str, List[float]]] = {}
    for g in sorted(set(a) & set(b)):
        keys = sorted(set(a[g]) & set(b[g]))
        s = stratum_of.get(g)
        if not keys or s is None:
            continue
        deltas.setdefault(s, {})[g] = [a[g][k] - b[g][k] for k in keys]
    if not all(s in deltas for s, w in weights.items() if w > 0):
        # a stratum without paired data cannot be re-weighted honestly
        return BootstrapCI(None, None, None, n_boot, 0, 0, BOOTSTRAP_SEED_DEFAULT)
    return weighted_strata_bootstrap(deltas, weights, n_boot=n_boot)


def _evidence_meets(e: Dict[str, Any], ew: Dict[str, Any]) -> bool:
    """Quality thresholds of the evidence-worker role (point estimates)."""
    return (
        e["answerable_f1"] is not None and e["false_abstention_rate"] is not None
        and e["reviewed_negative_correct_abstention"] is not None
        and e["answerable_f1"] >= ew["answerable_f1_min"]
        and e["false_abstention_rate"] <= ew["false_abstention_max"]
        and e["reviewed_negative_correct_abstention"] >= ew["reviewed_negative_abstention_min"]
    )


def _math_meets(m: Dict[str, Any], mc: Dict[str, Any]) -> bool:
    """Quality thresholds of the math-checker role (point estimates)."""
    return (
        m["macro_f1"] is not None and m["macro_correct_solution_acc"] is not None
        and m["macro_error_solution_acc"] is not None
        and m["macro_f1"] >= mc["macro_f1_min"]
        and m["macro_correct_solution_acc"] >= mc["macro_correct_solution_acc_min"]
        and m["macro_error_solution_acc"] >= mc["macro_error_solution_acc_min"]
    )


def _selection_nominees(out_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    """The nomination recorded in reports/selection/report.json (None if that
    report does not exist or nominates nothing)."""
    p = out_dir / "reports" / "selection" / "report.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    inner = (data.get("recommendation") or {}).get("recommendation") or {}
    return {
        "evidence_worker": inner.get("evidence_worker"),
        "math_checker": inner.get("math_checker"),
    }


def _holdout_check(
    out_dir: Path,
    evidence: List[Dict[str, Any]],
    math_ck: List[Dict[str, Any]],
    ew: Dict[str, Any],
    mc: Dict[str, Any],
) -> Dict[str, Any]:
    """Pass/fail of the SELECTION nominee of each role on the holdout data.
    Never nominates or re-ranks: a config that was not nominated at selection
    is not looked at, and a failing nominee is reported as failing."""
    nominees = _selection_nominees(out_dir)
    if nominees is None:
        return {
            "available": False,
            "note": "reports/selection/report.json not found: there is no selection nominee to check",
        }
    out: Dict[str, Any] = {"available": True, "source": "reports/selection/report.json"}
    for role, rows, meets, thr_role in (
        ("evidence_worker", evidence, _evidence_meets, ew),
        ("math_checker", math_ck, _math_meets, mc),
    ):
        nominee = nominees.get(role)
        if nominee is None:
            out[role] = {"nominee": None, "result": "no_selection_nominee"}
            continue
        row = next((r for r in rows if r["config"] == nominee), None)
        if row is None or not row["complete"]:
            out[role] = {"nominee": nominee, "result": "incomplete"}
            continue
        out[role] = {"nominee": nominee, "result": "pass" if meets(row, thr_role) else "fail"}
    return out


def _recommendation(cfg: ConfigBundle, plan: PlanCfg, report: ReportData, out_dir: Path) -> Dict[str, Any]:
    """Role qualification from thresholds.yaml.

    Only the SELECTION stage nominates, and only when thresholds.status ==
    "confirmed":

      * selection - qualified configs, lowest-Cproxy nominee per role
                    (mode "selection_nomination");
      * holdout   - pass/fail of the selection nominee on holdout data; never
                    nominates or re-picks (mode "holdout_check");
      * smoke / pilot / reference - candidate metrics are shown next to the
                    thresholds for information only; recommendation is None
                    (mode "informational_no_nomination"), whatever the
                    thresholds status.

    With unconfirmed thresholds, selection/holdout show the draft values and
    nominate/check nothing (mode "gated_unconfirmed_thresholds").
    """
    thr = cfg.thresholds
    evidence: List[Dict[str, Any]] = []
    math_ck: List[Dict[str, Any]] = []
    for config_name in plan.configs:
        m = report.per_config.get(config_name, {})
        mu = m.get("musique_local") or {}
        pb = m.get("processbench") or {}
        comp = report.compute.get(config_name, {})
        mu_comp = comp.get("musique_local") or {}
        pb_comp = comp.get("processbench") or {}

        def _est(d: Dict[str, Any]) -> Optional[float]:
            return d.get("estimate") if isinstance(d, dict) else None

        evidence.append({
            "config": config_name,
            "answerable_f1": _est(mu.get("answerable_f1")),
            "false_abstention_rate": _est(mu.get("false_abstention_rate")),
            "reviewed_negative_correct_abstention": _est(mu.get("reviewed_negative_correct_abstention")),
            "mean_cproxy_role_workload_50_50": _role_cproxy(mu_comp),
            "n_positives": mu.get("n_positives"),
            "n_negatives_reviewed": mu.get("n_negatives_reviewed"),
            "complete": bool((mu.get("completion") or {}).get("complete")),
        })
        math_ck.append({
            "config": config_name,
            "macro_f1": _est(pb.get("macro_f1")),
            "macro_correct_solution_acc": _est(pb.get("macro_correct_solution_acc")),
            "macro_error_solution_acc": _est(pb.get("macro_error_solution_acc")),
            "mean_cproxy": pb_comp.get("mean"),
            "complete": bool((pb.get("completion") or {}).get("complete")),
        })

    result: Dict[str, Any] = {
        "stage": plan.stage,
        "thresholds": {
            "status": thr.status,
            "evidence_worker": thr.evidence_worker,
            "math_checker": thr.math_checker,
            "tie_break": thr.tie_break,
        },
        "evidence_worker_candidates": evidence,
        "math_checker_candidates": math_ck,
    }
    if plan.stage not in (NOMINATING_STAGE, HOLDOUT_STAGE):
        result["mode"] = "informational_no_nomination"
        result["recommendation"] = None
        result["note"] = (
            f"stage {plan.stage!r} is not the selection stage: candidate metrics are "
            "shown next to the thresholds for information only and NO worker is "
            "nominated. Only `weval report --plan selection` nominates."
        )
        return result
    if thr.status != "confirmed":
        result["mode"] = "gated_unconfirmed_thresholds"
        result["recommendation"] = None
        result["note"] = (
            "thresholds.yaml status is not 'confirmed'; draft values are shown "
            "and NO worker is nominated. Confirm the thresholds config before selection."
        )
        return result

    ew = thr.evidence_worker
    mc = thr.math_checker
    if plan.stage == HOLDOUT_STAGE:
        result["mode"] = "holdout_check"
        result["recommendation"] = None
        result["holdout_check"] = _holdout_check(out_dir, evidence, math_ck, ew, mc)
        result["note"] = (
            "holdout checks the selection nominee of each role against the same "
            "thresholds; it never nominates or re-picks. A failing nominee is "
            "reported as failing."
        )
        return result

    result["mode"] = "selection_nomination"
    ev_qualified = [
        e for e in evidence
        if e["complete"] and e["mean_cproxy_role_workload_50_50"] is not None
        and _evidence_meets(e, ew)
    ]
    mc_qualified = [
        m for m in math_ck
        if m["complete"] and m["mean_cproxy"] is not None
        and _math_meets(m, mc)
    ]
    result["evidence_worker_qualified"] = sorted(ev_qualified, key=lambda e: (e["mean_cproxy_role_workload_50_50"] is None, e["mean_cproxy_role_workload_50_50"] or 0.0))
    result["math_checker_qualified"] = sorted(mc_qualified, key=lambda m: (m["mean_cproxy"] is None, m["mean_cproxy"] or 0.0))
    result["recommendation"] = {
        "evidence_worker": result["evidence_worker_qualified"][0]["config"] if result["evidence_worker_qualified"] else None,
        "math_checker": result["math_checker_qualified"][0]["config"] if result["math_checker_qualified"] else None,
        "note": (
            "lowest mean Cproxy among qualified configurations (50/50 declared "
            "mix for the evidence workload); no winner is forced when nobody "
            "qualifies. Holdout runs only for the nominated candidate."
        ),
    }
    return result


def _role_cproxy(mu_comp: Dict[str, Any]) -> Optional[float]:
    """Evidence-role workload Cproxy under the DECLARED 50/50 mix assumption
    (an assumption for planning, not a prevalence claim)."""
    by_kind = mu_comp.get("by_kind") or {}
    pos = by_kind.get("positive") or {}
    neg = by_kind.get("negative") or {}
    m_pos, m_neg = pos.get("mean"), neg.get("mean")
    if m_pos is None and m_neg is None:
        return None
    if m_pos is None or m_neg is None:
        return None
    return 0.5 * m_pos + 0.5 * m_neg


def write_report(out_dir: Path, plan_name: str, report: ReportData) -> Path:
    rdir = out_dir / "reports" / plan_name
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return path
