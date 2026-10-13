"""Report building: parse -> score -> aggregate -> statistics -> recommendation.

Reads raw prediction records (never regenerating anything), re-derives parsed
answers and scores with the pinned parser/scorer versions, then computes:

  * per-seed metrics and cross-seed means,
  * 95% clustered bootstrap CIs (parent problem groups; subset structure
    preserved for ProcessBench macro metrics),
  * paired config differences on identical items,
  * Cproxy / Cweighted / Efficiency_compute,
  * role qualification against thresholds.yaml - recommendations are produced
    ONLY when thresholds.status == "confirmed"; otherwise the draft values
    are shown without a recommendation.

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
from ..run.manifests import manifest_items_for_task
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
)

REPORT_VERSION = "report-v1"


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
    task_cfg, model_reasoning_mode: str, item: Dict[str, Any], record: Dict[str, Any]
) -> Tuple[Any, Any]:
    parser = PARSERS[task_cfg.parser]
    parsed = parser(
        record["output_text"],
        model_reasoning_mode,
        record.get("eos_token_strings"),
    )
    scorer = SCORERS[task_cfg.scorer]
    score = scorer(item, parsed)
    return parsed, score


def _aggregate_musique(cfg: ConfigBundle, records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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

    for r in records:
        item = items_by_id.get(r["item_id"])
        if item is None or item.get("kind") != "positive":
            continue
        n_pos += 1
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
        return group_bootstrap(values_by_group)

    return {
        "n_positives": n_pos,
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


def _aggregate_processbench(records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
            "n_unscored_format_errors": len(rs) - len(scored),
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
        "macro_f1": macro_group_bootstrap(macro_inputs["f1"], subset_stat=subset_f1_stat).to_dict(),
        "macro_correct_solution_acc": macro_group_bootstrap(
            macro_inputs["correct_solution_acc"], subset_stat=subset_acc_stat("c")
        ).to_dict(),
        "macro_error_solution_acc": macro_group_bootstrap(
            macro_inputs["error_solution_acc"], subset_stat=subset_acc_stat("e")
        ).to_dict(),
    }


def _aggregate_simple(task: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_group: Dict[str, List[float]] = {}
    extra: Dict[str, Any] = {}
    if task == "gsm8k":
        n = len(records)
        failures = sum(1 for r in records if r.get("status") == "invalid_format")
        relaxed_hits = sum(
            1 for r in records
            if r.get("status") == "invalid_format" and r.get("detail", {}).get("relaxed_matches_gold")
        )
        extra = {
            "extraction_failure_rate": failures / n if n else None,
            "n_extraction_failures": failures,
            "relaxed_would_rescue": relaxed_hits,
            "n": n,
        }
    elif task == "math":
        extra = {
            "n": len(records),
            "n_unscorable_gold_missing": sum(1 for r in records if r.get("status") == "unscorable"),
            "n_missing_final": sum(1 for r in records if r.get("status") == "missing_final"),
        }
    else:
        extra = {"n": len(records)}
    for r in records:
        # value is None ONLY for unscorable items; every other status
        # (missing_final, invalid_format) is an incorrect answer with value 0.
        if r.get("value") is not None:
            by_group.setdefault(r["group_id"], []).append(float(r["value"]))
    ci = group_bootstrap(by_group).to_dict()
    em = None
    if task == "hotpotqa":
        em_by_group: Dict[str, List[float]] = {}
        for r in records:
            if r.get("value") is not None:
                em_by_group.setdefault(r["group_id"], []).append(float(r["detail"].get("em", 0)))
        em = group_bootstrap(em_by_group).to_dict()
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
    scored_by_group: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for config_name in plan.configs:
        model_cfg = cfg.model(config_name)
        report.per_config[config_name] = {}
        report.compute[config_name] = {}
        for task in plan.tasks:
            task_cfg = cfg.task(task)
            items = manifest_items_for_task(cfg, out_dir, plan_name, task)
            items_by_id = {it["item_id"]: it for it in items}
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
            for r in records:
                item = items_by_id.get(r["item_id"])
                if item is None:
                    continue
                try:
                    _, score = _parse_and_score(task_cfg, model_cfg.reasoning_mode, item, r)
                except Exception as e:  # noqa: BLE001 - a broken parser must not kill the report
                    report.notes.append(f"parse/score error for {config_name}/{task}/{r['item_id']}: {e}")
                    continue
                scored.append({**score.to_dict(), "seed": r["seed"]})

            if task == "musique_local":
                metrics = _aggregate_musique(cfg, scored, items_by_id)
            elif task == "processbench":
                metrics = _aggregate_processbench(scored, items_by_id)
            else:
                metrics = _aggregate_simple(task, scored)
            metrics["n_records"] = len(records)
            metrics["seeds"] = seeds
            report.per_config[config_name][task] = metrics

            # paired-difference inputs: group -> scored values (any nonzero status
            # with a value is an evaluated outcome; None = unscorable only)
            by_group: Dict[str, List[float]] = {}
            for s in scored:
                if s.get("value") is not None:
                    by_group.setdefault(s["group_id"], []).append(float(s["value"]))
            scored_by_group[(config_name, task)] = by_group

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
            by_id = {s["item_id"]: s for s in scored}
            eff_scores: List[float] = []
            eff_cs: List[float] = []
            for r, ci in zip(records, c_items):
                s = by_id.get(r["item_id"])
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
    report.paired = _paired_differences(plan, scored_by_group, n_boot)

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
    scored_by_group: Dict[Tuple[str, str], Dict[str, List[float]]],
    n_boot: int,
) -> Dict[str, Any]:
    configs = list(plan.configs)
    out: Dict[str, Any] = {}
    for task in plan.tasks:
        per_config = {c: scored_by_group.get((c, task), {}) for c in configs}
        pairs: Dict[str, Any] = {}
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                a, b = configs[i], configs[j]
                ci, missing_a, missing_b = paired_difference(
                    per_config[a], per_config[b], n_boot=n_boot
                )
                pairs[f"{a} - {b}"] = {
                    **ci.to_dict(),
                    "n_groups_missing_in_a": missing_a,
                    "n_groups_missing_in_b": missing_b,
                }
        out[task] = pairs
    return out


def _recommendation(cfg: ConfigBundle, plan: PlanCfg, report: ReportData, out_dir: Path) -> Dict[str, Any]:
    """Role qualification from thresholds.yaml.

    Recommendations are produced ONLY when thresholds.status == "confirmed".
    Otherwise the draft values are echoed and no candidate is nominated.
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
        })
        math_ck.append({
            "config": config_name,
            "macro_f1": _est(pb.get("macro_f1")),
            "macro_correct_solution_acc": _est(pb.get("macro_correct_solution_acc")),
            "macro_error_solution_acc": _est(pb.get("macro_error_solution_acc")),
            "mean_cproxy": pb_comp.get("mean"),
        })

    result: Dict[str, Any] = {
        "thresholds": {
            "status": thr.status,
            "evidence_worker": thr.evidence_worker,
            "math_checker": thr.math_checker,
            "tie_break": thr.tie_break,
        },
        "evidence_worker_candidates": evidence,
        "math_checker_candidates": math_ck,
    }
    if thr.status != "confirmed":
        result["recommendation"] = None
        result["note"] = (
            "thresholds.yaml status is not 'confirmed'; draft values are shown "
            "and NO worker is nominated. Confirm the thresholds config before selection."
        )
        return result

    ew = thr.evidence_worker
    mc = thr.math_checker
    ev_qualified = [
        e for e in evidence
        if e["answerable_f1"] is not None and e["false_abstention_rate"] is not None
        and e["reviewed_negative_correct_abstention"] is not None
        and e["answerable_f1"] >= ew["answerable_f1_min"]
        and e["false_abstention_rate"] <= ew["false_abstention_max"]
        and e["reviewed_negative_correct_abstention"] >= ew["reviewed_negative_abstention_min"]
    ]
    mc_qualified = [
        m for m in math_ck
        if m["macro_f1"] is not None and m["macro_correct_solution_acc"] is not None
        and m["macro_error_solution_acc"] is not None
        and m["macro_f1"] >= mc["macro_f1_min"]
        and m["macro_correct_solution_acc"] >= mc["macro_correct_solution_acc_min"]
        and m["macro_error_solution_acc"] >= mc["macro_error_solution_acc_min"]
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
        return m_pos if m_pos is not None else m_neg
    return 0.5 * m_pos + 0.5 * m_neg


def write_report(out_dir: Path, plan_name: str, report: ReportData) -> Path:
    rdir = out_dir / "reports" / plan_name
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return path
