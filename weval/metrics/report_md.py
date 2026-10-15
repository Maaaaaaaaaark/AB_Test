"""Markdown rendering of ReportData -> artifacts/reports/{plan}/report.md"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .report import ReportData


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def _ci_cell(d: Optional[Dict[str, Any]]) -> str:
    if not isinstance(d, dict) or d.get("estimate") is None:
        return "-"
    lo, hi = d.get("ci95_lo"), d.get("ci95_hi")
    if lo is None or hi is None:
        return _fmt(d["estimate"], 3)
    return f"{d['estimate']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_rate(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.1f}%"


MATH_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]


def _cell_text(s: Any, limit: int = 80) -> str:
    """Model / gold text inside a markdown table cell (pipes escaped, one line)."""
    if s is None:
        return "-"
    t = " ".join(str(s).split())
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return "`" + t.replace("|", "\\|").replace("`", "'") + "`"


def _est(d: Optional[Dict[str, Any]]) -> Optional[float]:
    return d.get("estimate") if isinstance(d, dict) else None


def _no_answer(m: Dict[str, Any]) -> Optional[int]:
    if not isinstance(m, dict) or "n_records" not in m:
        return None
    return int(m.get("n_missing_final") or 0) + int(
        m.get("n_extraction_failures") if "n_extraction_failures" in m else (m.get("n_invalid_format") or 0)
    )


def _math_solve_summary(report: ReportData) -> List[str]:
    """Cross-config overview for the single-model math solving plan."""
    lines: List[str] = []
    lines.append("## Summary - single-model math solving (descriptive)")
    lines.append("")
    lines.append(
        "Each number is one model solving each problem on its own (no other agent). "
        "This plan nominates nothing and has no thresholds. MATH accuracy (headline) is "
        "re-weighted to the eligible post-exclusion pool's level proportions because the sample has an equal "
        "number of problems per level; the plain sample mean is shown next to it. "
        "Loose accuracy is a diagnostic second opinion on answer equivalence, never the score."
    )
    lines.append("")
    lines.append("| config | GSM8K accuracy | MATH accuracy (level-weighted) | MATH unweighted | MATH loose (diagnostic) | complete |")
    lines.append("|---|---|---|---|---|---|")
    for config, tasks in report.per_config.items():
        g = tasks.get("gsm8k") or {}
        m = tasks.get("math") or {}
        complete = all(
            (t.get("completion") or {}).get("complete") for t in (g, m) if "n_records" in t
        ) and "n_records" in g and "n_records" in m
        lines.append(
            f"| {config} | {_ci_cell(g.get('accuracy'))} | {_ci_cell(m.get('accuracy_level_weighted'))} | "
            f"{_ci_cell(m.get('accuracy'))} | {_fmt(_est(m.get('loose_accuracy_diagnostic')), 3)} | "
            f"{'yes' if complete else '**no**'} |"
        )
    lines.append("")
    lines.append("| config | GSM8K no answer | GSM8K length stops | MATH no answer | MATH length stops | mean output tokens GSM8K | mean output tokens MATH |")
    lines.append("|---|---|---|---|---|---|---|")
    for config, tasks in report.per_config.items():
        g = tasks.get("gsm8k") or {}
        m = tasks.get("math") or {}

        def _n(x: Optional[int], d: Dict[str, Any]) -> str:
            if x is None:
                return "-"
            return f"{x} / {d.get('n_records')}"

        lines.append(
            f"| {config} | {_n(_no_answer(g), g)} | {_n(g.get('n_length_terminated'), g) if 'n_records' in g else '-'} | "
            f"{_n(_no_answer(m), m)} | {_n(m.get('n_length_terminated'), m) if 'n_records' in m else '-'} | "
            f"{_fmt(g.get('mean_generated_tokens'), 0)} | {_fmt(m.get('mean_generated_tokens'), 0)} |"
        )
    lines.append("")
    lines.append("No answer = no final answer extracted (GSM8K: no `#### <number>` line; MATH: no `\\boxed{}`; "
                 "or the reasoning never closed) - scored as wrong. Counts are observations (problems x seeds).")
    lines.append("")
    lines.append("### MATH accuracy by level")
    lines.append("")
    lines.append("| config | " + " | ".join(MATH_LEVELS) + " |")
    lines.append("|---|" + "---|" * len(MATH_LEVELS))
    for config, tasks in report.per_config.items():
        per_level = (tasks.get("math") or {}).get("per_level") or {}
        lines.append(f"| {config} | " + " | ".join(_ci_cell(per_level.get(l)) for l in MATH_LEVELS) + " |")
    lines.append("")
    return lines


def _math_details(metrics: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if metrics.get("accuracy_level_weighted"):
        lines.append(f"| accuracy, level-weighted (headline) | {_ci_cell(metrics.get('accuracy_level_weighted'))} | - |")
    lines.append(f"| loose accuracy (diagnostic only) | {_ci_cell(metrics.get('loose_accuracy_diagnostic'))} | - |")
    lines.append(f"| unscorable (gold missing boxed) | {metrics.get('n_unscorable_gold_missing')} | - |")
    lines.append(f"| missing final answer (open region / no boxed) | {metrics.get('n_missing_final')} | - |")
    lines.append(f"| invalid format | {metrics.get('n_invalid_format')} | - |")
    lines.append(f"| length stops (hit max_new_tokens) | {metrics.get('n_length_terminated')} | - |")
    per_level = metrics.get("per_level") or {}
    weights = metrics.get("level_weights") or {}
    if per_level:
        lines.append("")
        lines.append("| level | problems | accuracy [95% CI] | weight (eligible pool) |")
        lines.append("|---|---|---|---|")
        for level in sorted(per_level):
            d = per_level[level]
            w = weights.get(level)
            lines.append(f"| {level} | {d.get('n_items')} | {_ci_cell(d)} | {'-' if w is None else f'{w:.3f}'} |")
    subjects = metrics.get("subject_counts") or {}
    if subjects:
        lines.append("")
        lines.append("- subjects (problems scored): " + ", ".join(f"{k} {v}" for k, v in subjects.items()))
    samp = metrics.get("sampling") or {}
    if samp:
        lines.append(
            f"- sample: {samp.get('design')}, {samp.get('n_sampled')} of {samp.get('population')} test problems "
            f"(seed {samp.get('seed')}); skipped while drawing: {samp.get('excluded_by_reason') or {}}"
        )
    lv = metrics.get("loose_vs_official") or {}
    if lv:
        lines.append(
            f"- loose vs official: {lv.get('n_loose_correct_official_wrong')} observation(s) loose-correct but "
            f"officially wrong (of {lv.get('n_observations')}). {lv.get('note')}"
        )
        ex = lv.get("examples") or []
        if ex:
            lines.append("")
            lines.append("| item | seed | level | predicted | gold |")
            lines.append("|---|---|---|---|---|")
            for e in ex:
                lines.append(
                    f"| {e.get('item_id')} | {e.get('seed')} | {e.get('level')} | "
                    f"{_cell_text(e.get('predicted'))} | {_cell_text(e.get('gold'))} |"
                )
    return lines


def render_markdown(report: ReportData) -> str:
    lines: List[str] = []
    lines.append(f"# Evaluation report - plan: {report.plan}")
    lines.append("")
    lines.append(f"- generated: {report.created_at_utc} (UTC)")
    lines.append(f"- thresholds status: **{report.thresholds_status}**")
    lines.append("")
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for n in report.notes:
            lines.append(f"- {n}")
        lines.append("")

    if report.plan == "math_solve":
        lines.extend(_math_solve_summary(report))

    for config, tasks in report.per_config.items():
        lines.append(f"## {config}")
        lines.append("")
        for task, metrics in tasks.items():
            if not isinstance(metrics, dict) or "n_records" not in metrics:
                lines.append(f"### {task}: {metrics.get('note', 'no data')}")
                lines.append("")
                continue
            lines.append(f"### {task} (n={metrics.get('n_records')}, seeds={metrics.get('seeds')})")
            lines.append("")
            completion = metrics.get("completion") or {}
            lines.append(
                "- comparison completeness: "
                f"**{'complete' if completion.get('complete') else 'INCOMPLETE'}** "
                f"(expected={completion.get('expected_observations')}, "
                f"ok={completion.get('ok_observations')}, "
                f"non-ok={completion.get('non_ok_observations')}, "
                f"parse errors={completion.get('parse_score_errors')})"
            )
            lines.append("")
            if task == "musique_local":
                lines.append("| metric | estimate | 95% CI |")
                lines.append("|---|---|---|")
                for label, key in [
                    ("answerable F1 (qualification)", "answerable_f1"),
                    ("false-abstention rate (qualification)", "false_abstention_rate"),
                    ("reviewed negative correct-abstention (qualification)", "reviewed_negative_correct_abstention"),
                    ("EM over answerable", "em_over_answerable"),
                    ("all-positive F1 incl. abstentions", "all_positive_f1_including_abstentions"),
                    ("unreviewed negative correct-abstention (dev only)", "unreviewed_negative_correct_abstention_dev_only"),
                ]:
                    lines.append(f"| {label} | {_ci_cell(metrics.get(key))} | - |")
                rn = metrics.get("relation_vs_natural") or {}
                lines.append("")
                lines.append(f"- positives: {metrics.get('n_positives')} (reviewed {metrics.get('n_positives_reviewed')}, "
                             f"unreviewed dev only {metrics.get('n_positives_unreviewed_dev_only')}), "
                             f"reviewed negatives: {metrics.get('n_negatives_reviewed')}, "
                             f"unreviewed negatives (dev only): {metrics.get('n_negatives_unreviewed_dev_only')}")
                lines.append(f"- relation questions: n={rn.get('relation', {}).get('n')}, mean F1={_fmt(rn.get('relation', {}).get('mean_f1'), 3)}; "
                             f"natural questions: n={rn.get('natural', {}).get('n')}, mean F1={_fmt(rn.get('natural', {}).get('mean_f1'), 3)}")
                af = metrics.get("answer_format") or {}
                lines.append(f"- answer format: Answer-line={af.get('answer_line')}, single-line={af.get('single_line')}, "
                             f"inline-marker={af.get('inline_marker')}, "
                             f"invalid format={af.get('invalid_format')}, missing final={af.get('missing_final')}")
            elif task == "processbench":
                lines.append("| subset | error acc | correct acc | F1 | n_error | n_correct | format errors | missing final | length stops |")
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for subset, s in (metrics.get("per_subset") or {}).items():
                    lines.append(
                        f"| {subset} | {_fmt_rate(s.get('error_acc'))} | {_fmt_rate(s.get('correct_acc'))} | "
                        f"{_fmt_rate(s.get('f1'))} | {s.get('n_error')} | {s.get('n_correct')} | "
                        f"{s.get('n_format_errors')} | {s.get('n_missing_final')} | {s.get('n_length_terminated')} |"
                    )
                lines.append("")
                lines.append("| macro metric (subset-structured, qualification) | estimate | 95% CI |")
                lines.append("|---|---|---|")
                lines.append(f"| macro F1 | {_ci_cell(metrics.get('macro_f1'))} | - |")
                lines.append(f"| macro correct-solution acc | {_ci_cell(metrics.get('macro_correct_solution_acc'))} | - |")
                lines.append(f"| macro error-solution acc | {_ci_cell(metrics.get('macro_error_solution_acc'))} | - |")
            else:
                lines.append("| metric | estimate | 95% CI |")
                lines.append("|---|---|---|")
                acc_label = "accuracy (unweighted sample mean)" if metrics.get("accuracy_level_weighted") else "accuracy"
                lines.append(f"| {acc_label} | {_ci_cell(metrics.get('accuracy'))} | - |")
                if metrics.get("em"):
                    lines.append(f"| EM | {_ci_cell(metrics.get('em'))} | - |")
                if task == "gsm8k":
                    lines.append(f"| format compliance rate | {_fmt_rate(metrics.get('format_compliance_rate'))} | - |")
                    lines.append(f"| extraction failure rate (official regex) | {_fmt_rate(metrics.get('extraction_failure_rate'))} | - |")
                    lines.append(f"| relaxed accuracy (diagnostic) | {_ci_cell(metrics.get('relaxed_accuracy'))} | - |")
                    lines.append(f"| extraction failures rescued by relaxed parser (diagnostic) | {metrics.get('relaxed_would_rescue')} | - |")
                    lines.append(f"| missing final answer (reasoning never closed) | {metrics.get('n_missing_final')} | - |")
                    lines.append(f"| length stops (hit max_new_tokens) | {metrics.get('n_length_terminated')} | - |")
                if task == "math":
                    lines.extend(_math_details(metrics))
                if task == "hotpotqa":
                    af = metrics.get("answer_format") or {}
                    lines.append(f"| answer format: Answer-line / single-line / inline-marker / invalid / missing | "
                                 f"{af.get('answer_line')} / {af.get('single_line')} / {af.get('inline_marker')} / "
                                 f"{af.get('invalid_format')} / {af.get('missing_final')} | - |")
            per_seed = metrics.get("per_seed") or {}
            if per_seed:
                lines.append("")
                lines.append("#### Per-seed point estimates")
                lines.append("")
                lines.append("| seed | primary metric | estimate |")
                lines.append("|---|---|---|")
                for seed, sm in sorted(per_seed.items(), key=lambda kv: int(kv[0])):
                    if task == "musique_local":
                        metric_name = "answerable F1"
                        value = (sm.get("answerable_f1") or {}).get("estimate")
                    elif task == "processbench":
                        metric_name = "macro F1"
                        value = (sm.get("macro_f1") or {}).get("estimate")
                    elif task == "math" and sm.get("accuracy_level_weighted"):
                        metric_name = "accuracy (level-weighted)"
                        value = (sm.get("accuracy_level_weighted") or {}).get("estimate")
                    else:
                        metric_name = "accuracy/F1"
                        value = (sm.get("accuracy") or {}).get("estimate")
                    lines.append(f"| {seed} | {metric_name} | {_fmt(value, 4)} |")
            lines.append("")

    if report.compute:
        lines.append("## Compute")
        lines.append("")
        lines.append("| config | task | Cproxy sum | Cproxy mean | n_text | Cweighted USD | Efficiency_compute |")
        lines.append("|---|---|---|---|---|---|---|")
        for config, tasks in report.compute.items():
            for task, c in tasks.items():
                mean_s = "-" if c.get("mean") is None else f"{c['mean']:.3e}"
                sum_s = "-" if c.get("sum") is None else f"{c['sum']:.3e}"
                n_text_s = "null (run inspect-params)" if c.get("n_text") is None else str(c["n_text"])
                lines.append(
                    f"| {config} | {task} | {sum_s} | {mean_s} | "
                    f"{n_text_s} | {_fmt(c.get('cweighted_usd'), 6)} | {_fmt(c.get('efficiency_compute'), 3)} |"
                )
                kinds = c.get("by_kind") or {}
                for k, v in kinds.items():
                    kmean_s = "-" if v.get("mean") is None else f"{v['mean']:.3e}"
                    ksum_s = "-" if v.get("sum") is None else f"{v['sum']:.3e}"
                    lines.append(f"| | {task} [{k}] | {ksum_s} | {kmean_s} | | | |")
        lines.append("")

    if report.paired:
        lines.append("## Paired config differences (A - B, same items, clustered CI)")
        lines.append("")
        for task, metrics in report.paired.items():
            lines.append(f"### {task}")
            lines.append("")
            for metric, pairs in metrics.items():
                if len(metrics) > 1 or metric != "primary":
                    direction = "lower is better" if metric == "false_abstention_rate" else "higher is better"
                    label = metric
                    if metric == "primary" and "accuracy_level_weighted" in metrics:
                        label = "accuracy, unweighted sample mean"
                    elif metric == "accuracy_level_weighted":
                        label = "accuracy, level-weighted (headline)"
                    lines.append(f"#### {label} ({direction})")
                    lines.append("")
                lines.append("| pair | mean diff | 95% CI |")
                lines.append("|---|---|---|")
                for pair, d in pairs.items():
                    lines.append(f"| {pair} | {_fmt(d.get('estimate'), 4)} | [{_fmt(d.get('ci95_lo'), 4)}, {_fmt(d.get('ci95_hi'), 4)}] |")
                lines.append("")

    rec = report.recommendation
    role_data = bool(rec) and (
        any(e.get("answerable_f1") is not None for e in rec.get("evidence_worker_candidates", []))
        or any(m.get("macro_f1") is not None for m in rec.get("math_checker_candidates", []))
    )
    if rec and rec.get("mode") == "informational_no_nomination" and not role_data:
        # plan without role tasks (e.g. math_solve, reference): nothing to show
        rec = None
    if rec:
        mode = rec.get("mode")
        stage = rec.get("stage")
        if mode == "selection_nomination":
            lines.append("## Role recommendation (selection)")
        elif mode == "holdout_check":
            lines.append("## Holdout check of the selection nominee (no re-pick)")
        elif mode == "informational_no_nomination":
            lines.append(f"## Role metrics vs thresholds (informational - stage `{stage}` never nominates)")
        else:
            lines.append("## Role recommendation")
        lines.append("")
        lines.append(f"stage: **{stage}** · thresholds status: **{rec.get('thresholds', {}).get('status')}**")
        lines.append("")
        lines.append("### Evidence worker (MuSiQue local)")
        lines.append("")
        lines.append("| config | answerable F1 | false-abstention | negative correct-abstention | role Cproxy (50/50 mix) |")
        lines.append("|---|---|---|---|---|")
        for e in rec.get("evidence_worker_candidates", []):
            lines.append(
                f"| {e['config']} | {_fmt(e['answerable_f1'], 3)} | {_fmt(e['false_abstention_rate'], 3)} | "
                f"{_fmt(e['reviewed_negative_correct_abstention'], 3)} | {_fmt(e['mean_cproxy_role_workload_50_50'], 1)} |"
            )
        lines.append("")
        lines.append("### Math checker (ProcessBench)")
        lines.append("")
        lines.append("| config | macro F1 | macro correct acc | macro error acc | mean Cproxy |")
        lines.append("|---|---|---|---|---|")
        for m in rec.get("math_checker_candidates", []):
            lines.append(
                f"| {m['config']} | {_fmt(m['macro_f1'], 3)} | {_fmt(m['macro_correct_solution_acc'], 3)} | "
                f"{_fmt(m['macro_error_solution_acc'], 3)} | {_fmt(m['mean_cproxy'], 1)} |"
            )
        lines.append("")
        hc = rec.get("holdout_check")
        if rec.get("recommendation"):
            lines.append(f"- nominated evidence worker: **{rec['recommendation']['evidence_worker']}**")
            lines.append(f"- nominated math checker: **{rec['recommendation']['math_checker']}**")
            lines.append(f"- {rec['recommendation']['note']}")
        elif hc is not None:
            if hc.get("available"):
                for role in ("evidence_worker", "math_checker"):
                    r = hc.get(role) or {}
                    lines.append(f"- {role.replace('_', ' ')}: selection nominee **{r.get('nominee')}** -> **{r.get('result')}**")
            else:
                lines.append(f"- {hc.get('note')}")
            if rec.get("note"):
                lines.append(f"- {rec['note']}")
        elif rec.get("note"):
            lines.append(f"- {rec['note']}")
        lines.append("")

    if report.unresolved:
        lines.append("## Unresolved infrastructure failures")
        lines.append("")
        lines.append("| config | task | seed | item | attempts | last error |")
        lines.append("|---|---|---|---|---|---|")
        for u in report.unresolved:
            lines.append(
                f"| {u.get('config')} | {u.get('task')} | {u.get('seed')} | {u.get('item_id')} | "
                f"{len(u.get('attempts', []))} | {u.get('last_error_kind')} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report_md(out_dir: Path, plan_name: str, report: ReportData) -> Path:
    rdir = out_dir / "reports" / plan_name
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return path
