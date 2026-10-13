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
                lines.append(f"- positives: {metrics.get('n_positives')}, reviewed negatives: {metrics.get('n_negatives_reviewed')}, "
                             f"unreviewed negatives (dev only): {metrics.get('n_negatives_unreviewed_dev_only')}")
                lines.append(f"- relation questions: n={rn.get('relation', {}).get('n')}, mean F1={_fmt(rn.get('relation', {}).get('mean_f1'), 3)}; "
                             f"natural questions: n={rn.get('natural', {}).get('n')}, mean F1={_fmt(rn.get('natural', {}).get('mean_f1'), 3)}")
            elif task == "processbench":
                lines.append("| subset | error acc | correct acc | F1 | n_error | n_correct | format errors |")
                lines.append("|---|---|---|---|---|---|---|")
                for subset, s in (metrics.get("per_subset") or {}).items():
                    lines.append(
                        f"| {subset} | {_fmt_rate(s.get('error_acc'))} | {_fmt_rate(s.get('correct_acc'))} | "
                        f"{_fmt_rate(s.get('f1'))} | {s.get('n_error')} | {s.get('n_correct')} | {s.get('n_unscored_format_errors')} |"
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
                lines.append(f"| accuracy | {_ci_cell(metrics.get('accuracy'))} | - |")
                if metrics.get("em"):
                    lines.append(f"| EM | {_ci_cell(metrics.get('em'))} | - |")
                if task == "gsm8k":
                    lines.append(f"| extraction failure rate (official regex) | {_fmt_rate(metrics.get('extraction_failure_rate'))} | - |")
                    lines.append(f"| extraction failures rescued by relaxed parser (diagnostic) | {metrics.get('relaxed_would_rescue')} | - |")
                if task == "math":
                    lines.append(f"| unscorable (gold missing boxed) | {metrics.get('n_unscorable_gold_missing')} | - |")
                    lines.append(f"| missing final answer (open region / no boxed) | {metrics.get('n_missing_final')} | - |")
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
        for task, pairs in report.paired.items():
            lines.append(f"### {task}")
            lines.append("")
            lines.append("| pair | mean diff | 95% CI |")
            lines.append("|---|---|---|")
            for pair, d in pairs.items():
                lines.append(f"| {pair} | {_fmt(d.get('estimate'), 4)} | [{_fmt(d.get('ci95_lo'), 4)}, {_fmt(d.get('ci95_hi'), 4)}] |")
            lines.append("")

    rec = report.recommendation
    if rec:
        lines.append("## Role recommendation")
        lines.append("")
        lines.append(f"thresholds status: **{rec.get('thresholds', {}).get('status')}**")
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
        if rec.get("recommendation"):
            lines.append(f"- nominated evidence worker: **{rec['recommendation']['evidence_worker']}**")
            lines.append(f"- nominated math checker: **{rec['recommendation']['math_checker']}**")
            lines.append(f"- {rec['recommendation']['note']}")
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
