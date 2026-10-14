"""Only the selection stage nominates role workers.

Found after the thresholds were confirmed: `_recommendation()` checked only
thresholds.status, so `weval report --plan pilot` (6 positives, 7 negatives,
21 ProcessBench records) printed "nominated evidence worker / math checker",
contradicting "the pilot is not used for formal selection". Now:

  * selection  -> nominates (confirmed thresholds only);
  * holdout    -> pass/fail of the SELECTION nominee, never re-picks;
  * smoke / pilot / reference -> informational, recommendation is None,
    whatever the thresholds status.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.conftest import PROJECT_ROOT, make_cfg
from weval.config import PlanCfg, ThresholdsCfg, load_configs
from weval.metrics.report import ReportData, _recommendation
from weval.metrics.report_md import render_markdown
from weval.run.runner import PlanGateError, _nominated_holdout_configs

CONFIRMED = ThresholdsCfg(
    status="confirmed",
    evidence_worker={
        "answerable_f1_min": 0.80,
        "false_abstention_max": 0.10,
        "reviewed_negative_abstention_min": 0.90,
    },
    math_checker={
        "macro_f1_min": 0.60,
        "macro_correct_solution_acc_min": 0.50,
        "macro_error_solution_acc_min": 0.50,
    },
    tie_break="lowest_mean_cproxy",
)

TASKS = ["musique_local", "processbench"]


def _plan(stage: str, configs=("good", "bad")) -> PlanCfg:
    return PlanCfg(
        stage=stage, configs=list(configs), tasks=list(TASKS),
        require_approved_negatives=True, require_locked_templates=False,
        require_locked_thresholds=stage in ("selection", "holdout"),
        max_attempts_per_item=3, deadline_seconds=60.0,
        report_incomplete_on_unresolved=True,
    )


def _est(v: float) -> Dict[str, Any]:
    return {"estimate": v}


def _metrics(f1: float, fa: float, neg: float, pb_f1: float, pb_c: float, pb_e: float) -> Dict[str, Any]:
    return {
        "musique_local": {
            "answerable_f1": _est(f1),
            "false_abstention_rate": _est(fa),
            "reviewed_negative_correct_abstention": _est(neg),
            "n_positives": 6,
            "n_negatives_reviewed": 7,
            "completion": {"complete": True},
        },
        "processbench": {
            "macro_f1": _est(pb_f1),
            "macro_correct_solution_acc": _est(pb_c),
            "macro_error_solution_acc": _est(pb_e),
            "completion": {"complete": True},
        },
    }


def _compute(c: float) -> Dict[str, Any]:
    return {
        "musique_local": {"by_kind": {"positive": {"mean": c}, "negative": {"mean": c}}},
        "processbench": {"mean": c},
    }


def _report(plan_name: str, good=None, bad=None) -> ReportData:
    r = ReportData(plan=plan_name, created_at_utc="2026-10-14T00:00:00Z", thresholds_status="confirmed")
    # "good" passes both roles (like the pilot's Qwen thinking-on on
    # ProcessBench / Qwen thinking-off on MuSiQue), "bad" fails both.
    r.per_config = {
        "good": good or _metrics(0.92, 0.0, 0.95, 0.91, 0.9, 0.9),
        "bad": bad or _metrics(0.40, 0.3, 0.14, 0.0, 0.0, 0.0),
    }
    r.compute = {"good": _compute(1e12), "bad": _compute(5e11)}
    return r


def _cfg():
    return make_cfg(thresholds=CONFIRMED)


@pytest.mark.parametrize("stage", ["smoke", "pilot", "reference"])
def test_non_selection_stage_never_nominates_even_with_confirmed_thresholds(stage, tmp_path):
    rec = _recommendation(_cfg(), _plan(stage), _report(stage), tmp_path)
    assert rec["thresholds"]["status"] == "confirmed"
    assert rec["stage"] == stage
    assert rec["mode"] == "informational_no_nomination"
    assert rec["recommendation"] is None
    assert "evidence_worker_qualified" not in rec
    assert "math_checker_qualified" not in rec
    assert "NOT the selection stage".lower() in rec["note"].lower()
    # candidate metrics are still shown for information
    assert [e["config"] for e in rec["evidence_worker_candidates"]] == ["good", "bad"]


def test_pilot_markdown_has_no_nomination_lines(tmp_path):
    report = _report("pilot")
    report.recommendation = _recommendation(_cfg(), _plan("pilot"), report, tmp_path)
    md = render_markdown(report)
    assert "nominated evidence worker" not in md
    assert "nominated math checker" not in md
    assert "informational - stage `pilot` never nominates" in md


def test_selection_with_confirmed_thresholds_nominates(tmp_path):
    report = _report("selection")
    rec = _recommendation(_cfg(), _plan("selection"), report, tmp_path)
    assert rec["mode"] == "selection_nomination"
    assert rec["recommendation"]["evidence_worker"] == "good"
    assert rec["recommendation"]["math_checker"] == "good"
    assert [e["config"] for e in rec["evidence_worker_qualified"]] == ["good"]
    report.recommendation = rec
    md = render_markdown(report)
    assert "nominated evidence worker: **good**" in md


def test_selection_with_unconfirmed_thresholds_is_gated(tmp_path):
    cfg = make_cfg(thresholds=dataclasses.replace(CONFIRMED, status="draft_unconfirmed"))
    rec = _recommendation(cfg, _plan("selection"), _report("selection"), tmp_path)
    assert rec["mode"] == "gated_unconfirmed_thresholds"
    assert rec["recommendation"] is None


def test_selection_tie_break_lowest_cproxy(tmp_path):
    good2 = _metrics(0.85, 0.05, 0.92, 0.70, 0.6, 0.6)
    report = _report("selection", bad=good2)  # both qualify; "bad" is cheaper
    rec = _recommendation(_cfg(), _plan("selection"), report, tmp_path)
    assert rec["recommendation"]["evidence_worker"] == "bad"
    assert rec["recommendation"]["math_checker"] == "bad"


def _write_selection_report(out_dir: Path, evidence: str, math: str, mode: str = "selection_nomination") -> None:
    p = out_dir / "reports" / "selection" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "plan": "selection",
        "recommendation": {
            "mode": mode,
            "recommendation": {"evidence_worker": evidence, "math_checker": math},
        },
    }), encoding="utf-8")


def test_holdout_checks_selection_nominee_and_never_repicks(tmp_path):
    # Selection nominated "bad" for both roles; on holdout "bad" fails and
    # "good" passes. Holdout must report the failure, NOT switch to "good".
    _write_selection_report(tmp_path, evidence="bad", math="bad")
    report = _report("holdout")
    rec = _recommendation(_cfg(), _plan("holdout"), report, tmp_path)
    assert rec["mode"] == "holdout_check"
    assert rec["recommendation"] is None
    hc = rec["holdout_check"]
    assert hc["available"] is True
    assert hc["evidence_worker"] == {"nominee": "bad", "result": "fail"}
    assert hc["math_checker"] == {"nominee": "bad", "result": "fail"}
    report.recommendation = rec
    md = render_markdown(report)
    assert "nominated evidence worker" not in md
    assert "selection nominee **bad** -> **fail**" in md


def test_holdout_check_pass_and_missing_nominee(tmp_path):
    _write_selection_report(tmp_path, evidence="good", math=None)
    rec = _recommendation(_cfg(), _plan("holdout"), _report("holdout"), tmp_path)
    assert rec["holdout_check"]["evidence_worker"] == {"nominee": "good", "result": "pass"}
    assert rec["holdout_check"]["math_checker"] == {"nominee": None, "result": "no_selection_nominee"}


def test_holdout_without_selection_report(tmp_path):
    rec = _recommendation(_cfg(), _plan("holdout"), _report("holdout"), tmp_path)
    assert rec["holdout_check"]["available"] is False
    assert rec["recommendation"] is None


def test_runner_rejects_selection_report_not_from_selection_stage(tmp_path):
    _write_selection_report(tmp_path, evidence="good", math="good", mode="informational_no_nomination")
    with pytest.raises(PlanGateError):
        _nominated_holdout_configs(tmp_path)
    _write_selection_report(tmp_path, evidence="good", math="good")
    assert _nominated_holdout_configs(tmp_path) == ["good"]


def test_real_pilot_plan_with_confirmed_project_thresholds_does_not_nominate(tmp_path):
    """The exact case reported: the project's thresholds.yaml is confirmed and
    `weval report --plan pilot` must still not nominate."""
    cfg = load_configs(PROJECT_ROOT)
    assert cfg.thresholds.status == "confirmed"
    plan = cfg.plan("pilot")
    report = ReportData(plan="pilot", created_at_utc="x", thresholds_status="confirmed")
    report.per_config = {c: _metrics(0.92, 0.0, 0.95, 0.91, 0.9, 0.9) for c in plan.configs}
    report.compute = {c: _compute(1e12) for c in plan.configs}
    rec = _recommendation(cfg, plan, report, tmp_path)
    assert rec["recommendation"] is None
    assert rec["mode"] == "informational_no_nomination"
