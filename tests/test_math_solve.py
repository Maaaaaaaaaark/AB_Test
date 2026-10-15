"""Standalone single-model math solving plan (math_solve).

Covers: MATH gold extraction (nested braces), duplicate index vs brute force,
GSM8K / stratified MATH sampling with exclusions, level weights, the
post-stratified bootstrap, the loose MATH diagnostic, manifest resolution,
per-config runs (--configs) and the math_solve report end to end.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

from weval.cli import PLAN_CHOICES, build_parser
from weval.config import PlanCfg
from weval.datasets import gsm8k as gsm8k_adapter
from weval.datasets import mathbench as math_adapter
from weval.datasets.build import build_math_solve_sets
from weval.datasets.math_solve import (
    MATH_LEVELS,
    DuplicateIndex,
    InsufficientPoolError,
    level_weights,
    sample_gsm8k,
    sample_math_stratified,
)
from weval.datasets.normalization import NEAR_DUP_JACCARD, normalize_for_matching
from weval.metrics.report import _aggregate_math, build_report
from weval.metrics.report_md import render_markdown
from weval.metrics.stats import weighted_strata_bootstrap
from weval.models.scripted import ScriptedBackend
from weval.parsing.base import ParsedAnswer
from weval.run.manifests import _manifest_paths, manifest_items_for_task, manifest_sampling
from weval.run.runner import PlanGateError, run_plan
from weval.scoring.math_loose import loose_equiv
from weval.scoring.math_score import is_equiv, score_math
from tests.conftest import make_cfg, make_model


# --------------------------------------------------------------------------
# MATH gold extraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("solution, gold", [
    ("so the answer is $\\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"),
    ("$\\boxed{\\sqrt{x^{2}+1}}$", "\\sqrt{x^{2}+1}"),
    ("first $\\boxed{1}$, finally $\\boxed{2}$", "2"),
    ("$\\boxed{(1,\\frac{3}{4})}$", "(1,\\frac{3}{4})"),
    ("the answer is \\fbox{7}", "7"),
    ("the answer is $\\boxed 5$.", "5"),
    ("$\\boxed{ \\text{(C)} }$", "\\text{(C)}"),
])
def test_extract_boxed_gold_handles_nested_braces(solution: str, gold: str) -> None:
    assert math_adapter.extract_boxed_gold(solution) == gold


@pytest.mark.parametrize("solution", ["no boxed answer here", "", "$\\boxed{1$", "$\\boxed{}$"])
def test_extract_boxed_gold_none_when_not_extractable(solution: str) -> None:
    assert math_adapter.extract_boxed_gold(solution) is None


def test_math_adapter_version_bumped_for_gold_fix() -> None:
    assert math_adapter.ADAPTER_VERSION == "math-adapter-v3"


# --------------------------------------------------------------------------
# duplicate index
# --------------------------------------------------------------------------

def _brute_force(refs: List[str], text: str, t: float):
    norm = normalize_for_matching(text)
    for r in refs:
        if normalize_for_matching(r) == norm:
            return ("exact", None)
    toks = set(norm.split())
    best = None
    for r in refs:
        rt = set(normalize_for_matching(r).split())
        if not rt or not toks:
            continue
        j = len(toks & rt) / len(toks | rt)
        if j >= t and (best is None or j > best):
            best = j
    return ("near_duplicate", round(best, 4)) if best is not None else None


def test_duplicate_index_matches_brute_force() -> None:
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(60)]
    refs = [" ".join(rng.sample(vocab, rng.randint(15, 45))) for _ in range(250)]
    idx = DuplicateIndex()
    for i, r in enumerate(refs):
        idx.add("ref", str(i), r)
    queries: List[str] = []
    for _ in range(150):
        base = rng.choice(refs).split()
        op = rng.random()
        if op < 0.3:
            q = base + [rng.choice(vocab) + "x"]          # one extra token
        elif op < 0.6:
            q = base[:-1]                                 # one token dropped
        elif op < 0.7:
            q = [w.upper() for w in base]                 # exact after normalization
        else:
            q = rng.sample(vocab, rng.randint(15, 45))    # unrelated
        queries.append(" ".join(q))
    n_near = 0
    for q in queries:
        got = idx.match(q)
        want = _brute_force(refs, q, NEAR_DUP_JACCARD)
        if want is None:
            assert got is None, q
        elif want[0] == "exact":
            assert got is not None and got["kind"] == "exact"
        else:
            n_near += 1
            assert got is not None and got["kind"] == "near_duplicate"
            assert got["jaccard"] == want[1]
    assert n_near > 0  # the test exercised the near-duplicate path


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

FILLER = " ".join(f"f{k}" for k in range(28))


def _gsm_records(n: int = 50) -> List[Dict[str, Any]]:
    # records differ in 2 of ~30 tokens -> pairwise Jaccard ~0.88 (< 0.95)
    return [{"question": f"q{i} {FILLER} r{i}", "answer": f"... #### {i}"} for i in range(n)]


def test_sample_gsm8k_exclusions_and_within_sample_duplicates() -> None:
    records = _gsm_records(50)
    records[11]["question"] = records[10]["question"]  # duplicate inside the pool
    ex = DuplicateIndex()
    ex.add("processbench", "gsm8k:0", records[3]["question"].upper() + " ?")  # exact
    ex.add("reference_or_pilot", "gsm8k_reference:7", records[5]["question"] + " extra")  # near-dup
    res = sample_gsm8k(records, 47, seed=1, exclusions=ex)
    assert len(res.chosen) == 47 == len(set(res.chosen))
    assert 3 not in res.chosen and 5 not in res.chosen
    assert len({10, 11} & set(res.chosen)) == 1
    by_reason = res.counts["excluded_by_reason"]
    assert by_reason == {
        "duplicate_within_sample": 1,
        "overlap_processbench": 1,
        "overlap_reference_or_pilot": 1,
    }
    kinds = {e["index"]: e["kind"] for e in res.excluded}
    assert kinds[3] == "exact" and kinds[5] == "near_duplicate"
    with pytest.raises(InsufficientPoolError):
        sample_gsm8k(records, 48, seed=1, exclusions=ex)


def test_sample_gsm8k_is_deterministic() -> None:
    records = _gsm_records(50)
    a = sample_gsm8k(records, 20, seed=7, exclusions=DuplicateIndex()).chosen
    b = sample_gsm8k(records, 20, seed=7, exclusions=DuplicateIndex()).chosen
    c = sample_gsm8k(records, 20, seed=8, exclusions=DuplicateIndex()).chosen
    assert a == b and a != c


def _math_records() -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    subjects = ["Algebra", "Geometry", "Prealgebra"]
    for k, lvl in enumerate(MATH_LEVELS):
        for j in range(12 + k):  # unequal population per level
            recs.append({
                "problem": f"p{k}_{j} {FILLER} s{k}_{j}",
                "solution": f"... $\\boxed{{{k}{j}}}$",
                "level": lvl, "type": subjects[j % 3],
            })
    recs.append({"problem": "odd one " + FILLER, "solution": "$\\boxed{0}$", "level": "Level ?", "type": "Algebra"})
    return recs


def test_sample_math_stratified_equal_per_level() -> None:
    recs = _math_records()
    ex = DuplicateIndex()
    ex.add("processbench", "math:0", recs[13]["problem"])  # a Level 2 record
    res = sample_math_stratified(recs, per_level=10, seed=3, exclusions=ex)
    assert res.counts["sample_level_counts"] == {lvl: 10 for lvl in MATH_LEVELS}
    assert res.counts["n_invalid_level_excluded"] == 1
    assert res.counts["population_level_counts"] == {lvl: 12 + k for k, lvl in enumerate(MATH_LEVELS)}
    assert res.counts["eligible_level_counts"] == {
        "Level 1": 12, "Level 2": 12, "Level 3": 14,
        "Level 4": 15, "Level 5": 16,
    }
    assert res.counts["n_externally_excluded_in_population"] == 1
    assert 13 not in res.chosen
    assert sum(res.counts["sample_subject_counts"].values()) == 50
    assert all(recs[i]["level"] in MATH_LEVELS for i in res.chosen)
    again = sample_math_stratified(recs, per_level=10, seed=3, exclusions=ex)
    assert again.chosen == res.chosen
    with pytest.raises(InsufficientPoolError):
        sample_math_stratified(recs, per_level=13, seed=3, exclusions=ex)  # Level 1 has only 12


def test_level_weights_are_population_proportions() -> None:
    w = level_weights({"Level 1": 437, "Level 2": 894, "Level 3": 1131, "Level 4": 1214, "Level 5": 1324})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["Level 5"] == pytest.approx(1324 / 5000)
    assert level_weights({}) == {}


# --------------------------------------------------------------------------
# build: frozen samples written with exclusions
# --------------------------------------------------------------------------

def test_build_math_solve_sets_writes_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gsm = _gsm_records(30)
    maths = _math_records()
    monkeypatch.setattr(gsm8k_adapter, "load_gsm8k_main_test", lambda cache_dir=None: gsm)
    monkeypatch.setattr(math_adapter, "load_math", lambda cache_dir=None: {"train": [], "test": maths})
    out = tmp_path / "artifacts"
    (out / "manifests").mkdir(parents=True)
    # reference manifests written earlier in the build
    (out / "manifests" / "gsm8k_reference.json").write_text(json.dumps(
        {"items": [{"item_id": "0", "question": gsm[2]["question"]}]}), encoding="utf-8")
    (out / "manifests" / "math_reference.json").write_text(json.dumps(
        {"items": [{"item_id": "0", "problem": maths[0]["problem"]}]}), encoding="utf-8")
    cfg = make_cfg(splits={
        "stages": {"math_solve": {"gsm8k_items": 20, "math_per_level": 5}},
        "seeds": {"math_solve": [0, 1, 2]},
        "split_seeds": {"math_solve_order": 11},
    })
    pb = [("gsm8k", "1", gsm[4]["question"]), ("math", "9", maths[1]["problem"])]
    res = build_math_solve_sets(cfg, tmp_path / "data", out, {}, pb)
    assert not res.blocked, res.blocked
    g = json.loads((out / "manifests" / "gsm8k_math_solve.json").read_text(encoding="utf-8"))
    m = json.loads((out / "manifests" / "math_math_solve.json").read_text(encoding="utf-8"))
    assert g["n_items"] == 20 and m["n_items"] == 25
    g_ids = {it["item_id"] for it in g["items"]}
    assert "test-2" not in g_ids and "test-4" not in g_ids
    m_ids = {it["item_id"] for it in m["items"]}
    assert "test-0" not in m_ids and "test-1" not in m_ids
    assert all(it["split"] == "math_solve" and it["group_id"] == f"math:{it['item_id']}" for it in m["items"])
    assert m["sampling"]["sample_level_counts"] == {lvl: 5 for lvl in MATH_LEVELS}
    assert sum(m["sampling"]["level_weights"].values()) == pytest.approx(1.0)
    assert m["sampling"]["level_weights_source"] == "eligible_level_counts_after_external_exclusions"
    eligible = m["sampling"]["eligible_level_counts"]
    assert eligible["Level 1"] == 10  # two externally excluded Level 1 records
    assert m["sampling"]["level_weights"]["Level 1"] == pytest.approx(
        eligible["Level 1"] / sum(eligible.values())
    )
    assert m["sampling"]["n_items_without_gold_boxed"] == 0
    ex = json.loads((out / "manifests" / "math_solve_exclusions.json").read_text(encoding="utf-8"))
    assert ex["exclusion_sources"]["processbench_problems"] == 2
    assert "excluded" in ex["gsm8k"] and "excluded" in ex["math"]


def test_build_math_solve_blocked_without_processbench(tmp_path: Path) -> None:
    cfg = make_cfg(splits={"stages": {"math_solve": {"gsm8k_items": 2, "math_per_level": 1}},
                           "seeds": {}, "split_seeds": {}})
    res = build_math_solve_sets(cfg, tmp_path, tmp_path, {}, [])
    assert "math_solve" in res.blocked
    assert not (tmp_path / "manifests" / "gsm8k_math_solve.json").exists()


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def test_weighted_strata_bootstrap_estimate_and_ci() -> None:
    data = {
        "A": {f"a{i}": [1.0, 1.0] for i in range(5)},
        "B": {f"b{i}": [0.0, 0.0, 0.0] for i in range(5)},
    }
    ci = weighted_strata_bootstrap(data, {"A": 0.25, "B": 0.75}, n_boot=200)
    assert ci.estimate == pytest.approx(0.25)
    assert ci.lo == pytest.approx(0.25) and ci.hi == pytest.approx(0.25)
    mixed = {
        "A": {"a0": [1.0], "a1": [0.0], "a2": [1.0], "a3": [1.0]},
        "B": {"b0": [0.0], "b1": [1.0], "b2": [0.0], "b3": [0.0]},
    }
    ci = weighted_strata_bootstrap(mixed, {"A": 0.4, "B": 0.6}, n_boot=500)
    assert ci.estimate == pytest.approx(0.4 * 0.75 + 0.6 * 0.25)
    assert ci.lo <= ci.estimate <= ci.hi
    # a stratum without data: weights renormalized over the strata present
    ci = weighted_strata_bootstrap({"A": {"a": [1.0]}, "B": {}}, {"A": 0.5, "B": 0.5}, n_boot=0)
    assert ci.estimate == pytest.approx(1.0)


# --------------------------------------------------------------------------
# loose MATH diagnostic
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pred, gold, expected", [
    ("0.25", "\\frac{1}{4}", True),
    ("x=3", "3", True),
    ("5\\text{ cm}", "5", True),
    ("2\\pi", "\\pi\\cdot 2", True),
    ("\\dfrac{3}{4}", "\\frac34", True),
    ("(1, 2)", "(1,2)", True),
    ("(1,2)", "(2,1)", False),
    ("\\text{(C)}", "C", True),
    ("90^\\circ", "90", True),
    ("50%", "0.5", True),
    ("50\\%", "\\frac{1}{2}", True),
    ("50%", "50", False),
    ("1,000", "1000", True),
    ("\\sqrt{2}/2", "\\frac{\\sqrt{2}}{2}", True),
    ("x^{2}+1", "x^2+1", True),
    ("[0,1)", "[0,1]", False),
    ("3", "4", False),
    (None, None, False),
    ("", "", False),
])
def test_loose_equiv(pred: Any, gold: Any, expected: bool) -> None:
    assert loose_equiv(pred, gold, is_equiv) is expected


def _parsed(value: Any, status: str = "ok") -> ParsedAnswer:
    return ParsedAnswer(task="math", parser="math_boxed", raw_text="", cleaned_text="",
                        final_region="", region_state="closed", value=value, status=status)


def test_score_math_primary_is_official_loose_is_detail_only() -> None:
    item = {"item_id": "x", "group_id": "math:x", "split": "math_solve", "gold": {"boxed_answer": "\\frac{1}{4}"}}
    s = score_math(item, _parsed("0.25"))  # equal, but not to the official string checker
    assert s.value == 0.0 and s.detail["correct"] is False and s.detail["loose_correct"] is True
    s = score_math(item, _parsed("\\frac14"))
    assert s.value == 1.0 and s.detail["loose_correct"] is True
    s = score_math(item, _parsed(None, "missing_final"))
    assert s.value == 0.0 and s.detail["loose_correct"] is False


# --------------------------------------------------------------------------
# manifests / plan wiring
# --------------------------------------------------------------------------

def _math_item(idx: int, level: str, gold: str, subject: str = "Algebra") -> Dict[str, Any]:
    [it] = math_adapter.build_items([{
        "problem": f"Problem {idx}: compute something {FILLER}",
        "solution": f"... $\\boxed{{{gold}}}$", "level": level, "type": subject,
    }], "math_solve")
    d = it.to_dict()
    d["item_id"] = f"test-{idx}"
    d["group_id"] = f"math:test-{idx}"
    return d


def _gsm_item(idx: int, answer: int) -> Dict[str, Any]:
    [it] = gsm8k_adapter.build_items([{"question": f"How many apples? case {idx}",
                                       "answer": f"... #### {answer}"}], "math_solve")
    d = it.to_dict()
    d["item_id"] = f"test-{idx}"
    d["group_id"] = f"gsm8k:test-{idx}"
    return d


# MATH: 2 problems per level; golds chosen so a "0.25" answer is loose-only
MATH_GOLDS = {
    100: ("Level 1", "4"), 101: ("Level 1", "\\frac{1}{4}"),
    200: ("Level 2", "7"), 201: ("Level 2", "9"),
    300: ("Level 3", "10"), 301: ("Level 3", "11"),
    400: ("Level 4", "12"), 401: ("Level 4", "13"),
    500: ("Level 5", "14"), 501: ("Level 5", "15"),
}
WEIGHTS = {"Level 1": 0.1, "Level 2": 0.15, "Level 3": 0.2, "Level 4": 0.25, "Level 5": 0.3}


def _write_math_solve_manifests(out: Path) -> None:
    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    gsm_items = [_gsm_item(i, i) for i in range(4)]
    (mdir / "gsm8k_math_solve.json").write_text(json.dumps(
        {"items": gsm_items, "sampling": {"design": "simple_random", "n_sampled": 4, "population": 10}}),
        encoding="utf-8")
    math_items = [_math_item(i, lvl, g) for i, (lvl, g) in MATH_GOLDS.items()]
    (mdir / "math_math_solve.json").write_text(json.dumps({"items": math_items, "sampling": {
        "design": "stratified_equal_per_level", "seed": 1, "population": 50, "n_sampled": 10,
        "level_weights": WEIGHTS, "excluded": [{"index": 1}],
        "excluded_by_reason": {"overlap_processbench": 1},
    }}), encoding="utf-8")


def _math_solve_cfg():
    plan = PlanCfg(stage="math_solve", configs=["m1", "m3"], tasks=["gsm8k", "math"],
                   require_approved_negatives=False, require_locked_templates=False,
                   require_locked_thresholds=False, max_attempts_per_item=3,
                   deadline_seconds=60.0, report_incomplete_on_unresolved=True)
    return make_cfg(
        plans={"math_solve": plan},
        models={"m1": make_model(), "m3": make_model(name="m3")},
        splits={"stages": {"math_solve": {"gsm8k_items": 4, "math_per_level": 2}},
                "seeds": {"math_solve": [0, 1]}, "split_seeds": {}},
    )


def test_manifest_resolution_for_math_solve(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    with pytest.raises(FileNotFoundError):
        _manifest_paths(out, "math", "math_solve")
    assert manifest_sampling(out, "math_solve", "math") == {}
    _write_math_solve_manifests(out)
    assert _manifest_paths(out, "math", "math_solve") == [out / "manifests" / "math_math_solve.json"]
    assert _manifest_paths(out, "gsm8k", "math_solve") == [out / "manifests" / "gsm8k_math_solve.json"]
    cfg = _math_solve_cfg()
    assert len(manifest_items_for_task(cfg, out, "math_solve", "math")) == 10  # no stage cap
    assert manifest_sampling(out, "math_solve", "math")["level_weights"] == WEIGHTS


def test_cli_accepts_math_solve_and_configs() -> None:
    assert "math_solve" in PLAN_CHOICES
    p = build_parser()
    a = p.parse_args(["run", "--plan", "math_solve", "--configs", "a,b"])
    assert a.plan == "math_solve" and a.configs == "a,b"
    a = p.parse_args(["estimate", "--source", "pilot", "--target", "math_solve"])
    assert a.target == "math_solve"
    a = p.parse_args(["report", "--plan", "math_solve"])
    assert a.plan == "math_solve"


def _script(path: Path, outputs: Dict[str, str]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in outputs.items():
            f.write(json.dumps({"item_id": k, "text": v}) + "\n")
    return path


def _factory(tmp_path: Path, calls: Dict[str, int]):
    # m1: every GSM8K right; MATH right except test-101 answered "0.25"
    # (loose-only) and test-501 with no boxed answer.
    # m3: every GSM8K wrong; MATH right only on Level 1-2.
    m1 = {f"test-{i}": f"Reasoning.\n#### {i}" for i in range(4)}
    m3 = {f"test-{i}": "Reasoning.\n#### 999" for i in range(4)}
    for i, (lvl, g) in MATH_GOLDS.items():
        m1[f"test-{i}"] = f"so $\\boxed{{{g}}}$"
        m3[f"test-{i}"] = f"so $\\boxed{{{g}}}$" if lvl in ("Level 1", "Level 2") else "so $\\boxed{-1}$"
    m1["test-101"] = "so $\\boxed{0.25}$"
    m1["test-501"] = "I am not sure."
    scripts = {"m1": _script(tmp_path / "m1.jsonl", m1), "m3": _script(tmp_path / "m3.jsonl", m3)}

    def factory(name: str) -> ScriptedBackend:
        calls[name] = calls.get(name, 0) + 1
        return ScriptedBackend(script_path=scripts[name])

    return factory


def test_run_only_configs_rejects_unknown_config(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    _write_math_solve_manifests(out)
    with pytest.raises(PlanGateError):
        run_plan(_math_solve_cfg(), "math_solve", _factory(tmp_path, {}), out, only_configs=["nope"])


def test_math_solve_per_config_runs_and_report(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    _write_math_solve_manifests(out)
    cfg = _math_solve_cfg()
    calls: Dict[str, int] = {}
    s1 = run_plan(cfg, "math_solve", _factory(tmp_path, calls), out, only_configs=["m1"])
    assert s1.configs == ["m1"] and calls == {"m1": 1}
    assert not (out / "predictions" / "math_solve" / "m3").exists()
    run_plan(cfg, "math_solve", _factory(tmp_path, calls), out, only_configs=["m3"])
    assert (out / "predictions" / "math_solve" / "m3" / "math" / "1").exists()

    report = build_report(cfg, "math_solve", out, n_boot=200)
    m1, m3 = report.per_config["m1"], report.per_config["m3"]
    assert m1["gsm8k"]["accuracy"]["estimate"] == pytest.approx(1.0)
    assert m3["gsm8k"]["accuracy"]["estimate"] == pytest.approx(0.0)
    assert m1["gsm8k"]["mean_generated_tokens"] is not None

    mm = m1["math"]
    assert mm["completion"]["complete"] is True
    assert mm["accuracy"]["estimate"] == pytest.approx(0.8)  # 8 of 10 problems, both seeds
    # level-weighted: L1 0.5, L5 0.5, others 1.0
    want = 0.1 * 0.5 + 0.15 + 0.2 + 0.25 + 0.3 * 0.5
    assert mm["accuracy_level_weighted"]["estimate"] == pytest.approx(want)
    assert set(mm["per_level"]) == set(MATH_LEVELS)
    assert mm["per_level"]["Level 1"]["n_items"] == 2
    assert mm["loose_accuracy_diagnostic"]["estimate"] == pytest.approx(0.9)
    lv = mm["loose_vs_official"]
    assert lv["n_loose_correct_official_wrong"] == 2  # test-101 x 2 seeds
    assert lv["n_official_correct_loose_wrong"] == 0
    assert {e["item_id"] for e in lv["examples"]} == {"test-101"}
    assert mm["n_invalid_format"] + mm["n_missing_final"] == 2  # test-501 x 2 seeds
    assert mm["subject_counts"] == {"Algebra": 10}
    assert "excluded" not in mm["sampling"]
    assert m3["math"]["accuracy_level_weighted"]["estimate"] == pytest.approx(0.25)
    # per-seed headline is the level-weighted accuracy
    assert m1["math"]["per_seed"]["0"]["accuracy_level_weighted"]["estimate"] == pytest.approx(want)

    paired = report.paired["math"]
    assert set(paired) == {"primary", "accuracy_level_weighted"}
    assert paired["accuracy_level_weighted"]["m1 - m3"]["estimate"] == pytest.approx(want - 0.25)
    assert paired["primary"]["m1 - m3"]["estimate"] == pytest.approx(0.8 - 0.4)

    md = render_markdown(report)
    assert "## Summary - single-model math solving" in md
    assert "### MATH accuracy by level" in md
    assert "accuracy, level-weighted (headline)" in md
    assert "Evidence worker" not in md  # no role section for a plan without role tasks
    assert "test-101" in md


def test_aggregate_math_without_sampling_has_no_weighted_accuracy() -> None:
    items = {f"t{i}": {"item_id": f"t{i}", "level": "Level 1", "type": "Algebra"} for i in range(3)}
    recs = [{"item_id": f"t{i}", "group_id": f"g{i}", "value": float(i % 2), "status": "scored",
             "seed": 0, "detail": {"loose_correct": bool(i % 2)}} for i in range(3)]
    m = _aggregate_math(recs, items, {}, n_boot=0)
    assert m["accuracy_level_weighted"] is None
    assert m["accuracy"]["estimate"] == pytest.approx(1 / 3)
    assert m["per_level"]["Level 1"]["n_items"] == 3


def test_estimate_covers_only_target_configs_and_tasks(tmp_path: Path) -> None:
    from weval.estimate import estimate

    out = tmp_path / "artifacts"
    _write_math_solve_manifests(out)
    cfg = _math_solve_cfg()
    cfg.plans["math_solve"] = PlanCfg(
        stage="math_solve", configs=["m1"], tasks=["math"],
        require_approved_negatives=False, require_locked_templates=False,
        require_locked_thresholds=False, max_attempts_per_item=3,
        deadline_seconds=60.0, report_incomplete_on_unresolved=True,
    )
    cfg.plans["pilot"] = PlanCfg(
        stage="pilot", configs=["m1", "m3"], tasks=["musique_local", "gsm8k", "math"],
        require_approved_negatives=False, require_locked_templates=False,
        require_locked_thresholds=False, max_attempts_per_item=3,
        deadline_seconds=60.0, report_incomplete_on_unresolved=True,
    )
    cfg.splits["seeds"]["pilot"] = [0]
    for config in ("m1", "m3"):
        for task in ("gsm8k", "math"):
            d = out / "predictions" / "pilot" / config / task / "0"
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{i}.json").write_text(json.dumps({
                    "status": "ok", "item_id": str(i), "group_id": f"g{i}",
                    "n_prompt_tokens": 100, "n_generated_tokens": 4096 if i == 0 else 500,
                    "wall_seconds": 10.0, "finish_reason": "length" if i == 0 else "stop",
                }), encoding="utf-8")
    est = estimate(cfg, out, "pilot", "math_solve", n_boot=50)
    assert set(est["per_config_task"]) == {"m1/math"}
    cell = est["per_config_task"]["m1/math"]
    assert cell["n_target_items_per_seed"] == 10 and cell["n_target_seeds"] == 2
    assert cell["n_source_length_stops"] == 1
    assert any("lower bound" in w for w in est["warnings"])
