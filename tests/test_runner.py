"""Runner behavior through the scripted backend (no torch, no network).

Every run below is driven programmatically the same way weval/cli.py cmd_run
does: run_plan(cfg, plan, backend_factory, out_dir, seeds=[...]). Manifests are
tiny hand-written musique_debug.json files in a tmp out_dir (never
D:/AB_Text/artifacts); the scripted backend is selected by building it directly
in the factory, which is exactly what cli._backend_factory does for
--backend scripted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from weval.config import ConfigBundle, PlanCfg, fingerprint
from weval.models.backend import InfrastructureError
from weval.models.scripted import ScriptedBackend
from weval.run.runner import run_plan
from weval.run.store import PredictionStore
from tests.conftest import make_cfg


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _visible() -> List[Dict[str, Any]]:
    return [
        {"slot": 1, "title": "Paris", "text": "Paris is the capital of France.", "role": "support"},
        {"slot": 2, "title": "L1", "text": "London is a city.", "role": "distractor"},
        {"slot": 3, "title": "R1", "text": "Rome is a city.", "role": "distractor"},
        {"slot": 4, "title": "B1", "text": "Berlin is a city.", "role": "distractor"},
        {"slot": 5, "title": "M1", "text": "Madrid is a city.", "role": "distractor"},
    ]


def _item(
    item_id: str,
    parent_id: str,
    *,
    kind: str = "positive",
    review_status: str = "na",
    question: str | None = None,
) -> Dict[str, Any]:
    """Minimal hand-written musique manifest item (fields the renderer,
    selector and runner actually read)."""
    return {
        "item_id": item_id,
        "group_id": parent_id,
        "split": "debug",
        "task": "musique_local",
        "kind": kind,
        "pair_id": f"{parent_id}:pair",
        "parent_id": parent_id,
        "subq_index": 0,
        "n_hops": 1,
        "is_terminal": True,
        "question": question or f"What is the question behind {item_id}?",
        "reference_answers": [],
        "alias_source": "subquestion_canonical_only",
        "expected": "answer" if kind == "positive" else "not_found",
        "visible": _visible(),
        "syntax": {"has_ref": False, "has_relation": False},
        "support_literal_screening": True,
        "review_status": review_status,
        "human_reviewed": False,
        "excluded_reason": None,
        "gold": {"reference_answers": [], "alias_source": "subquestion_canonical_only"},
    }


def _write_musique_manifest(out_dir: Path, items: List[Dict[str, Any]]) -> Path:
    mdir = out_dir / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / "musique_debug.json"
    path.write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_musique_manifest(out_dir: Path) -> List[Dict[str, Any]]:
    return json.loads(
        (out_dir / "manifests" / "musique_debug.json").read_text(encoding="utf-8")
    )["items"]


def _runner_cfg(
    stage: str = "smoke",
    n_parents: int = 10,
    **plan_kw: Any,
) -> ConfigBundle:
    """A ConfigBundle whose only plan is `stage`, big enough parent quota to
    keep every manifest item, and everything else gated off."""
    plan = dict(
        stage=stage,
        configs=["m1"],
        tasks=["musique_local"],
        require_approved_negatives=False,
        require_locked_templates=False,
        require_locked_thresholds=False,
        max_attempts_per_item=3,
        deadline_seconds=60.0,
        report_incomplete_on_unresolved=True,
    )
    plan.update(plan_kw)
    splits = {
        "stages": {stage: {"musique_parents": n_parents}},
        "seeds": {stage: [0]},
        "split_seeds": {},
    }
    return make_cfg(plans={stage: PlanCfg(**plan)}, splits=splits)


class CountingScriptedBackend(ScriptedBackend):
    """ScriptedBackend that records every generate() call, per item."""

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.calls: List[str] = []

    def generate(self, item_id: str, seed: int, prompt_text: str,
                 max_new_tokens: int, deadline_seconds: float):
        self.calls.append(item_id)
        return super().generate(item_id, seed, prompt_text, max_new_tokens, deadline_seconds)


class InfraFailingBackend(ScriptedBackend):
    """Raises InfrastructureError for designated items; valid (possibly wrong)
    scripted outputs for everything else."""

    def __init__(self, fail_item_ids: List[str], kind: str = "oom", **kw: Any):
        super().__init__(**kw)
        self.fail_item_ids = set(fail_item_ids)
        self.kind = kind
        self.calls: Dict[str, int] = {}

    def generate(self, item_id: str, seed: int, prompt_text: str,
                 max_new_tokens: int, deadline_seconds: float):
        self.calls[item_id] = self.calls.get(item_id, 0) + 1
        if item_id in self.fail_item_ids:
            raise InfrastructureError(self.kind, f"simulated {self.kind} for {item_id}")
        return super().generate(item_id, seed, prompt_text, max_new_tokens, deadline_seconds)


def store_factory() -> Any:
    """Plain helper: returns (factory, backends) so tests can inspect the
    per-config scripted backends."""
    backends: Dict[str, CountingScriptedBackend] = {}

    def factory(name: str) -> CountingScriptedBackend:
        b = CountingScriptedBackend()
        backends[name] = b
        return b

    return factory, backends


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------

def test_resume_regenerates_only_the_missing_record(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    items = [_item("p1", "P1"), _item("p2", "P2"), _item("p3", "P3")]
    _write_musique_manifest(out_dir, items)
    cfg = _runner_cfg(n_parents=3)

    factory, backends = store_factory()
    s1 = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
    assert s1.counters["resume"] is True
    assert s1.counters["n_unresolved"] == 0
    assert sorted(backends["m1"].calls) == ["p1", "p2", "p3"]

    store = PredictionStore(out_dir / "predictions" / "smoke")
    paths = {i["item_id"]: store.record_path("m1", "musique_local", 0, i["item_id"]) for i in items}
    for p in paths.values():
        assert p.exists()
    created_before = {iid: store.load(paths[iid])["created_at_utc"] for iid in paths}
    mtimes_before = {iid: paths[iid].stat().st_mtime_ns for iid in paths}

    # Simulate losing one record file on disk.
    paths["p2"].unlink()
    backends.clear()

    s2 = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
    assert s2.counters["n_unresolved"] == 0

    # Only the missing item hit the backend again.
    assert backends["m1"].calls == ["p2"]

    # The surviving records were reused, not rewritten: same bytes on disk
    # (mtime unchanged) and the same created_at_utc inside the record.
    for iid in ("p1", "p3"):
        rec = store.load(paths[iid])
        assert rec["status"] == "ok"
        assert rec["created_at_utc"] == created_before[iid]
        assert paths[iid].stat().st_mtime_ns == mtimes_before[iid]

    # The missing record was regenerated and is complete again.
    rec2 = store.load(paths["p2"])
    assert rec2 is not None
    assert rec2["status"] == "ok"
    assert rec2["item_id"] == "p2"

    # The status counters survive and accumulate across the resume.
    status = store.load_status("m1", "musique_local", 0)
    assert status["completed"] == 4  # 3 first run + 1 regenerated
    assert status["reused"] == 2


def test_resume_no_resume_flag_regenerates_everything(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [_item("p1", "P1"), _item("p2", "P2")])
    cfg = _runner_cfg()

    factory, backends = store_factory()
    run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
    assert sorted(backends["m1"].calls) == ["p1", "p2"]

    backends.clear()
    s2 = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=False)
    assert sorted(backends["m1"].calls) == ["p1", "p2"]
    assert s2.counters.get("resume", False) is False


def test_resume_does_not_reuse_records_from_a_different_backend(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [_item("p1", "P1")])
    cfg = _runner_cfg()

    first = CountingScriptedBackend()
    run_plan(cfg, "smoke", lambda _name: first, out_dir, seeds=[0], resume=True)
    assert first.calls == ["p1"]

    class OtherBackend(CountingScriptedBackend):
        def cache_identity(self) -> Dict[str, str]:
            return {"name": "other", "version": "other-v1"}

    second = OtherBackend()
    run_plan(cfg, "smoke", lambda _name: second, out_dir, seeds=[0], resume=True)

    # The item must be regenerated even though every data/model/prompt
    # fingerprint is otherwise identical.
    assert second.calls == ["p1"]
    store = PredictionStore(out_dir / "predictions" / "smoke")
    rec = store.load(store.record_path("m1", "musique_local", 0, "p1"))
    assert rec["fingerprints"]["generation_settings"]


# --------------------------------------------------------------------------
# fingerprint invalidation
# --------------------------------------------------------------------------

def test_fingerprint_mismatch_regenerates_only_the_mutated_item(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [
        _item("p1", "P1", question="What is Alpha?"),
        _item("p2", "P2", question="What is Beta?"),
    ])
    cfg = _runner_cfg()

    factory, backends = store_factory()
    run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)

    store = PredictionStore(out_dir / "predictions" / "smoke")
    p1_path = store.record_path("m1", "musique_local", 0, "p1")
    p2_path = store.record_path("m1", "musique_local", 0, "p2")
    rec1_before = store.load(p1_path)
    rec2_before = store.load(p2_path)
    assert rec1_before["status"] == "ok"
    mtime2_before = p2_path.stat().st_mtime_ns

    # Mutate one item's content (the prompt input) in the manifest.
    manifest_items = _read_musique_manifest(out_dir)
    for it in manifest_items:
        if it["item_id"] == "p1":
            it["question"] = "What is Alpha, exactly?"
    _write_musique_manifest(out_dir, manifest_items)
    mutated_p1 = next(it for it in manifest_items if it["item_id"] == "p1")

    backends.clear()
    run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)

    # Only the mutated item was regenerated.
    assert backends["m1"].calls == ["p1"]

    rec1_after = store.load(p1_path)
    assert rec1_after["status"] == "ok"
    # The item fingerprint in the record now matches the mutated item ...
    assert rec1_after["fingerprints"]["item"] == fingerprint(mutated_p1)
    # ... and differs from the original, and the prompt fingerprint changed too.
    assert rec1_after["fingerprints"]["item"] != rec1_before["fingerprints"]["item"]
    assert rec1_after["fingerprints"]["prompt"] != rec1_before["fingerprints"]["prompt"]

    # The untouched item was reused, not rewritten.
    rec2_after = store.load(p2_path)
    assert rec2_after["created_at_utc"] == rec2_before["created_at_utc"]
    assert p2_path.stat().st_mtime_ns == mtime2_before


# --------------------------------------------------------------------------
# infra retries
# --------------------------------------------------------------------------

def test_infra_retries_bounded_and_wrong_answers_never_retried(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [
        _item("failing", "P1", question="What is X?"),
        _item("wrong", "P2", question="What is Y?"),
    ])
    cfg = _runner_cfg(max_attempts_per_item=3)

    backends: Dict[str, InfraFailingBackend] = {}

    def factory(name: str) -> InfraFailingBackend:
        # A finished generation with a valid-but-wrong answer ("definitely not
        # the answer") must never be retried; "failing" always hits the infra
        # path.
        b = InfraFailingBackend(fail_item_ids=["failing"], default_text="definitely not the answer")
        backends[name] = b
        return b

    s = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
    assert s.counters["n_unresolved"] == 1

    store = PredictionStore(out_dir / "predictions" / "smoke")
    failing = store.load(store.record_path("m1", "musique_local", 0, "failing"))
    wrong = store.load(store.record_path("m1", "musique_local", 0, "wrong"))

    # Infra item: exactly max_attempts_per_item (3) attempts - 1 initial + 2
    # retries - all classified infra, and the record ends unresolved with no
    # generated output to be scored.
    assert failing["status"] == "infra_failed"
    assert failing["last_error_kind"] == "oom"
    assert [a["attempt"] for a in failing["attempts"]] == [1, 2, 3]
    assert [a["outcome"] for a in failing["attempts"]] == ["infra_error", "infra_error", "infra_error"]
    assert {a["kind"] for a in failing["attempts"]} == {"oom"}
    assert "output_text" not in failing
    assert backends["m1"].calls["failing"] == 3

    # The unresolved failure is listed in the summary with its attempts.
    assert len(s.unresolved) == 1
    assert s.unresolved[0]["item_id"] == "failing"
    assert len(s.unresolved[0]["attempts"]) == 3

    # Wrong-but-valid answer: a single attempt, recorded as an ordinary ok
    # record (scoring happens at report time, not here).
    assert wrong["status"] == "ok"
    assert len(wrong["attempts"]) == 1
    assert wrong["attempts"][0]["attempt"] == 1
    assert wrong["attempts"][0]["outcome"] == "ok"
    assert wrong["output_text"] == "definitely not the answer"
    assert backends["m1"].calls["wrong"] == 1

    # Seed-level counters agree with the record files.
    status = store.load_status("m1", "musique_local", 0)
    assert status["infra_failed"] == 1
    assert status["completed"] == 1


def test_infra_retry_count_follows_max_attempts_per_item(tmp_path: Path) -> None:
    """The runner retries while attempt < max_attempts_per_item: with the
    config raised to 5, an always-failing item gets exactly 5 attempts."""
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [_item("failing", "P1")])
    cfg = _runner_cfg(max_attempts_per_item=5)

    backends: Dict[str, InfraFailingBackend] = {}

    def factory(name: str) -> InfraFailingBackend:
        b = InfraFailingBackend(fail_item_ids=["failing"], kind="timeout")
        backends[name] = b
        return b

    run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
    store = PredictionStore(out_dir / "predictions" / "smoke")
    rec = store.load(store.record_path("m1", "musique_local", 0, "failing"))
    assert rec["status"] == "infra_failed"
    assert [a["attempt"] for a in rec["attempts"]] == [1, 2, 3, 4, 5]
    assert rec["last_error_kind"] == "timeout"
    assert backends["m1"].calls["failing"] == 5


# --------------------------------------------------------------------------
# unapproved-negative skipping
# --------------------------------------------------------------------------

def test_unapproved_negatives_skipped_until_approved(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [
        _item("pos1", "P1", kind="positive", review_status="na"),
        _item("neg1", "P1", kind="negative", review_status="proposed"),
    ])
    cfg = _runner_cfg(require_approved_negatives=True)

    factory, backends = store_factory()
    s1 = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)

    store = PredictionStore(out_dir / "predictions" / "smoke")
    pos_path = store.record_path("m1", "musique_local", 0, "pos1")
    neg_path = store.record_path("m1", "musique_local", 0, "neg1")

    # The proposed (unapproved) negative produced no prediction record ...
    assert store.load(pos_path)["status"] == "ok"
    assert not neg_path.exists()
    assert backends["m1"].calls == ["pos1"]
    # ... and the skip was counted, not silently dropped.
    assert s1.counters["m1/musique_local"]["n_skipped_unapproved_negatives"] == 1
    status = store.load_status("m1", "musique_local", 0)
    assert status["skipped_unapproved_negatives"] == 1
    assert status["completed"] == 1

    # A human approves the negative in the manifest.
    manifest_items = _read_musique_manifest(out_dir)
    for it in manifest_items:
        if it["item_id"] == "neg1":
            it["review_status"] = "approved"
    _write_musique_manifest(out_dir, manifest_items)

    backends.clear()
    s2 = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)

    # Now the negative runs (the positive is resumed from cache).
    assert backends["m1"].calls == ["neg1"]
    neg_rec = store.load(neg_path)
    assert neg_rec is not None
    assert neg_rec["status"] == "ok"
    assert neg_rec["kind"] == "negative"
    assert s2.counters["m1/musique_local"]["n_skipped_unapproved_negatives"] == 0


def test_unapproved_negatives_run_when_gate_is_off(tmp_path: Path) -> None:
    """Without require_approved_negatives the proposed negative runs anyway."""
    out_dir = tmp_path / "artifacts"
    _write_musique_manifest(out_dir, [
        _item("pos1", "P1", kind="positive"),
        _item("neg1", "P1", kind="negative", review_status="proposed"),
    ])
    cfg = _runner_cfg(require_approved_negatives=False)

    factory, backends = store_factory()
    s = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)

    assert sorted(backends["m1"].calls) == ["neg1", "pos1"]
    assert s.counters["m1/musique_local"]["n_skipped_unapproved_negatives"] == 0
    store = PredictionStore(out_dir / "predictions" / "smoke")
    assert store.load(store.record_path("m1", "musique_local", 0, "neg1"))["status"] == "ok"
