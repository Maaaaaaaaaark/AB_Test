"""Run orchestration: plan gating, per-item generation, retries, resume.

Flow per (config, task, seed, item):
  0. holdout special rule: configs are restricted to the selection report's
     nominated candidates (never used to re-pick),
  1. gate the plan (locked templates / locked thresholds / negatives),
  2. render the prompt (gold never enters prompts),
  3. resume check: an existing record whose fingerprints all match is reused,
  4. eligibility: context-length check with the REAL tokenizer (scripted
     dry-runs record the check as skipped),
  5. generate once; infrastructure failures (timeout/OOM/crash) are retried
     on the same item+seed up to max_attempts_per_item total attempts
     (default 3) - wrong/empty answers are NEVER retried,
  6. persist atomically + progress log.

Token-limit termination (finish_reason "length") is a normal evaluated
outcome. Unresolved infra failures are listed in the summary and mark the
comparison incomplete (report_incomplete_on_unresolved).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import GENERATION_PARAMS_VERSION, ConfigBundle, PlanCfg
from ..models.backend import GenerationBackend, InfrastructureError
from ..parsing import PARSER_VERSIONS
from ..prompts.renderers import render_item
from ..scoring import SCORER_VERSIONS
from .eligibility import check_context_fits
from .manifests import manifest_items_for_task
from .store import PREDICTION_RECORD_VERSION, PredictionStore, record_fingerprints

RUNNER_VERSION = "runner-v1"


class PlanGateError(RuntimeError):
    """The plan refuses to run until its stated requirements are met."""


@dataclass
class RunSummary:
    plan: str
    configs: List[str]
    tasks: List[str]
    seeds: List[int]
    counters: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    gating: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runner_version": RUNNER_VERSION,
            "plan": self.plan,
            "configs": self.configs,
            "tasks": self.tasks,
            "seeds": self.seeds,
            "counters": self.counters,
            "unresolved": self.unresolved,
            "gating": self.gating,
        }


def check_gates(cfg: ConfigBundle, plan: PlanCfg) -> Dict[str, Any]:
    """Return {ok, problems[]}; raises nothing."""
    problems: List[str] = []
    if plan.require_locked_templates:
        unlocked = [m for m in plan.configs if not cfg.model(m).template_sha256]
        if unlocked:
            problems.append(
                f"require_locked_templates: {unlocked} have no pinned template "
                "sha256 - run `weval precheck` on the cluster first"
            )
    if plan.require_locked_thresholds:
        status = cfg.thresholds.status
        if status != "confirmed":
            problems.append(
                f"require_locked_thresholds: thresholds.yaml status is "
                f"{status!r}, must be user-confirmed (set status: confirmed)"
            )
    return {"ok": not problems, "problems": problems}


def _nominated_holdout_configs(out_dir: Path) -> List[str]:
    """Read the selection report's nomination. The holdout plan runs ONLY the
    nominated candidate(s) - it may never be used to re-pick."""
    sel = out_dir / "reports" / "selection" / "report.json"
    if not sel.exists():
        raise PlanGateError(
            "holdout requires the selection report: run "
            "`weval run --plan selection` and `weval report --plan selection` first"
        )
    data = json.loads(sel.read_text(encoding="utf-8"))
    rec = data.get("recommendation") or {}
    inner = rec.get("recommendation") or {}
    nominated = {inner.get("evidence_worker"), inner.get("math_checker")} - {None}
    if not nominated:
        raise PlanGateError(
            "the selection report nominates no candidate (nobody qualified); "
            "holdout cannot run"
        )
    return sorted(nominated)


def run_plan(
    cfg: ConfigBundle,
    plan_name: str,
    backend_factory: Callable[[str], GenerationBackend],
    out_dir: Path,
    seeds: Optional[List[int]] = None,
    resume: bool = True,
    log_every: int = 10,
) -> RunSummary:
    plan = cfg.plan(plan_name)
    gates = check_gates(cfg, plan)
    configs = list(plan.configs)
    if plan_name == "holdout":
        nominated = _nominated_holdout_configs(out_dir)
        configs = [c for c in configs if c in nominated]
        if not configs:
            raise PlanGateError(
                f"holdout: nominated configs {nominated} are not in the holdout plan"
            )
        gates["holdout_nominated"] = configs
    summary = RunSummary(plan=plan_name, configs=configs, tasks=list(plan.tasks), seeds=seeds or cfg.splits["seeds"].get(plan_name, [0]), gating=gates)
    if not gates["ok"]:
        raise PlanGateError("plan gating failed:\n" + "\n".join(f" - {p}" for p in gates["problems"]))

    store = PredictionStore(out_dir / "predictions" / plan_name)
    seeds = summary.seeds

    manifest_cache: Dict[str, List[Dict[str, Any]]] = {}

    def items_for(task: str) -> List[Dict[str, Any]]:
        if task not in manifest_cache:
            manifest_cache[task] = manifest_items_for_task(cfg, out_dir, plan_name, task)
        return manifest_cache[task]

    for config_name in configs:
        model_cfg = cfg.model(config_name)
        backend = backend_factory(config_name)
        try:
            eos_strings = getattr(backend, "eos_strings", lambda: None)()
            for task in plan.tasks:
                task_cfg = cfg.task(task)
                items = items_for(task)
                if plan.require_approved_negatives:
                    runnable_items = [
                        it for it in items
                        if not (it.get("kind") == "negative" and it.get("review_status") != "approved")
                    ]
                    n_skipped_unapproved = len(items) - len(runnable_items)
                else:
                    runnable_items = items
                    n_skipped_unapproved = 0
                parser_version = PARSER_VERSIONS.get(task_cfg.parser, "unknown")
                scorer_version = SCORER_VERSIONS.get(task_cfg.scorer, "unknown")
                for seed in seeds:
                    counters = {
                        "completed": 0, "reused": 0, "excluded_context": 0,
                        "infra_failed": 0, "skipped_unapproved_negatives": 0,
                        "n_tokens_in": 0, "n_tokens_out": 0, "elapsed_seconds": 0.0,
                    }
                    existing = store.load_status(config_name, task, seed)
                    for k, v in existing.items():
                        if k in counters:
                            counters[k] = v
                    t_seed_start = time.monotonic()
                    n_done_in_seed = 0
                    for item in runnable_items:
                        item_id = item["item_id"]
                        rendered = render_item(item, cfg)
                        expected = record_fingerprints(
                            item, rendered.user_message, model_cfg.revision,
                            model_cfg.template_sha256, GENERATION_PARAMS_VERSION,
                            parser_version, scorer_version,
                        )
                        rpath = store.record_path(config_name, task, seed, item_id)
                        if resume:
                            prev = store.load(rpath)
                            if prev is not None and prev.get("status") == "ok" and store.is_valid(prev, expected):
                                counters["reused"] += 1
                                counters["n_tokens_in"] += prev["n_prompt_tokens"]
                                counters["n_tokens_out"] += prev["n_generated_tokens"]
                                continue
                        fits, ctx_info = check_context_fits(backend, rendered.user_message, task_cfg.max_new_tokens)
                        if not fits:
                            counters["excluded_context"] += 1
                            store.save_atomic(rpath, {
                                "record_version": PREDICTION_RECORD_VERSION,
                                "status": "excluded_context",
                                "item_id": item_id,
                                "config": config_name,
                                "task": task,
                                "seed": seed,
                                "fingerprints": expected,
                                "context": ctx_info,
                                "excluded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            })
                            store.log(config_name, task, f"[s{seed}] {item_id}: excluded context ({ctx_info.get('required_tokens')}/{ctx_info.get('context_limit')} tokens)")
                            continue
                        attempts: List[Dict[str, Any]] = []
                        result = None
                        last_err = None
                        for attempt in range(1, plan.max_attempts_per_item + 1):
                            t0 = time.monotonic()
                            try:
                                result = backend.generate(
                                    item_id=item_id, seed=seed,
                                    prompt_text=rendered.user_message,
                                    max_new_tokens=task_cfg.max_new_tokens,
                                    deadline_seconds=plan.deadline_seconds,
                                )
                                attempts.append({"attempt": attempt, "outcome": "ok", "seconds": round(time.monotonic() - t0, 3)})
                                break
                            except InfrastructureError as e:
                                attempts.append({"attempt": attempt, "outcome": "infra_error", "kind": e.kind, "seconds": round(time.monotonic() - t0, 3), "message": str(e)[:500]})
                                last_err = e
                        if result is None:
                            counters["infra_failed"] += 1
                            store.save_atomic(rpath, {
                                "record_version": PREDICTION_RECORD_VERSION,
                                "status": "infra_failed",
                                "item_id": item_id,
                                "config": config_name,
                                "task": task,
                                "seed": seed,
                                "fingerprints": expected,
                                "attempts": attempts,
                                "last_error_kind": getattr(last_err, "kind", None),
                                "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            })
                            summary.unresolved.append({
                                "config": config_name, "task": task, "seed": seed,
                                "item_id": item_id, "attempts": attempts,
                                "kind": getattr(last_err, "kind", None),
                            })
                            store.log(config_name, task, f"[s{seed}] {item_id}: INFRA FAILED after {len(attempts)} attempt(s)")
                            continue
                        record = {
                            "record_version": PREDICTION_RECORD_VERSION,
                            "status": "ok",
                            "item_id": item_id,
                            "group_id": item.get("group_id"),
                            "config": config_name,
                            "task": task,
                            "kind": item.get("kind", "na"),
                            "seed": seed,
                            "item_seed": result.item_seed,
                            "fingerprints": expected,
                            "model": {
                                "repo": model_cfg.repo,
                                "revision": model_cfg.revision,
                                "reasoning_mode": model_cfg.reasoning_mode,
                            },
                            "prompt_text": result.prompt_text,
                            "prompt_token_ids": result.prompt_token_ids,
                            "generated_token_ids": result.generated_token_ids,
                            "output_text": result.output_text,
                            "eos_token_strings": eos_strings,
                            "finish_reason": result.finish_reason,
                            "n_prompt_tokens": result.n_prompt_tokens,
                            "n_generated_tokens": result.n_generated_tokens,
                            "wall_seconds": result.wall_seconds,
                            "backend": result.backend,
                            "tokenizer_used": result.tokenizer_used,
                            "merged_generation_params": result.merged_generation_params,
                            "template": result.extra.get("template"),
                            "attempts": attempts,
                            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        store.save_atomic(rpath, record)
                        counters["completed"] += 1
                        counters["n_tokens_in"] += result.n_prompt_tokens
                        counters["n_tokens_out"] += result.n_generated_tokens
                        n_done_in_seed += 1
                        if n_done_in_seed % log_every == 0:
                            store.log(
                                config_name, task,
                                f"[s{seed}] progress {counters['completed'] + counters['reused']}/"
                                f"{len(runnable_items)} completed, {counters['infra_failed']} infra-failed, "
                                f"tokens_in={counters['n_tokens_in']}, tokens_out={counters['n_tokens_out']}, "
                                f"elapsed={time.monotonic() - t_seed_start:.0f}s",
                            )
                    counters["elapsed_seconds"] = round(time.monotonic() - t_seed_start, 1)
                    counters["skipped_unapproved_negatives"] = n_skipped_unapproved
                    store.save_status(config_name, task, seed, counters)
                    store.log(
                        config_name, task,
                        f"[s{seed}] DONE: completed={counters['completed']} reused={counters['reused']} "
                        f"excluded_context={counters['excluded_context']} infra_failed={counters['infra_failed']} "
                        f"skipped_unapproved_negatives={counters['skipped_unapproved_negatives']} "
                        f"tokens_in={counters['n_tokens_in']} tokens_out={counters['n_tokens_out']} "
                        f"elapsed={counters['elapsed_seconds']}s",
                    )
                summary.counters[f"{config_name}/{task}"] = {
                    "seeds": seeds,
                    "n_items": len(items),
                    "n_skipped_unapproved_negatives": n_skipped_unapproved,
                }
        finally:
            backend.close()

    if resume:
        summary.counters["resume"] = True
    summary.counters["n_unresolved"] = len(summary.unresolved)
    return summary
