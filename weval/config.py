"""Configuration loading, validation and fingerprinting.

All configuration is YAML under configs/. A canonical-JSON SHA-256 fingerprint
of the relevant config objects is part of every prediction record, so cached
results can only be reused when data, prompt, model revision, template,
generation settings, parser and scorer versions all match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Generation settings shared by every configuration (controlled comparison).
# These are an experimental protocol, NOT per-model recommended settings.
GENERATION_PARAMS: Dict[str, Any] = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "num_beams": 1,
    "use_cache": True,
    "dtype": "bfloat16",
    "quantization": None,
    "backend": "transformers",
    "batch_size": 1,
}
GENERATION_PARAMS_VERSION = "generation-params-v1"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """SHA-256 of the canonical JSON of obj."""
    return sha256_hex(canonical_json(obj))


@dataclass
class ModelCfg:
    name: str
    repo: str
    revision: str
    tokenizer_revision: str
    license: str
    gated: bool
    gated_note: Optional[str]
    reasoning_mode: str  # prefilled_thinking | optional_thinking | thinking_disabled | no_thinking_channel
    template_kwargs: Dict[str, Any] = field(default_factory=dict)
    template_file: Optional[str] = None
    template_sha256: Optional[str] = None
    context_limit_source: str = "model_config.max_position_embeddings"
    param_exclusions: List[str] = field(default_factory=list)


@dataclass
class TaskCfg:
    name: str
    adapter: str
    instruction_version: str
    max_new_tokens: int
    scorer: str
    parser: str
    # Optional: tasks whose prompt comes from vendored official code
    # (e.g. ProcessBench critique template) have no inline instruction.
    instruction: Optional[str] = None
    relaxed_parser: Optional[str] = None
    relation_note: Optional[str] = None
    answer_format: Optional[str] = None


@dataclass
class ThresholdsCfg:
    status: str
    evidence_worker: Dict[str, Any]
    math_checker: Dict[str, Any]
    tie_break: str


@dataclass
class PlanCfg:
    stage: str
    configs: List[str]
    tasks: List[str]
    require_approved_negatives: bool
    require_locked_templates: bool
    require_locked_thresholds: bool
    max_attempts_per_item: int
    deadline_seconds: float
    report_incomplete_on_unresolved: bool


@dataclass
class ConfigBundle:
    root: Path
    models: Dict[str, ModelCfg]
    tasks: Dict[str, TaskCfg]
    thresholds: ThresholdsCfg
    splits: Dict[str, Any]
    plans: Dict[str, PlanCfg]

    def model(self, name: str) -> ModelCfg:
        if name not in self.models:
            raise KeyError(f"unknown model configuration {name!r}; available: {sorted(self.models)}")
        return self.models[name]

    def task(self, name: str) -> TaskCfg:
        if name not in self.tasks:
            raise KeyError(f"unknown task {name!r}; available: {sorted(self.tasks)}")
        return self.tasks[name]

    def plan(self, stage: str) -> PlanCfg:
        if stage not in self.plans:
            raise KeyError(f"unknown plan/stage {stage!r}; available: {sorted(self.plans)}")
        return self.plans[stage]

    def validate(self) -> List[str]:
        """Cross-checks; returns a list of problems (empty = ok)."""
        problems: List[str] = []
        for plan in self.plans.values():
            for c in plan.configs:
                if c not in self.models:
                    problems.append(f"plan {plan.stage} references unknown model config {c!r}")
            for t in plan.tasks:
                if t not in self.tasks:
                    problems.append(f"plan {plan.stage} references unknown task {t!r}")
        return problems


def load_configs(root: Path) -> ConfigBundle:
    def y(path: Path) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    root = Path(root)
    models_raw = y(root / "configs" / "models.yaml")["models"]
    models = {name: ModelCfg(name=name, **kw) for name, kw in models_raw.items()}

    tasks_raw = y(root / "configs" / "tasks.yaml")
    tasks: Dict[str, TaskCfg] = {}
    for name, kw in tasks_raw.items():
        if name == "task_config_version":
            continue
        tasks[name] = TaskCfg(name=name, **kw)

    thr_raw = y(root / "configs" / "thresholds.yaml")
    thresholds = ThresholdsCfg(
        status=thr_raw.get("status", "draft_unconfirmed"),
        evidence_worker=thr_raw.get("evidence_worker", {}),
        math_checker=thr_raw.get("math_checker", {}),
        tie_break=thr_raw.get("tie_break", "lowest_mean_cproxy"),
    )

    splits = y(root / "configs" / "splits.yaml")

    plans: Dict[str, PlanCfg] = {}
    for p in sorted((root / "configs" / "plans").glob("*.yaml")):
        raw = y(p)
        plans[p.stem] = PlanCfg(
            stage=raw["stage"],
            configs=raw["configs"],
            tasks=raw["tasks"],
            require_approved_negatives=bool(raw.get("require_approved_negatives", False)),
            require_locked_templates=bool(raw.get("require_locked_templates", False)),
            require_locked_thresholds=bool(raw.get("require_locked_thresholds", False)),
            max_attempts_per_item=int(raw.get("max_attempts_per_item", 3)),
            deadline_seconds=float(raw.get("deadline_seconds", 3600)),
            report_incomplete_on_unresolved=bool(raw.get("report_incomplete_on_unresolved", True)),
        )

    bundle = ConfigBundle(
        root=root,
        models=models,
        tasks=tasks,
        thresholds=thresholds,
        splits=splits,
        plans=plans,
    )
    problems = bundle.validate()
    if problems:
        raise ValueError("config validation failed:\n" + "\n".join(f" - {p}" for p in problems))
    return bundle
