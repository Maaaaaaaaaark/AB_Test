"""Shared fixtures: in-memory config bundle + synthetic MuSiQue records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from weval.config import (
    GENERATION_PARAMS_VERSION,
    ConfigBundle,
    ModelCfg,
    PlanCfg,
    TaskCfg,
    ThresholdsCfg,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_task(**kw: Any) -> TaskCfg:
    base = dict(
        adapter="musique",
        instruction_version="test-v1",
        instruction="Answer the question below using only the provided information.\n"
        "Give a short answer. If the provided information is insufficient to answer the question, output exactly NOT FOUND.",
        relation_note='In questions, an expression like "X >> relation" asks for that relation of X.',
        answer_format="short_answer_or_not_found",
        max_new_tokens=2048,
        scorer="musique_em_f1",
        parser="musique_answer",
    )
    name = kw.pop("name", "musique_local")
    base.update(kw)
    return TaskCfg(name=name, **base)


def make_model(**kw: Any) -> ModelCfg:
    base = dict(
        repo="test/test-model",
        revision="1234567890abcdef",
        tokenizer_revision="1234567890abcdef",
        license="apache-2.0",
        gated=False,
        gated_note=None,
        reasoning_mode="no_thinking_channel",
        template_kwargs={},
        template_file=None,
        template_sha256=None,
    )
    name = kw.pop("name", "m1")
    base.update(kw)
    return ModelCfg(name=name, **base)


def make_cfg(**kw: Any) -> ConfigBundle:
    tasks = kw.pop("tasks", None) or {
        "musique_local": make_task(),
        "processbench": make_task(
            name="processbench", adapter="processbench", instruction=None,
            relation_note=None, max_new_tokens=4096, scorer="processbench_official",
            parser="processbench_boxed_index",
        ),
        "gsm8k": make_task(
            name="gsm8k", adapter="gsm8k", instruction="Question: {question}\n\nSolve the problem step by step, and put your final numeric answer on a line by itself in the format #### <number>.\nAnswer:",
            relation_note=None, max_new_tokens=4096, scorer="gsm8k_official", parser="gsm8k_official_regex",
        ),
        "math": make_task(
            name="math", adapter="math", instruction="Problem:\n{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:",
            relation_note=None, max_new_tokens=4096, scorer="math_equivalence", parser="math_boxed",
        ),
        "hotpotqa": make_task(
            name="hotpotqa", adapter="hotpotqa", instruction="Answer the following question using the information in the paragraphs below. Give a short answer.\n\n{paragraphs}\n\nQuestion: {question}\nAnswer:",
            relation_note=None, max_new_tokens=4096, scorer="hotpotqa_answer_f1", parser="hotpotqa_free_text",
        ),
    }
    models = kw.pop("models", None) or {
        "m1": make_model(),
        "m2": make_model(name="m2", reasoning_mode="prefilled_thinking"),
    }
    plans = kw.pop("plans", None) or {
        "smoke": PlanCfg(
            stage="smoke", configs=["m1"], tasks=["musique_local"],
            require_approved_negatives=False, require_locked_templates=False,
            require_locked_thresholds=False, max_attempts_per_item=3,
            deadline_seconds=60.0, report_incomplete_on_unresolved=True,
        ),
        "pilot": PlanCfg(
            stage="pilot", configs=["m1"], tasks=["musique_local"],
            require_approved_negatives=True, require_locked_templates=False,
            require_locked_thresholds=False, max_attempts_per_item=3,
            deadline_seconds=60.0, report_incomplete_on_unresolved=True,
        ),
        "selection": PlanCfg(
            stage="selection", configs=["m1"], tasks=["musique_local"],
            require_approved_negatives=True, require_locked_templates=False,
            require_locked_thresholds=True, max_attempts_per_item=3,
            deadline_seconds=60.0, report_incomplete_on_unresolved=True,
        ),
    }
    splits = kw.pop("splits", None) or {
        "stages": {
            "smoke": {"musique_parents": 2, "processbench_per_subset": 2, "reference_per_dataset": 2},
            "pilot": {"musique_parents": 4, "processbench_per_subset": 5, "reference_per_dataset": 10},
            "selection": {"musique_parents": 6, "processbench_per_subset": 25},
            "holdout": {"musique_parents": 6, "processbench_per_subset": 25},
            "reference": {"reference_per_dataset": 100},
        },
        "seeds": {
            "smoke": [0], "pilot": [0], "selection": [0, 1, 2], "holdout": [0, 1, 2], "reference": [0],
        },
        "split_seeds": {
            "musique_subquestion_choice": 20261013,
            "processbench_group_order": 20261013,
        },
    }
    thresholds = kw.pop("thresholds", None) or ThresholdsCfg(
        status="draft_unconfirmed",
        evidence_worker={"answerable_f1": 0.80, "false_abstention_rate": 0.10, "reviewed_negative_correct_abstention": 0.90},
        math_checker={"macro_f1": 0.60, "macro_correct_solution_acc": 0.50, "macro_error_solution_acc": 0.50},
        tie_break="lowest_mean_cproxy",
    )
    return ConfigBundle(
        root=kw.pop("root", PROJECT_ROOT),
        models=models, tasks=tasks, thresholds=thresholds, splits=splits, plans=plans,
    )


def musique_record(
    pid: str = "2hop__1_2",
    subquestions: List[Dict[str, Any]] | None = None,
    paragraphs: List[Dict[str, Any]] | None = None,
    answer: str = "Final Answer",
    aliases: List[str] | None = None,
    answerable: bool = True,
) -> Dict[str, Any]:
    """A synthetic MuSiQue-Ans record.

    Default: 2-hop, subq0 (natural) supported by p0, subq1 (relation, depends
    on #1) supported by p1, 6 distractors p2..p7 (non-supporting anywhere).
    """
    subquestions = subquestions or [
        {"id": 1, "question": "What is the capital of France?", "answer": "Paris", "paragraph_support_idx": 0},
        {"id": 2, "question": "#1 >> population", "answer": "2.1 million", "paragraph_support_idx": 1},
    ]
    paragraphs = paragraphs or [
        {"idx": 0, "title": "Paris", "paragraph_text": "Paris is the capital of France.", "is_supporting": True},
        {"idx": 1, "title": "Paris population", "paragraph_text": "Paris has a population of 2.1 million.", "is_supporting": True},
        {"idx": 2, "title": "London", "paragraph_text": "London is the capital of England.", "is_supporting": False},
        {"idx": 3, "title": "Rome", "paragraph_text": "Rome is the capital of Italy.", "is_supporting": False},
        {"idx": 4, "title": "Berlin", "paragraph_text": "Berlin is the capital of Germany.", "is_supporting": False},
        {"idx": 5, "title": "Madrid", "paragraph_text": "Madrid is the capital of Spain.", "is_supporting": False},
        {"idx": 6, "title": "Tokyo", "paragraph_text": "Tokyo is the capital of Japan.", "is_supporting": False},
        {"idx": 7, "title": "Beijing", "paragraph_text": "Beijing is the capital of China.", "is_supporting": False},
    ]
    return {
        "id": pid,
        "question": f"What is the question behind {pid}?",
        "question_decomposition": subquestions,
        "answer": answer,
        "answer_aliases": aliases if aliases is not None else [],
        "answerable": answerable,
        "paragraphs": paragraphs,
    }


@pytest.fixture
def cfg() -> ConfigBundle:
    return make_cfg()


@pytest.fixture
def record() -> Dict[str, Any]:
    return musique_record()
