"""Regression tests for the problems found in the first smoke run.

1. Gemma stops on <end_of_turn>, which is NOT tokenizer.eos_token ("<eos>"):
   trailing stop strings must all be trimmed, including when old records are
   re-scored (declared stop_strings are merged in at report time).
2. Templates that render {{ bos_token }} (Gemma 3, Llama 3.2, R1-Distill) must
   receive the BOS string; prompts are tokenized with add_special_tokens=False,
   so otherwise the prompt has no BOS token at all. A runtime check asserts
   exactly one leading BOS.
3. MuSiQue paragraphs are presented in a seeded random order with seeded
   distractors (the support paragraph used to be ~always last).
4. MuSiQue positives are screened (gold answer visible in the support
   paragraph) and human-reviewed before formal stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from tests.conftest import PROJECT_ROOT, make_cfg, make_model, musique_record
from weval.config import ModelCfg, PlanCfg
from weval.datasets.musique import (
    build_musique,
    build_pair,
    build_positive,
    eligible_subquestions,
    item_content_fingerprint,
    load_positive_review_map,
    paragraph_plan,
    support_contains_answer,
    write_review_file,
)
from weval.metrics.report import _parse_and_score
from weval.models.templates import render_prompt, render_prompt_runtime
from weval.models.transformers_backend import (
    PromptContractError,
    check_prompt_bos,
    collect_stop_strings,
)
from weval.parsing.hotpotqa_answer import parse_hotpotqa_answer
from weval.parsing.musique_answer import parse_musique_answer
from weval.parsing.reasoning import merge_stop_strings
from weval.review import ReviewError, apply_decision, positive_review_summary
from weval.run.runner import PlanGateError, run_plan
from weval.scoring.hotpotqa_score import score_hotpotqa

MODELS = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]


def _model(name: str) -> ModelCfg:
    return ModelCfg(name=name, **MODELS[name])


# --------------------------------------------------------------------------
# 1. stop strings
# --------------------------------------------------------------------------

class _FakeTokenizer:
    def __init__(self, eos_token: str, id_to_token: Dict[int, str], bos_token=None, bos_token_id=None):
        self.eos_token = eos_token
        self._map = id_to_token
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id

    def convert_ids_to_tokens(self, i: int) -> str:
        return self._map[i]


class _FakeGenCfg:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class TestStopStrings:
    def test_gemma_generation_config_end_of_turn_is_collected(self):
        tok = _FakeTokenizer("<eos>", {1: "<eos>", 106: "<end_of_turn>"})
        stops = collect_stop_strings(tok, _FakeGenCfg([1, 106]))
        assert stops == ["<eos>", "<end_of_turn>"]

    def test_int_eos_id_and_declared_strings_are_merged_without_duplicates(self):
        tok = _FakeTokenizer("<|im_end|>", {151645: "<|im_end|>"})
        stops = collect_stop_strings(tok, _FakeGenCfg(151645), ["<|im_end|>", "<|endoftext|>"])
        assert stops == ["<|im_end|>", "<|endoftext|>"]

    def test_every_model_declares_stop_strings(self):
        for name, raw in MODELS.items():
            assert raw.get("stop_strings"), name
        assert "<end_of_turn>" in MODELS["gemma-3-4b-it"]["stop_strings"]
        assert "<|eot_id|>" in MODELS["llama-3.2-3b-instruct"]["stop_strings"]

    @pytest.mark.parametrize("raw", ["NOT FOUND<end_of_turn>", "NOT FOUND\n<end_of_turn>"])
    def test_gemma_not_found_is_recognized(self, raw):
        stops = merge_stop_strings(["<eos>"], MODELS["gemma-3-4b-it"]["stop_strings"])
        parsed = parse_musique_answer(raw, "no_thinking_channel", stops)
        assert parsed.diagnostics["is_not_found"] is True
        assert parsed.value == "NOT FOUND"

    def test_smoke_bug_without_declared_stops_is_reproduced(self):
        # what the smoke run did: only tokenizer.eos_token was recorded
        parsed = parse_musique_answer("NOT FOUND<end_of_turn>", "no_thinking_channel", ["<eos>"])
        assert parsed.diagnostics["is_not_found"] is False

    def test_report_rescoring_merges_declared_stop_strings(self):
        """Old smoke records only carry ['<eos>']; re-scoring must still trim
        <end_of_turn> because the model config declares it."""
        cfg = make_cfg()
        task_cfg = cfg.task("musique_local")
        item = build_positive(musique_record(), 1, "debug").to_dict()
        record = {"output_text": "NOT FOUND<end_of_turn>", "eos_token_strings": ["<eos>"]}
        _, score = _parse_and_score(
            task_cfg, "no_thinking_channel", item, record,
            MODELS["gemma-3-4b-it"]["stop_strings"],
        )
        assert score.detail["false_abstention"] is True
        _, old = _parse_and_score(task_cfg, "no_thinking_channel", item, record)
        assert old.detail["false_abstention"] is False  # the old behaviour

    def test_gemma_one_word_hotpot_answer_gets_full_credit(self):
        stops = merge_stop_strings(["<eos>"], MODELS["gemma-3-4b-it"]["stop_strings"])
        parsed = parse_hotpotqa_answer("Five<end_of_turn>", "no_thinking_channel", stops)
        item = {"item_id": "h1", "group_id": "h1", "split": "reference", "gold": {"answer": "five"}}
        score = score_hotpotqa(item, parsed)
        assert score.detail["em"] == 1 and score.value == 1.0


# --------------------------------------------------------------------------
# 2. BOS
# --------------------------------------------------------------------------

MSGS = [{"role": "user", "content": "Hello?"}]
# Stand-in with the same opening as the Gemma 3 template ({{ bos_token }}
# first); the real template is vendored by `weval precheck` on the cluster.
GEMMA_LIKE = "{{ bos_token }}<start_of_turn>user\n{{ messages[0]['content'] }}<end_of_turn>\n<start_of_turn>model\n"


class TestBos:
    def test_gated_models_pin_their_bos_string(self):
        assert MODELS["gemma-3-4b-it"]["template_kwargs"]["bos_token"] == "<bos>"
        assert MODELS["llama-3.2-3b-instruct"]["template_kwargs"]["bos_token"] == "<|begin_of_text|>"

    def test_vendored_templates_that_render_bos_have_a_pinned_bos(self):
        for name, raw in MODELS.items():
            path = PROJECT_ROOT / raw["template_file"]
            if path.exists() and "bos_token" in path.read_text(encoding="utf-8"):
                assert raw.get("template_kwargs", {}).get("bos_token"), name

    def test_gemma_prompt_starts_with_bos(self):
        rt = render_prompt_runtime(_model("gemma-3-4b-it"), MSGS, GEMMA_LIKE)
        assert rt.text.startswith("<bos><start_of_turn>user\nHello?")
        assert rt.references_bos and rt.bos_source == "template_kwargs"

    def test_tokenizer_bos_is_the_fallback_when_nothing_is_pinned(self):
        unpinned = make_model(name="x")
        rt = render_prompt_runtime(unpinned, MSGS, GEMMA_LIKE, bos_fallback="<bos>")
        assert rt.text.startswith("<bos>") and rt.bos_source == "tokenizer"
        none = render_prompt_runtime(unpinned, MSGS, GEMMA_LIKE)
        assert none.text.startswith("<start_of_turn>") and none.bos_source == "none"

    def test_templates_without_bos_are_not_checked(self):
        rt = render_prompt(_model("qwen3-4b-thinking-off"), MSGS, PROJECT_ROOT)
        assert rt.references_bos is False
        assert check_prompt_bos([151644, 872], None, rt.references_bos)["bos_check"] == "not_applicable"

    def test_r1_template_references_bos_and_renders_it(self):
        rt = render_prompt(_model("deepseek-r1-distill-qwen-1.5b"), MSGS, PROJECT_ROOT)
        assert rt.references_bos and rt.text.startswith("<｜begin▁of▁sentence｜>")

    def test_missing_bos_aborts(self):
        with pytest.raises(PromptContractError):
            check_prompt_bos([105, 2364], 2, True, "gemma-3-4b-it")

    def test_double_bos_aborts(self):
        with pytest.raises(PromptContractError):
            check_prompt_bos([2, 2, 105], 2, True, "gemma-3-4b-it")

    def test_single_bos_passes(self):
        assert check_prompt_bos([2, 105, 2364], 2, True)["bos_check"] == "ok"

    def test_prompt_contract_error_is_not_retried_as_infra(self):
        from weval.models.backend import InfrastructureError

        assert not issubclass(PromptContractError, InfrastructureError)


# --------------------------------------------------------------------------
# 3. MuSiQue paragraph order
# --------------------------------------------------------------------------

def _many_records(n: int = 60) -> List[Dict[str, Any]]:
    return [musique_record(pid=f"2hop__{i}_{i + 1000}") for i in range(n)]


class TestParagraphOrder:
    def test_plan_is_deterministic(self):
        rec = musique_record()
        assert paragraph_plan(rec, 1) == paragraph_plan(rec, 1)
        assert paragraph_plan(rec, 1, 5)["order_seed"] == 5

    def test_support_slot_is_spread_over_positions(self):
        slots = [build_positive(r, 1, "debug").gold["paragraph_plan"]["support_slot"] for r in _many_records()]
        # builder v1 put the support at the last slot almost always
        assert len(set(slots)) >= 4
        assert slots.count(5) < len(slots) * 0.5

    def test_distractors_are_a_seeded_sample_not_the_lowest_idx(self):
        firsts = {tuple(sorted(paragraph_plan(r, 1)["distractor_idxs"])) for r in _many_records()}
        assert len(firsts) > 1

    def test_slots_follow_the_plan_and_negative_shares_the_order(self):
        rec = musique_record()
        pos = build_positive(rec, 1, "debug")
        neg, reason = build_pair(rec, 1, pos, "debug")
        assert reason is None
        plan = pos.gold["paragraph_plan"]
        assert [p.source_idx for p in pos.visible] == plan["slot_order"]
        assert [p.slot for p in pos.visible] == [1, 2, 3, 4, 5]
        diff = [(a.slot, b.slot) for a, b in zip(pos.visible, neg.visible) if a.source_idx != b.source_idx]
        assert diff == [(plan["support_slot"], plan["support_slot"])]

    def test_build_report_records_support_slot_distribution(self):
        recs = _many_records(20)
        items, report, _ = build_musique(recs, "debug", 1)
        assert sum(report.support_slot_distribution.values()) == report.n_positives_built == 20


# --------------------------------------------------------------------------
# 4. positive screening + review
# --------------------------------------------------------------------------

def _record_answer_not_in_support() -> Dict[str, Any]:
    rec = musique_record(pid="2hop__nosupport")
    for p in rec["paragraphs"]:
        if p["idx"] == 1:
            p["paragraph_text"] = "Paris is a large city with many residents."  # no "2.1 million"
    return rec


class TestPositiveScreening:
    def test_answer_missing_from_support_is_not_eligible(self):
        rec = _record_answer_not_in_support()
        assert support_contains_answer(rec, 1) is False
        assert 1 not in eligible_subquestions(rec)

    def test_answer_in_support_title_counts_as_visible(self):
        rec = musique_record(pid="2hop__title")
        for p in rec["paragraphs"]:
            if p["idx"] == 1:
                p["title"] = "2.1 million"
                p["paragraph_text"] = "This is the number of people living in Paris."
        assert support_contains_answer(rec, 1) is True

    def test_parent_without_any_supported_answer_reports_reason(self):
        rec = _record_answer_not_in_support()
        for p in rec["paragraphs"]:
            if p["idx"] == 0:
                p["title"] = "Europe"
                p["paragraph_text"] = "A city in Europe."
        _, report, _ = build_musique([rec], "debug", 1)
        assert report.n_parents_eligible == 0
        assert report.parent_eligibility_reasons == {"no_subquestion_with_answer_in_support": 1}


def _review_file(tmp_path: Path, rec: Dict[str, Any]):
    pos = build_positive(rec, 1, "selection")
    neg, _ = build_pair(rec, 1, pos, "selection")
    path = tmp_path / "manifests" / "musique_negative_review.json"
    write_review_file(path, [rec], {rec["id"]: pos}, {rec["id"]: neg}, {})
    return path, pos, neg


class TestPositiveReview:
    def test_review_entry_carries_paragraphs_and_positive_fields(self, tmp_path):
        rec = musique_record()
        path, pos, neg = _review_file(tmp_path, rec)
        e = json.loads(path.read_text(encoding="utf-8"))[0]
        assert e["support_paragraph"]["text"] == "Paris has a population of 2.1 million."
        assert e["replacement_paragraph"]["slot"] == e["support_paragraph"]["slot"]
        assert e["positive_review_status"] == "proposed"
        assert e["positive_fingerprint"] == item_content_fingerprint(pos)
        assert e["negative_fingerprint"] == item_content_fingerprint(neg)

    def test_positive_decisions(self, tmp_path):
        rec = musique_record()
        path, _, _ = _review_file(tmp_path, rec)
        with pytest.raises(ReviewError):
            apply_decision(path, rec["id"], "exclude", target="positive")  # note required
        e = apply_decision(path, rec["id"], "approve", target="positive", note="paragraph states it")
        assert e["positive_review_status"] == "approved" and e["positive_human_reviewed"]
        # the negative decision is independent
        assert e["review_status"] == "proposed"
        assert positive_review_summary(path)["approved"] == 1
        e = apply_decision(path, rec["id"], "reset", target="positive")
        assert e["positive_review_status"] == "proposed" and not e["positive_human_reviewed"]

    def test_build_applies_positive_decisions(self, tmp_path):
        rec = musique_record()
        path, _, _ = _review_file(tmp_path, rec)
        # unreviewed -> proposed (dev use) / dropped when approval is required
        items, _, _ = build_musique([rec], "selection", 1, positive_reviews=load_positive_review_map(path))
        assert [i.review_status for i in items if i.kind == "positive"] == ["proposed"]
        items, _, _ = build_musique(
            [rec], "selection", 1, positive_reviews=load_positive_review_map(path),
            require_approved_positives=True,
        )
        assert [i for i in items if i.kind == "positive"] == []

        apply_decision(path, rec["id"], "approve", target="positive")
        items, report, _ = build_musique([rec], "selection", 1, positive_reviews=load_positive_review_map(path))
        pos = next(i for i in items if i.kind == "positive")
        assert pos.review_status == "approved" and pos.human_reviewed
        assert report.n_positives_approved == 1

        apply_decision(path, rec["id"], "exclude", target="positive", note="relation not stated")
        items, report, _ = build_musique([rec], "selection", 1, positive_reviews=load_positive_review_map(path))
        assert [i for i in items if i.kind == "positive"] == []
        assert report.exclusions == {"positive_review:relation not stated": 1}


def _pos_item(item_id: str, parent: str, status: str) -> Dict[str, Any]:
    from tests.test_runner import _item

    return _item(item_id, parent, kind="positive", review_status=status)


class TestRunnerPositiveGate:
    def test_unapproved_positives_are_skipped_and_counted(self, tmp_path):
        from tests.test_runner import _runner_cfg, _write_musique_manifest, store_factory

        out_dir = tmp_path / "artifacts"
        _write_musique_manifest(out_dir, [
            _pos_item("p_ok", "P1", "approved"),
            _pos_item("p_new", "P2", "proposed"),
        ])
        cfg = _runner_cfg(require_approved_positives=True)
        factory, backends = store_factory()
        s = run_plan(cfg, "smoke", factory, out_dir, seeds=[0], resume=True)
        assert backends["m1"].calls == ["p_ok"]
        assert s.counters["m1/musique_local"]["n_skipped_unapproved_positives"] == 1

    def test_formal_stage_refuses_unreviewed_positives(self, tmp_path):
        from tests.test_runner import _write_musique_manifest, store_factory

        out_dir = tmp_path / "artifacts"
        mdir = out_dir / "manifests"
        mdir.mkdir(parents=True)
        items = [_pos_item("p_new", "P2", "proposed")]
        (mdir / "musique_selection.json").write_text(json.dumps({"items": items}), encoding="utf-8")
        plan = PlanCfg(
            stage="selection", configs=["m1"], tasks=["musique_local"],
            require_approved_negatives=False, require_locked_templates=False,
            require_locked_thresholds=False, max_attempts_per_item=1,
            deadline_seconds=60.0, report_incomplete_on_unresolved=True,
            require_approved_positives=True,
        )
        splits = {"stages": {"selection": {"musique_parents": 5}}, "seeds": {"selection": [0]}, "split_seeds": {}}
        cfg = make_cfg(plans={"selection": plan}, splits=splits)
        factory, _ = store_factory()
        with pytest.raises(PlanGateError):
            run_plan(cfg, "selection", factory, out_dir, seeds=[0])


# --------------------------------------------------------------------------
# 5. answer-line extraction (MuSiQue local, HotpotQA)
# --------------------------------------------------------------------------

from weval.parsing.answer_line import extract_short_answer  # noqa: E402
from weval.scoring.musique_em_f1 import score_musique  # noqa: E402


class TestAnswerLine:
    @pytest.mark.parametrize(
        "final, expected, source",
        [
            ("NOT FOUND", "NOT FOUND", "single_line"),
            ("Lisa Jackson", "Lisa Jackson", "single_line"),
            ("Paragraph 5 names her.\nAnswer: Lisa Jackson", "Lisa Jackson", "answer_line"),
            ("**Answer:** Five", "Five", "answer_line"),
            ("Some reasoning.\n\n**Answer**: five", "five", "answer_line"),
            ("Some reasoning.\nAnswer:\nFive", "Five", "answer_line"),
            ("Final answer: \\boxed{3}", "3", "answer_line"),
            ("Answer: first guess\nwait\nAnswer: second", "second", "answer_line"),
            ("She is a model.\nShe appeared in the video.", None, "none"),
        ],
    )
    def test_extraction_rules(self, final, expected, source):
        assert extract_short_answer(final) == (expected, source)

    def test_r1_style_not_found_after_reasoning(self):
        raw = "the paragraphs do not say...</think>\n\nNothing states it.\n\nAnswer: NOT FOUND<｜end▁of▁sentence｜>"
        parsed = parse_musique_answer(raw, "prefilled_thinking", ["<｜end▁of▁sentence｜>"])
        assert parsed.diagnostics["is_not_found"] is True
        assert parsed.diagnostics["answer_source"] == "answer_line"

    def test_multiline_reply_without_answer_line_is_invalid_format(self):
        item = build_positive(musique_record(), 1, "debug").to_dict()
        parsed = parse_musique_answer("Paris has many people.\nIt is big.", "no_thinking_channel")
        assert parsed.status == "invalid_format" and parsed.value is None
        score = score_musique(item, parsed)
        assert score.value == 0.0 and score.status == "invalid_format"

    def test_sentence_answers_no_longer_dilute_f1(self):
        # Qwen (thinking off) in the smoke run answered with full sentences
        raw = "The artist is Jessica Lange. She has won five Golden Globe Awards.\nAnswer: five<|im_end|>"
        parsed = parse_hotpotqa_answer(raw, "thinking_disabled", ["<|im_end|>"])
        item = {"item_id": "h", "group_id": "h", "split": "reference", "gold": {"answer": "five"}}
        assert score_hotpotqa(item, parsed).detail["em"] == 1

    def test_instructions_state_the_answer_line_convention(self):
        tasks = yaml.safe_load((PROJECT_ROOT / "configs" / "tasks.yaml").read_text(encoding="utf-8"))
        for t in ("musique_local", "hotpotqa"):
            assert "Answer: <short answer>" in tasks[t]["instruction"], t
        assert tasks["hotpotqa"]["parser"] == "hotpotqa_answer_line"


# --------------------------------------------------------------------------
# 6. GSM8K format exemplar
# --------------------------------------------------------------------------

from weval.config import load_configs  # noqa: E402
from weval.datasets.gsm8k import extract_answer  # noqa: E402
from weval.prompts.renderers import render_item  # noqa: E402


class TestGsm8kExemplar:
    def _item(self) -> Dict[str, Any]:
        return {"item_id": "0", "group_id": "gsm8k:0", "split": "reference", "task": "gsm8k",
                "question": "John has 3 apples and buys 2 more. How many apples does he have?",
                "answer": "3+2=5\n#### 5", "gold": {"extracted_answer": "5"}}

    def test_prompt_contains_one_formatted_example_then_the_question(self):
        cfg = load_configs(PROJECT_ROOT)
        msg = render_item(self._item(), cfg).user_message
        assert "{exemplar" not in msg and "{question}" not in msg
        assert msg.count("#### 72") == 1
        assert msg.index("Natalia sold clips") < msg.index("John has 3 apples")
        assert msg.rstrip().endswith("Answer:")
        # the example demonstrates exactly the official extractor's format
        assert extract_answer(cfg.task("gsm8k").exemplar["answer"]) == "72"

    def test_example_has_no_calculator_annotations(self):
        cfg = load_configs(PROJECT_ROOT)
        assert "<<" not in cfg.task("gsm8k").exemplar["answer"]

    def test_missing_exemplar_with_placeholder_fails_loudly(self):
        from tests.conftest import make_task

        cfg = make_cfg()
        cfg.tasks["gsm8k"] = make_task(
            name="gsm8k", adapter="gsm8k", relation_note=None, scorer="gsm8k_official",
            parser="gsm8k_official_regex",
            instruction="Example: {exemplar_question}\nQuestion: {question}\nAnswer:",
        )
        with pytest.raises(ValueError):
            render_item(self._item(), cfg)
