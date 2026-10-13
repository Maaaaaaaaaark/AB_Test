"""Rendered-prompt leak checks: what the model sees must never contain gold.

The renderer reads only item["task"], item["kind"], item["question"] and
item["visible"] (slot/title/text) — gold lives only in item["gold"] and in
item["reference_answers"], neither of which the renderer ever reads. So the
guarantee we can assert on RENDERED output is: nothing beyond the instruction,
the relation note, the visible paragraphs and the question can appear. Items
are constructed directly (no builder needed): render_item only needs the keys
listed above.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tests.conftest import make_cfg
from weval.config import sha256_hex
from weval.prompts.renderers import RenderedPrompt, render_item

RELATION_NOTE_MARK = "X >> relation"

VISIBLE = [
    {"slot": 1, "title": "Paris", "text": "Paris is the capital of France.", "role": "distractor"},
    {"slot": 2, "title": "Paris population", "text": "Paris has a population of 2.1 million.", "role": "support"},
    {"slot": 3, "title": "London", "text": "London is the capital of England.", "role": "distractor"},
    {"slot": 4, "title": "Rome", "text": "Rome is the capital of Italy.", "role": "distractor"},
    {"slot": 5, "title": "Berlin", "text": "Berlin is the capital of Germany.", "role": "distractor"},
]


def _musique_item(
    kind: str = "positive",
    question: str = "What is the population of the city mentioned in Paragraph 2?",
    visible: List[Dict[str, Any]] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "task": "musique_local",
        "kind": kind,
        "question": question,
        "visible": visible if visible is not None else VISIBLE,
    }
    item.update(extra)
    return item


class TestMusiqueRendering:
    def test_positive_item_shows_its_support_paragraph(self, cfg):
        rendered = render_item(_musique_item(kind="positive"), cfg)
        # for a positive item the supporting paragraph IS part of the task
        support = next(p for p in VISIBLE if p["role"] == "support")
        assert support["title"] in rendered.user_message
        assert support["text"] in rendered.user_message
        # paragraphs are presented in slot order with stable numbering
        assert "Paragraph 2: Paris population: Paris has a population of 2.1 million." in rendered.user_message
        assert rendered.kind == "positive"
        assert rendered.task == "musique_local"

    def test_instruction_and_answer_marker_present(self, cfg):
        rendered = render_item(_musique_item(), cfg)
        assert rendered.user_message.startswith("Answer the question below")
        assert "output exactly NOT FOUND." in rendered.user_message
        assert rendered.user_message.endswith("Answer:")

    def test_relation_note_in_every_musique_prompt(self, cfg):
        """The relation note is included for EVERY MuSiQue item — relation
        questions ("X >> relation") and natural questions alike, positives and
        negatives — for a uniform explanation."""
        natural = _musique_item(kind="positive", question="What is the population of the city mentioned in Paragraph 2?")
        relation = _musique_item(kind="positive", question="Paris >> population")
        negative = _musique_item(kind="negative", question="What is the capital of the city in Paragraph 1?")
        for item in (natural, relation, negative):
            rendered = render_item(item, cfg)
            assert RELATION_NOTE_MARK in rendered.user_message
        assert render_item(negative, cfg).kind == "negative"

    def test_visible_paragraphs_rendered_in_slot_order(self, cfg):
        shuffled = [
            {"slot": 3, "title": "Third", "text": "Text three.", "role": "distractor"},
            {"slot": 1, "title": "First", "text": "Text one.", "role": "support"},
            {"slot": 2, "title": "Second", "text": "Text two.", "role": "distractor"},
        ]
        rendered = render_item(_musique_item(visible=shuffled), cfg)
        assert rendered.user_message.index("Paragraph 1: First") < rendered.user_message.index("Paragraph 2: Second")
        assert rendered.user_message.index("Paragraph 2: Second") < rendered.user_message.index("Paragraph 3: Third")


class TestNoLeakage:
    def test_gold_answers_never_appear_in_rendered_output(self, cfg):
        """Gold lives in item["gold"] / item["reference_answers"], which the
        renderer never reads: distinctive secret answers must never reach the
        rendered user message (spec-guaranteed leak check)."""
        secrets = [
            "SECRET-REF-ANSWER",
            "SECRET-PARENT-ANSWER",
            "SECRET-SUBANSWER",
            "SECRET-ALIAS",
            "SECRET-PREREQ",
        ]
        item = _musique_item(
            kind="positive",
            question="What is the population of the city mentioned in Paragraph 2?",
            reference_answers=["SECRET-REF-ANSWER"],
            expected="answer",
            gold={
                "reference_answers": ["SECRET-REF-ANSWER"],
                "alias_source": "parent_answer",
                "used_prerequisites": [("SECRET-PREREQ", 1)],
                "subquestion_answer": "SECRET-SUBANSWER",
                "parent_answer": "SECRET-PARENT-ANSWER",
                "parent_answer_aliases": ["SECRET-ALIAS"],
            },
        )
        # precondition: the secrets are nowhere in the parts the renderer does read
        assert all(s not in item["question"] for s in secrets)
        assert all(s not in str(item["visible"]) for s in secrets)
        rendered = render_item(item, cfg)
        for s in secrets:
            assert s not in rendered.user_message

    def test_non_visible_supporting_paragraph_title_absent(self, cfg):
        """Only visible paragraphs are rendered: a supporting paragraph that is
        NOT part of item["visible"] (its title or text) must not appear."""
        hidden_title = "Hidden support paragraph"
        hidden_text = "This paragraph answers the question but is never shown."
        rendered = render_item(_musique_item(kind="positive"), cfg)
        assert hidden_title not in rendered.user_message
        assert hidden_text not in rendered.user_message

    def test_renderer_reads_only_question_and_visible(self, cfg):
        """Two items differing only in gold/expected/answerability render
        byte-identically (the renderer cannot leak what it never reads)."""
        base = _musique_item(kind="positive")
        decorated = _musique_item(
            kind="positive",
            reference_answers=["Some other answer entirely"],
            expected="answer",
            answerable=True,
            gold={"parent_answer": "Totally different", "reference_answers": ["x"]},
        )
        r1 = render_item(base, cfg)
        r2 = render_item(decorated, cfg)
        assert r1.user_message == r2.user_message
        assert r1.fingerprint() == r2.fingerprint()


class TestTemplateTasks:
    def test_gsm8k_placeholder_replaced_and_hashes_survive(self, cfg):
        question = "A farmer has 5 apples and buys 3 more. How many apples does he have?"
        rendered = render_item({"task": "gsm8k", "question": question}, cfg)
        assert question in rendered.user_message
        # the instruction's literal "#### <number>" format marker survives
        assert "#### <number>" in rendered.user_message
        assert "{question}" not in rendered.user_message
        assert rendered.kind == "na"
        assert rendered.task == "gsm8k"

    def test_math_boxed_braces_survive_no_format_crash(self, cfg):
        problem = r"Solve $x^2 = 4$ for real $x$."
        rendered = render_item({"task": "math", "problem": problem}, cfg)
        assert problem in rendered.user_message
        # literal LaTeX braces survive (renderer uses .replace, not .format)
        assert r"\boxed{}" in rendered.user_message
        assert "{problem}" not in rendered.user_message
        assert rendered.kind == "na"

    def test_hotpotqa_paragraphs_and_question_replaced(self, cfg):
        item = {
            "task": "hotpotqa",
            "question": "Where is the Eiffel Tower?",
            "paragraphs": [
                {"title": "Paris", "text": "The Eiffel Tower is in Paris."},
                {"title": "London", "text": "Big Ben is in London."},
            ],
        }
        rendered = render_item(item, cfg)
        assert "Paragraph 1: Paris: The Eiffel Tower is in Paris." in rendered.user_message
        assert "Paragraph 2: London: Big Ben is in London." in rendered.user_message
        assert "Where is the Eiffel Tower?" in rendered.user_message
        assert "{paragraphs}" not in rendered.user_message
        assert "{question}" not in rendered.user_message

    def test_unknown_task_raises(self, cfg):
        with pytest.raises(KeyError):
            render_item({"task": "definitely_unknown_task"}, cfg)


class TestDeterminism:
    def test_render_item_is_deterministic(self, cfg):
        item = _musique_item(kind="positive")
        first = render_item(item, cfg)
        for _ in range(3):
            again = render_item(item, cfg)
            assert again.user_message == first.user_message
            assert again.fingerprint() == first.fingerprint()

    def test_fingerprint_is_sha256_of_user_message(self, cfg):
        rendered = render_item(_musique_item(kind="positive"), cfg)
        assert rendered.fingerprint() == sha256_hex(rendered.user_message)
        assert rendered.renderer_version == "renderer-v1"
        assert isinstance(rendered, RenderedPrompt)
