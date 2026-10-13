"""Reasoning-region splitting per reasoning_mode (weval/parsing/reasoning.py).

Encodes the documented contract for every reasoning_mode:

  * prefilled_thinking  - the prompt ends inside an OPEN <think> region
                          (DeepSeek-R1 style). The generated text carries
                          </think> WITHOUT an opening tag; a close tag means
                          the answer channel follows. No close at EOS =>
                          missing final answer.
  * optional_thinking   - Qwen3 (thinking on): the model may emit
                          <think>...</think> itself or answer directly. An
                          opening tag with no close at EOS => missing final.
  * thinking_disabled   - Qwen3 (thinking off): the prompt prefills an empty
                          CLOSED think block, so the output is the answer
                          channel. A phantom </think> splits at the LAST
                          close: text before it is discarded, text after it
                          is the answer (never graded as reasoning).
  * no_thinking_channel - Gemma / Llama: the whole output is the answer
                          channel.

Contract assertions covered here:
  * final-answer extraction happens only AFTER a closed reasoning region;
  * an open reasoning region at EOS is missing_final (value None, never a
    crash and never an answer attempt);
  * answers are never parsed from inside the reasoning region;
  * trailing end-of-sequence token strings are stripped before splitting.
"""

from __future__ import annotations

import pytest

from weval.parsing.musique_answer import parse_musique_answer
from weval.parsing.reasoning import (
    REASONING_PARSER_VERSION,
    ReasoningSplit,
    final_region,
    split_reasoning,
    strip_eos,
)

EOS_IM_END = ["<|im_end|>"]
EOS_ENDOFTEXT = ["<|endoftext|>"]


class TestStripEos:
    def test_strips_single_trailing_eos(self):
        assert strip_eos("Paris<|im_end|>", EOS_IM_END) == "Paris"

    def test_strips_repeated_eos_tokens_and_whitespace(self):
        assert strip_eos("42<|endoftext|><|endoftext|>", EOS_ENDOFTEXT) == "42"
        assert strip_eos("42<|endoftext|>  \n", EOS_ENDOFTEXT) == "42"

    def test_longest_eos_first_and_never_interior(self):
        # longer eos string must be stripped before its shorter suffix, so a
        # token that is itself a suffix of another eos string is not half-cut
        assert strip_eos("ax<|end|>", ["<|end|>", "x<|end|>"]) == "a"
        # interior eos strings are preserved (only trailing ones stripped)
        assert strip_eos("a<|im_end|>b<|im_end|>", EOS_IM_END) == "a<|im_end|>b"

    def test_unknown_eos_strings_trim_whitespace_only(self):
        # dry-run (no eos strings): trailing whitespace trimmed, leading
        # whitespace and interior untouched
        assert strip_eos("answer \n\t", None) == "answer"
        assert strip_eos("  answer \n", None) == "  answer"
        assert strip_eos("  answer \n", []) == "  answer"


class TestPrefilledThinking:
    def test_deepseek_prefill_prefix_with_close(self):
        # prompt prefill leaves "<|Assistant|><think>\n" in the text; the
        # generated </think> closes the region and the answer follows it
        split = split_reasoning("<|Assistant|><think>\nreasoning</think>Paris", "prefilled_thinking")
        assert split.region_state == "closed"
        assert split.final == "Paris"
        assert split.reasoning is not None and split.reasoning.startswith("<|Assistant|><think>")

    def test_close_without_opening_tag(self):
        # the canonical prefill case: </think> appears with NO opening tag
        split = split_reasoning("some reasoning</think>Paris", "prefilled_thinking")
        assert split.region_state == "closed"
        assert split.reasoning == "some reasoning"
        assert split.final == "Paris"

    def test_last_close_wins(self):
        split = split_reasoning("a</think>b</think>c", "prefilled_thinking")
        assert split.region_state == "closed"
        assert split.reasoning == "a</think>b"
        assert split.final == "c"

    def test_open_region_at_eos(self):
        split = split_reasoning("unfinished reasoning", "prefilled_thinking")
        assert split.region_state == "open_at_eos"
        assert split.final == ""
        assert split.reasoning == "unfinished reasoning"

    def test_empty_output_is_open_at_eos(self):
        split = split_reasoning("", "prefilled_thinking")
        assert split.region_state == "open_at_eos"
        assert split.final == ""


class TestOptionalThinking:
    def test_answer_with_think_block(self):
        split = split_reasoning("<think>chain of thought</think>42", "optional_thinking")
        assert split.region_state == "closed"
        assert split.reasoning == "<think>chain of thought"
        assert split.final == "42"

    def test_answer_without_think_tags(self):
        split = split_reasoning("42", "optional_thinking")
        assert split.region_state == "no_tags"
        assert split.reasoning is None
        assert split.final == "42"

    def test_open_region_at_eos(self):
        split = split_reasoning("<think>unfinished", "optional_thinking")
        assert split.region_state == "open_at_eos"
        assert split.final == ""

    def test_closed_region_with_empty_final(self):
        # thinking present but NO answer channel content: the parser reports
        # missing_final downstream, never grades the reasoning text
        split = split_reasoning("<think>42</think>", "optional_thinking")
        assert split.region_state == "closed"
        assert split.final == ""
        assert split.reasoning == "<think>42"


class TestThinkingDisabled:
    def test_output_contains_no_think_tags(self):
        split = split_reasoning("plain answer", "thinking_disabled")
        assert split.region_state == "no_tags"
        assert split.reasoning is None
        assert split.final == "plain answer"
        assert "think" not in split.cleaned_text

    def test_phantom_close_text_after_is_answer_not_reasoning(self):
        # the prompt prefilled an empty CLOSED think block; a model-emitted
        # </think> splits at the last close: everything after it is the
        # answer channel and everything before it is discarded
        split = split_reasoning("junk</think>Paris", "thinking_disabled")
        assert split.region_state == "tags_after_prefill"
        assert split.reasoning is None
        assert split.final == "Paris"

    def test_phantom_close_end_to_end_musique_parser(self):
        parsed = parse_musique_answer("junk</think>Paris", "thinking_disabled")
        assert parsed.status == "ok"
        assert parsed.value == "Paris"  # never parsed as reasoning

    def test_everything_after_last_close_is_used(self):
        split = split_reasoning("Paris</think>junk", "thinking_disabled")
        assert split.region_state == "tags_after_prefill"
        assert split.final == "junk"


class TestNoThinkingChannel:
    def test_eos_tokens_stripped_and_text_preserved(self):
        split = split_reasoning("Paris<|im_end|>", "no_thinking_channel", EOS_IM_END)
        assert split.region_state == "no_channel"
        assert split.reasoning is None
        assert split.cleaned_text == "Paris"
        assert split.final == "Paris"
        assert split.raw_text == "Paris<|im_end|>"  # raw keeps the eos token

    def test_interior_eos_preserved(self):
        split = split_reasoning("a<|im_end|>b", "no_thinking_channel", EOS_IM_END)
        assert split.final == "a<|im_end|>b"

    def test_end_to_end_musique_parser_with_eos(self):
        parsed = parse_musique_answer("NOT FOUND<|im_end|>", "no_thinking_channel", EOS_IM_END)
        assert parsed.status == "ok"
        assert parsed.value == "NOT FOUND"
        assert parsed.diagnostics["is_not_found"] is True


class TestAnswerChannelContract:
    def test_answer_only_after_closed_reasoning_region(self):
        parsed = parse_musique_answer("<think>42</think>Paris", "optional_thinking")
        assert parsed.status == "ok"
        assert parsed.value == "Paris"
        assert parsed.final_region == "Paris"

    def test_answer_inside_reasoning_region_never_parsed(self):
        # "Paris" appears only inside the closed think region -> no answer
        parsed = parse_musique_answer("<think>Paris</think>", "optional_thinking")
        assert parsed.status == "missing_final"
        assert parsed.value is None
        assert parsed.final_region == ""

    @pytest.mark.parametrize("mode", ["prefilled_thinking", "optional_thinking"])
    def test_open_region_at_eos_is_missing_final(self, mode):
        parsed = parse_musique_answer("<think>unfinished", mode)
        assert parsed.status == "missing_final"
        assert parsed.value is None
        assert parsed.region_state == "open_at_eos"
        assert parsed.final_region == ""

    def test_open_region_with_plausible_answer_still_missing(self):
        # "NOT FOUND" inside an unclosed region is reasoning, not an abstention
        parsed = parse_musique_answer("NOT FOUND <think>...", "prefilled_thinking")
        assert parsed.status == "missing_final"
        assert parsed.value is None

    def test_eos_token_strings_stripped_end_to_end(self):
        parsed = parse_musique_answer("Paris<|im_end|>", "no_thinking_channel", EOS_IM_END)
        assert parsed.value == "Paris"
        assert parsed.cleaned_text == "Paris"
        assert parsed.raw_text == "Paris<|im_end|>"

    def test_final_region_convenience(self):
        assert final_region("<think>x</think>42", "optional_thinking") == "42"
        assert final_region("Paris<|im_end|>", "no_thinking_channel", EOS_IM_END) == "Paris"

    def test_split_dict_carries_version_and_mode(self):
        d = split_reasoning("<think>x</think>42", "optional_thinking").to_dict()
        assert d["version"] == REASONING_PARSER_VERSION
        assert d["reasoning_mode"] == "optional_thinking"
        assert isinstance(split_reasoning("x", "no_thinking_channel"), ReasoningSplit)

    def test_unknown_reasoning_mode_raises(self):
        with pytest.raises(ValueError):
            split_reasoning("x", "bogus_mode")
