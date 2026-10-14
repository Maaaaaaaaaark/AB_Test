"""Reasoning-region splitting per reasoning_mode.

The chat templates themselves define the channels:

  prefilled_thinking  - the prompt ends inside an OPEN <think> region
                        (DeepSeek-R1: "...<think>\n"). The generated text may
                        contain the closing </think> WITHOUT an opening tag;
                        that is the prefill case and is handled here.
  optional_thinking   - Qwen3 (thinking on): the model may emit
                        <think>...</think> itself or answer directly.
  thinking_disabled   - Qwen3 (thinking off): the prompt prefills an empty
                        CLOSED <think>\n\n</think>\n\n block, so generated
                        text is the answer channel; if the model emits tags
                        anyway, everything after the last </think> is used.
  no_thinking_channel - Gemma / Llama: the whole output is the answer channel.
                        No hidden-reasoning token counts are invented.

Contract:
  * an open reasoning region at EOS (opening without closing) => MISSING final
    answer: region_state "open_at_eos", final "" - the item scores incorrect,
    it is not a crash;
  * answers are never graded from the reasoning region;
  * trailing end-of-sequence token strings (passed in by the caller from the
    tokenizer) are trimmed from the whole text before splitting; when unknown
    (dry-run), only whitespace is trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

REASONING_PARSER_VERSION = "reasoning-split-v1"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class ReasoningSplit:
    raw_text: str
    cleaned_text: str
    reasoning: Optional[str]  # None = no reasoning channel content
    final: str               # answer-channel text ("" when missing)
    region_state: str        # closed | open_at_eos | no_tags | no_channel | tags_after_prefill
    reasoning_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "reasoning": self.reasoning,
            "final": self.final,
            "region_state": self.region_state,
            "reasoning_mode": self.reasoning_mode,
            "version": REASONING_PARSER_VERSION,
        }


def merge_stop_strings(*groups: Optional[List[str]]) -> Optional[List[str]]:
    """Union of stop-string lists (order-preserving, empties dropped).

    Used to combine the strings recorded in a prediction record with the
    model configuration's declared stop_strings, so a record produced before
    a stop string was known (e.g. Gemma's <end_of_turn>) is still trimmed
    correctly when it is re-scored. Returns None when nothing is known.
    """
    out: List[str] = []
    for g in groups:
        for s in g or []:
            if s and s not in out:
                out.append(str(s))
    return out or None


def strip_eos(text: str, eos_strings: Optional[List[str]] = None) -> str:
    """Remove trailing end-of-sequence token strings (and whitespace).

    Applied repeatedly so a generated "<answer><|im_end|>" becomes "<answer>".
    Never touches the interior of the text.
    """
    t = text.rstrip()
    if not eos_strings:
        return t
    eos = sorted((s for s in eos_strings if s), key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for s in eos:
            if t.endswith(s):
                t = t[: -len(s)].rstrip()
                changed = True
    return t


def _split_at_last_close(text: str) -> tuple[str, str]:
    idx = text.rfind(THINK_CLOSE)
    if idx == -1:
        return text, ""
    reasoning = text[:idx]
    final = text[idx + len(THINK_CLOSE) :]
    return reasoning, final


def split_reasoning(
    raw_text: str,
    reasoning_mode: str,
    eos_strings: Optional[List[str]] = None,
) -> ReasoningSplit:
    cleaned = strip_eos(raw_text, eos_strings)

    if reasoning_mode == "prefilled_thinking":
        if THINK_CLOSE in cleaned:
            reasoning, final = _split_at_last_close(cleaned)
            return ReasoningSplit(raw_text, cleaned, reasoning, final.strip(), "closed", reasoning_mode)
        return ReasoningSplit(raw_text, cleaned, cleaned, "", "open_at_eos", reasoning_mode)

    if reasoning_mode == "optional_thinking":
        if THINK_CLOSE in cleaned:
            reasoning, final = _split_at_last_close(cleaned)
            return ReasoningSplit(raw_text, cleaned, reasoning, final.strip(), "closed", reasoning_mode)
        if THINK_OPEN in cleaned:
            # opening tag, no closing tag -> open region at EOS
            return ReasoningSplit(raw_text, cleaned, cleaned, "", "open_at_eos", reasoning_mode)
        return ReasoningSplit(raw_text, cleaned, None, cleaned.strip(), "no_tags", reasoning_mode)

    if reasoning_mode == "thinking_disabled":
        if THINK_CLOSE in cleaned:
            _, final = _split_at_last_close(cleaned)
            return ReasoningSplit(raw_text, cleaned, None, final.strip(), "tags_after_prefill", reasoning_mode)
        return ReasoningSplit(raw_text, cleaned, None, cleaned.strip(), "no_tags", reasoning_mode)

    if reasoning_mode == "no_thinking_channel":
        return ReasoningSplit(raw_text, cleaned, None, cleaned.strip(), "no_channel", reasoning_mode)

    raise ValueError(f"unknown reasoning_mode {reasoning_mode!r}")


def final_region(raw_text: str, reasoning_mode: str, eos_strings: Optional[List[str]] = None) -> str:
    """Convenience: answer-channel text only."""
    return split_reasoning(raw_text, reasoning_mode, eos_strings).final
