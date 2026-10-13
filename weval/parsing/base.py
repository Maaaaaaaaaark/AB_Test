"""Shared parsed-answer structure.

Every answer parser receives the RAW generated text (special tokens included),
the model's reasoning_mode and the tokenizer's eos token strings, and returns
a ParsedAnswer. Nothing is thrown away: raw text, cleaned text, the final
answer-channel region and parser diagnostics are all kept in the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PARSED_ANSWER_VERSION = "parsed-answer-v1"


@dataclass
class ParsedAnswer:
    task: str
    parser: str
    raw_text: str
    cleaned_text: str
    final_region: str
    region_state: str
    value: Any  # parser-specific: str answer | int index | None (missing)
    status: str  # "ok" | "missing_final" | "invalid_format"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    version: str = PARSED_ANSWER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "parser": self.parser,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "final_region": self.final_region,
            "region_state": self.region_state,
            "value": self.value,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "version": self.version,
        }


def missing_final(task: str, parser: str, raw: str, cleaned: str, region_state: str, diagnostics: Optional[Dict[str, Any]] = None) -> ParsedAnswer:
    """Open reasoning region at EOS (or empty final): missing final answer.

    Scored as incorrect - never treated as a crash and never retried.
    """
    return ParsedAnswer(
        task=task,
        parser=parser,
        raw_text=raw,
        cleaned_text=cleaned,
        final_region="",
        region_state=region_state,
        value=None,
        status="missing_final",
        diagnostics=dict(diagnostics or {}),
    )
