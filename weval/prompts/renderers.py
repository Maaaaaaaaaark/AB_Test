"""Prompt rendering: manifest items -> user messages.

Rendering rules are fixed per task in configs/tasks.yaml (instruction_version
pinned). Gold answers live only in item["gold"] and are NEVER read here, so a
rendered prompt cannot leak them by construction. The rendered user message is
what the (vendored, sha256-pinned) chat template wraps; for the two native
prompts we already vendored (Qwen3, DeepSeek-R1) the template is applied in
weval/models/templates.py.

MuSiQue relation questions keep the official "X >> relation" syntax; the
relation note is included for EVERY MuSiQue item (uniform explanation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..config import ConfigBundle, sha256_hex

RENDERER_VERSION = "renderer-v1"


@dataclass
class RenderedPrompt:
    task: str
    kind: str  # musique: positive/negative; others: "na"
    user_message: str
    renderer_version: str = RENDERER_VERSION

    def fingerprint(self) -> str:
        return sha256_hex(self.user_message)


def _format_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    """Numbered paragraph block shared by musique/hotpotqa renderings.

    Paragraphs are presented in visible (slot/context) order; the model sees
    the same numbering as recorded in the manifest.
    """
    blocks = []
    for i, p in enumerate(paragraphs, start=1):
        blocks.append(f"Paragraph {i}: {p['title']}: {p['text']}")
    return "\n\n".join(blocks)


def _render_musique(item: Dict[str, Any], cfg: ConfigBundle) -> str:
    task = cfg.task("musique_local")
    instruction = task.instruction.strip()
    note = getattr(task, "relation_note", None)
    parts = [instruction]
    if note:
        parts.append(note)
    paragraphs = sorted(item["visible"], key=lambda p: p["slot"])
    parts.append("Information:")
    parts.append("")
    parts.append(_format_paragraphs(paragraphs))
    parts.append("")
    parts.append(f"Question: {item['question']}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def _render_processbench(item: Dict[str, Any], cfg: ConfigBundle) -> str:
    from ..datasets.processbench import render_official_prompt

    return render_official_prompt(item["problem"], item["steps"])


def _render_template_task(item: Dict[str, Any], cfg: ConfigBundle) -> str:
    task = cfg.task(item["task"])
    # Plain .replace, NOT str.format: task instructions contain LaTeX braces
    # (e.g. \\boxed{}) that format() would treat as fields.
    if item["task"] == "gsm8k":
        return task.instruction.replace("{question}", item["question"])
    if item["task"] == "math":
        return task.instruction.replace("{problem}", item["problem"])
    if item["task"] == "hotpotqa":
        paragraphs = _format_paragraphs(item["paragraphs"])
        return (
            task.instruction.replace("{paragraphs}", paragraphs).replace("{question}", item["question"])
        )
    raise ValueError(f"no renderer for task {item['task']!r}")


def render_item(item: Dict[str, Any], cfg: ConfigBundle) -> RenderedPrompt:
    """Render one manifest item into a single user message."""
    task_name = item["task"]
    if task_name == "musique_local":
        return RenderedPrompt(
            task=task_name,
            kind=item.get("kind", "positive"),
            user_message=_render_musique(item, cfg),
        )
    if task_name == "processbench":
        return RenderedPrompt(task=task_name, kind="na", user_message=_render_processbench(item, cfg))
    return RenderedPrompt(task=task_name, kind="na", user_message=_render_template_task(item, cfg))
