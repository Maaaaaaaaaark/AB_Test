"""Generate docs/worked_examples.md from REAL manifest items.

For every task, a few real items are rendered exactly as the runner would
render them (same renderer code path). Gold answers and supporting labels are
shown in a separate "Gold (never shown to the model)" section.

Generation status per item is honest:
  - "pending" when no prediction record exists yet (nothing is fabricated),
  - otherwise the record path and the model's raw output text are included.

The document is plain English, aimed at a reviewer who wants to see exactly
what a model sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigBundle
from .prompts.renderers import render_item
from .run.manifests import manifest_items_for_task
from .run.store import safe_item_id

WORKED_EXAMPLES_VERSION = "worked-examples-v2"

# How many items per task go into the document (first N of the pilot pool).
N_PER_TASK = {"musique_local": 2, "processbench": 2, "gsm8k": 1, "math": 1, "hotpotqa": 1}

TASK_TITLES = {
    "musique_local": "MuSiQue local evidence task",
    "processbench": "ProcessBench",
    "gsm8k": "GSM8K (whole question, reference)",
    "math": "MATH (whole question, reference)",
    "hotpotqa": "HotpotQA distractor (whole question, reference)",
}


def _find_predictions(out_dir: Path, plan: str, item: Dict[str, Any]) -> List[Path]:
    """One real prediction per config for this item, using the lowest seed.

    Worked examples are a cross-model audit surface.  Returning only the first
    alphabetically sorted config silently hid every other model.
    """
    base = out_dir / "predictions" / plan
    if not base.exists():
        return []
    safe = safe_item_id(item["item_id"])
    found: List[Path] = []
    for cfg_dir in sorted(base.iterdir()):
        if not cfg_dir.is_dir():
            continue
        task_dir = cfg_dir / item["task"]
        if not task_dir.is_dir():
            continue
        for seed_dir in sorted(task_dir.iterdir()):
            p = seed_dir / f"{safe}.json"
            if p.exists():
                found.append(p)
                break
    return found


def _render_example_section(
    item: Dict[str, Any], cfg: ConfigBundle, out_dir: Path, plan: str
) -> str:
    rendered = render_item(item, cfg)
    lines: List[str] = []
    lines.append(f"#### {item['item_id']}")
    lines.append("")
    lines.append(f"- task: `{item['task']}`; kind: `{item.get('kind', 'na')}`; "
                 f"group: `{item.get('group_id', 'na')}`")
    if "syntax" in item:
        s = item["syntax"]
        lines.append(f"- question syntax: {'relation' if s.get('has_relation') else 'natural'}"
                     f"{'; is_terminal: ' + str(item['is_terminal']) if 'is_terminal' in item else ''}")
    if item.get("alias_source"):
        lines.append(f"- gold alias source: {item['alias_source']}")
    lines.append("")
    lines.append("User message (before the model's chat template; each prediction below shows the start of the templated prompt):")
    lines.append("")
    lines.append("```text")
    lines.append(rendered.user_message)
    lines.append("```")
    lines.append("")
    lines.append("Gold (never shown to the model):")
    lines.append("")
    gold = item.get("gold") or {}
    if item["task"] == "musique_local":
        lines.append(f"- expected answer(s): {gold.get('reference_answers')}")
        if item.get("kind") == "positive":
            lines.append(f"- supporting paragraph idx (in source record): {gold.get('support_paragraph_idx')}")
            if gold.get("used_prerequisites"):
                lines.append(f"- prerequisite answers substituted into the question: {gold['used_prerequisites']}")
        else:
            lines.append(f"- supporting paragraph replaced by paragraph idx: {gold.get('replaced_with_paragraph_idx')}")
            lines.append(f"- label: unanswerable (model must output exactly NOT FOUND)")
    elif item["task"] == "processbench":
        lbl = gold.get("label")
        if lbl == -1:
            lines.append("- gold: no error (the solution is correct; model must output -1)")
        else:
            lines.append(f"- gold: first error at step {lbl}")
        lines.append(f"- final answer marked correct by the authors: {gold.get('final_answer_correct')}")
    else:
        lines.append(f"- gold: {gold}")
    lines.append("")
    predictions = _find_predictions(out_dir, plan, item)
    if not predictions:
        lines.append(f"Generation: **pending** (no prediction record for plan `{plan}` yet).")
    else:
        import json
        for pred in predictions:
            rec = json.loads(pred.read_text(encoding="utf-8"))
            lines.append(f"##### Prediction: `{rec.get('config')}` (seed {rec.get('seed')})")
            lines.append("")
            lines.append(f"Generation record: `{pred.relative_to(out_dir)}`")
            lines.append("")
            prompt_text = rec.get("prompt_text") or ""
            ids = rec.get("prompt_token_ids") or []
            bos = ((rec.get("template") or {}).get("bos") or {})
            lines.append(f"- templated prompt starts with: `{prompt_text[:80]!r}`")
            lines.append(f"- first prompt token ids: {ids[:3]}; BOS check: {bos.get('bos_check', 'not recorded')}")
            lines.append(f"- stop strings trimmed before parsing: {rec.get('eos_token_strings')}")
            lines.append("")
            lines.append("```text")
            lines.append(rec.get("output_text", "<no output_text>"))
            lines.append("```")
            lines.append(f"(finish_reason `{rec.get('finish_reason')}`, "
                         f"Tin={rec.get('n_prompt_tokens')}, Tout={rec.get('n_generated_tokens')})")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def build_worked_examples(cfg: ConfigBundle, out_dir: Path, plan: str) -> str:
    """Render the full markdown document. plan selects the item pool
    (smoke/pilot manifests) and where to look for prediction records."""
    doc: List[str] = []
    doc.append("# Worked examples - exactly what the model sees")
    doc.append("")
    doc.append(f"- pool: plan `{plan}` manifests; renderer version pinned in each run record")
    doc.append("- prompts below are rendered by the same code path the runner uses")
    doc.append("- gold answers are shown for review only and never enter prompts")
    doc.append("- generation outputs are real prediction records when present; otherwise marked **pending**")
    doc.append("")
    for task, n in N_PER_TASK.items():
        try:
            items = manifest_items_for_task(cfg, out_dir, plan, task)
        except FileNotFoundError:
            doc.append(f"## {TASK_TITLES[task]}")
            doc.append("")
            doc.append(f"_no manifest items for plan `{plan}` yet - run `weval build` first_")
            doc.append("")
            continue
        # Show one negative too when available (musique).
        picked: List[Dict[str, Any]] = []
        for it in items:
            if it.get("kind") == "positive" or task != "musique_local":
                picked.append(it)
            if len(picked) >= n:
                break
        if task == "musique_local":
            neg = next((it for it in items if it.get("kind") == "negative"), None)
            if neg is not None and len(picked) >= n:
                picked.append(neg)
        doc.append(f"## {TASK_TITLES[task]}")
        doc.append("")
        for it in picked:
            doc.append(_render_example_section(it, cfg, out_dir, plan))
    return "\n".join(doc)


def write_worked_examples(cfg: ConfigBundle, out_dir: Path, docs_dir: Path, plan: str) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "worked_examples.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_worked_examples(cfg, out_dir, plan))
    return path
