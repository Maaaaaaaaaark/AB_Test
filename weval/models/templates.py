"""Native chat template rendering (offline, sha256-locked).

We do not call tokenizer.apply_chat_template for configurations whose template
is vendored: the pinned jinja string is rendered by this module with the same
environment transformers uses (trim_blocks / lstrip_blocks / keep_trailing_newline
/ StrictUndefined / tojson), so the rendered prompt is byte-identical to the
native one while being fully reproducible from configs/.

For configurations without a vendored template (gated Gemma / Llama), the
template hash is locked at the first cluster precheck (`weval precheck`); until
then formal stages refuse to run (require_locked_templates), and smoke/pilot
fall back to tokenizer.apply_chat_template with the unlocked status recorded in
the prediction record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2

from ..config import ModelCfg, sha256_hex

# Mirrors transformers' compile_jinja_template environment.
TEMPLATE_ENV = jinja2.Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)


def _tojson(value: Any) -> str:
    """Mirrors transformers' tojson filter (not needed for our text-only
    prompts, but the Qwen3 template references it in the tools branch)."""
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


TEMPLATE_ENV.filters["tojson"] = _tojson

TEMPLATES_VERSION = "template-render-v2"


class TemplateHashMismatch(RuntimeError):
    """The template file on disk does not match the pinned sha256."""


class TemplateNotLockedError(RuntimeError):
    """The configuration has no vendored template and no locked hash."""


@dataclass
class RenderedTemplate:
    text: str
    template_sha256: str
    template_locked: bool
    renderer_version: str = TEMPLATES_VERSION
    # True when the template source renders {{ bos_token }}; the backend then
    # asserts that the first prompt token is the tokenizer's BOS id.
    references_bos: bool = False
    # where the rendered BOS string came from: template_kwargs | tokenizer | none
    bos_source: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_sha256": sha256_hex(self.text),
            "template_sha256": self.template_sha256,
            "template_locked": self.template_locked,
            "renderer_version": self.renderer_version,
            "references_bos": self.references_bos,
            "bos_source": self.bos_source,
        }


def template_references_bos(template_str: str) -> bool:
    """Does the template render the bos_token variable?"""
    return "bos_token" in template_str


def _resolve_bos(model: ModelCfg, bos_fallback: Optional[str]) -> tuple[Dict[str, Any], str, str]:
    """(kwargs without bos_token, bos string, source).

    The pinned template_kwargs.bos_token wins; otherwise the tokenizer's
    bos_token (passed in by the backend) is used. Prompts are tokenized with
    add_special_tokens=False, so a template that renders {{ bos_token }} as ""
    produces a prompt with NO BOS token at all.
    """
    kwargs = dict(model.template_kwargs)
    pinned = kwargs.pop("bos_token", None)
    if pinned:
        return kwargs, str(pinned), "template_kwargs"
    if bos_fallback:
        return kwargs, str(bos_fallback), "tokenizer"
    return kwargs, "", "none"


def load_template_source(model: ModelCfg, root: Path) -> str:
    if not model.template_file:
        raise TemplateNotLockedError(
            f"model configuration {model.name!r} has no vendored template "
            "(gated model: lock it with `weval precheck` on the cluster)"
        )
    path = Path(root) / model.template_file
    if not path.exists():
        raise TemplateNotLockedError(
            f"template file missing for {model.name!r}: {path}"
        )
    with open(path, encoding="utf-8") as f:
        source = f.read()
    if model.template_sha256 and sha256_hex(source) != model.template_sha256:
        raise TemplateHashMismatch(
            f"template for {model.name!r} changed on disk: expected sha256 "
            f"{model.template_sha256}, got {sha256_hex(source)}. Restore the "
            "vendored template or update the config deliberately."
        )
    return source


def render_with_template(
    template_str: str,
    messages: List[Dict[str, Any]],
    template_kwargs: Optional[Dict[str, Any]] = None,
    bos_token: str = "",
) -> str:
    """Render a single-turn prompt with add_generation_prompt=True.

    tools and documents are always passed as None, mirroring transformers'
    render_with_template: the Qwen3 template's `{% if tools %}` branch
    depends on it under StrictUndefined.
    """
    template = TEMPLATE_ENV.from_string(template_str)
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": True,
        "tools": None,
        "documents": None,
        "bos_token": bos_token,
        "eos_token": "",
    }
    kwargs.update(template_kwargs or {})
    return template.render(**kwargs)


def render_prompt(
    model: ModelCfg,
    messages: List[Dict[str, Any]],
    root: Path,
    bos_fallback: Optional[str] = None,
) -> RenderedTemplate:
    """Render the prompt with the vendored template (hash-verified)."""
    source = load_template_source(model, root)
    kwargs, bos, bos_source = _resolve_bos(model, bos_fallback)
    text = render_with_template(source, messages, kwargs, bos_token=bos)
    refs = template_references_bos(source)
    return RenderedTemplate(
        text=text,
        template_sha256=model.template_sha256 or sha256_hex(source),
        template_locked=bool(model.template_sha256),
        references_bos=refs,
        bos_source=bos_source if refs else "none",
    )


def render_prompt_runtime(
    model: ModelCfg,
    messages: List[Dict[str, Any]],
    template_str: str,
    bos_fallback: Optional[str] = None,
) -> RenderedTemplate:
    """Unlocked fallback: render the tokenizer's own chat_template string.

    Used only for gated models before precheck locks their templates and only
    in stages that do not require locked templates. The template string itself
    is hashed and recorded so it can be locked later; `template_locked` is
    False. (Rendering the raw template string through this module, rather than
    calling apply_chat_template, keeps one code path for all configs.)
    """
    kwargs, bos, bos_source = _resolve_bos(model, bos_fallback)
    text = render_with_template(template_str, messages, kwargs, bos_token=bos)
    refs = template_references_bos(template_str)
    return RenderedTemplate(
        text=text,
        template_sha256=sha256_hex(template_str),
        template_locked=False,
        references_bos=refs,
        bos_source=bos_source if refs else "none",
    )
