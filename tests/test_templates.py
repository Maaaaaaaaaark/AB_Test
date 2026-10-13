"""Golden-template tests against the REAL pinned templates in configs/models.yaml.

The pinned chat templates and their sha256 live in configs/models.yaml; these
tests load that file directly (never a copy), so any change to the vendored
template or the pinned hash is caught here.

The "golden" rendering is computed by running the pinned jinja template with
the documented template_kwargs in an environment that mirrors transformers'
compile_jinja_template (trim_blocks / lstrip_blocks / keep_trailing_newline /
StrictUndefined / tojson). transformers' render_with_template ALWAYS passes
tools=None, so the golden render passes tools=None too (see the module-bug
test below for what happens without it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import jinja2
import pytest
import yaml

from tests.conftest import PROJECT_ROOT
from weval.config import ModelCfg, sha256_hex
from weval.models.templates import (
    TEMPLATE_ENV,
    TEMPLATES_VERSION,
    TemplateHashMismatch,
    TemplateNotLockedError,
    load_template_source,
    render_prompt,
    render_prompt_runtime,
    render_with_template,
)

MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"

USER_MSGS: List[Dict[str, Any]] = [{"role": "user", "content": "Hello?"}]

# Goldens computed by running the pinned templates with the documented
# template_kwargs (see module docstring for how they were produced).
QWEN3_OFF_GOLDEN = "<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_ON_GOLDEN = "<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n"
DEEPSEEK_GOLDEN = "<｜begin▁of▁sentence｜><｜User｜>Hello?<｜Assistant｜><think>\n"


def _models() -> Dict[str, Any]:
    with open(MODELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def _model_cfg(name: str) -> ModelCfg:
    return ModelCfg(name=name, **_models()[name])


def _template_source(name: str) -> str:
    return (PROJECT_ROOT / _models()[name]["template_file"]).read_text(encoding="utf-8")


def _transformers_env() -> jinja2.Environment:
    """Independent mirror of transformers' compile_jinja_template environment."""
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    return env


def _golden_render(template_str: str, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    """Direct render the way transformers does: tools=None is always passed."""
    render_kwargs: Dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": True,
        "bos_token": "",
        "eos_token": "",
        "tools": None,
    }
    render_kwargs.update(kwargs)
    return _transformers_env().from_string(template_str).render(**render_kwargs)


class TestVendoredTemplateHashes:
    def test_sha256_of_vendored_files_match_pinned_hashes(self):
        for name in ("qwen3-4b-thinking-off", "qwen3-4b-thinking-on", "deepseek-r1-distill-qwen-1.5b"):
            raw = _models()[name]
            src = (PROJECT_ROOT / raw["template_file"]).read_text(encoding="utf-8")
            assert sha256_hex(src) == raw["template_sha256"], name

    def test_qwen3_on_off_pin_the_same_template_file(self):
        assert _models()["qwen3-4b-thinking-on"]["template_file"] == _models()["qwen3-4b-thinking-off"]["template_file"]
        assert _models()["qwen3-4b-thinking-on"]["template_sha256"] == _models()["qwen3-4b-thinking-off"]["template_sha256"]

    def test_gated_configs_declare_precheck_vendor_destinations(self):
        for name in ("gemma-3-4b-it", "llama-3.2-3b-instruct"):
            raw = _models()[name]
            assert raw["template_file"].startswith("configs/templates/")
            assert raw["template_sha256"] is None


class TestQwen3Golden:
    def test_thinking_off_prefills_closed_think_block(self):
        src = _template_source("qwen3-4b-thinking-off")
        rendered = render_with_template(src, USER_MSGS, {"enable_thinking": False, "tools": None})
        assert rendered == QWEN3_OFF_GOLDEN
        # the pinned template_kwargs switch: enable_thinking=false -> prefilled
        # empty closed think block; answer channel is the whole output
        assert "<think>\n\n</think>\n\n" in rendered
        assert "<|im_start|>user\nHello?<|im_end|>\n" in rendered
        assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_thinking_on_emits_no_think(self):
        src = _template_source("qwen3-4b-thinking-on")
        rendered = render_with_template(src, USER_MSGS, {"enable_thinking": True, "tools": None})
        assert rendered == QWEN3_ON_GOLDEN
        assert "<think>" not in rendered
        assert rendered.endswith("<|im_start|>assistant\n")

    def test_matches_independent_transformers_style_jinja_render(self):
        src = _template_source("qwen3-4b-thinking-off")
        for enable_thinking, kwargs in ((False, {"enable_thinking": False}), (True, {"enable_thinking": True})):
            golden = _golden_render(src, USER_MSGS, **kwargs)
            ours = render_with_template(src, USER_MSGS, {**kwargs, "tools": None})
            assert ours == golden

    def test_render_prompt_qwen3_without_tools_kwarg(self):
        """render_with_template passes tools=None (transformers parity), so
        the pinned Qwen3 template's `{% if tools %}` branch renders with the
        shipped config (template_kwargs carries enable_thinking only)."""
        model = _model_cfg("qwen3-4b-thinking-off")
        rendered = render_prompt(model, USER_MSGS, PROJECT_ROOT)
        golden = _golden_render(_template_source("qwen3-4b-thinking-off"), USER_MSGS, enable_thinking=False)
        assert rendered.text == golden


class TestDeepseekGolden:
    def test_prefill_and_bos_from_pinned_template_kwargs(self):
        model = _model_cfg("deepseek-r1-distill-qwen-1.5b")
        rendered = render_prompt(model, USER_MSGS, PROJECT_ROOT)
        assert rendered.text == DEEPSEEK_GOLDEN
        # {{bos_token}} rendered from the pinned template_kwargs value
        assert rendered.text.startswith("<｜begin▁of▁sentence｜>")
        # prefill: the template ends inside an open <think> region
        assert rendered.text.endswith("<｜Assistant｜><think>\n")
        assert "<｜User｜>Hello?" in rendered.text
        assert rendered.template_sha256 == _models()["deepseek-r1-distill-qwen-1.5b"]["template_sha256"]
        assert rendered.template_locked is True
        assert rendered.renderer_version == TEMPLATES_VERSION

    def test_matches_independent_transformers_style_jinja_render(self):
        src = _template_source("deepseek-r1-distill-qwen-1.5b")
        golden = _golden_render(src, USER_MSGS, bos_token="<｜begin▁of▁sentence｜>")
        rendered = render_prompt(_model_cfg("deepseek-r1-distill-qwen-1.5b"), USER_MSGS, PROJECT_ROOT)
        assert rendered.text == golden

    def test_rendered_template_to_dict(self):
        rendered = render_prompt(_model_cfg("deepseek-r1-distill-qwen-1.5b"), USER_MSGS, PROJECT_ROOT)
        d = rendered.to_dict()
        assert d["text_sha256"] == sha256_hex(rendered.text)
        assert d["template_sha256"] == rendered.template_sha256
        assert d["template_locked"] is True
        assert d["renderer_version"] == TEMPLATES_VERSION


class TestHashVerification:
    def test_load_template_source_returns_verified_source(self):
        model = _model_cfg("qwen3-4b-thinking-off")
        src = load_template_source(model, PROJECT_ROOT)
        assert src == _template_source("qwen3-4b-thinking-off")

    def test_tampered_template_raises_hash_mismatch(self, tmp_path):
        (tmp_path / "tampered.jinja").write_text("{{ messages }}", encoding="utf-8")
        model = ModelCfg(
            name="tampered",
            repo="r", revision="r", tokenizer_revision="r", license="l",
            gated=False, gated_note=None, reasoning_mode="no_thinking_channel",
            template_file="tampered.jinja", template_sha256="0" * 64,
        )
        with pytest.raises(TemplateHashMismatch):
            load_template_source(model, tmp_path)
        with pytest.raises(TemplateHashMismatch):
            render_prompt(model, USER_MSGS, tmp_path)

    def test_correct_hash_passes_verification(self, tmp_path):
        src = "{{ bos_token }}{{ messages[0].content }}"
        (tmp_path / "ok.jinja").write_text(src, encoding="utf-8")
        model = ModelCfg(
            name="ok",
            repo="r", revision="r", tokenizer_revision="r", license="l",
            gated=False, gated_note=None, reasoning_mode="no_thinking_channel",
            template_file="ok.jinja", template_sha256=sha256_hex(src),
        )
        assert load_template_source(model, tmp_path) == src

    def test_no_vendored_template_raises_not_locked(self):
        model = _model_cfg("gemma-3-4b-it")  # gated: template_file is null
        with pytest.raises(TemplateNotLockedError):
            load_template_source(model, PROJECT_ROOT)

    def test_missing_template_file_raises_not_locked(self, tmp_path):
        model = ModelCfg(
            name="missing",
            repo="r", revision="r", tokenizer_revision="r", license="l",
            gated=False, gated_note=None, reasoning_mode="no_thinking_channel",
            template_file="does-not-exist.jinja", template_sha256=None,
        )
        with pytest.raises(TemplateNotLockedError):
            load_template_source(model, tmp_path)


class TestRenderPromptRuntime:
    def test_runtime_hashes_the_template_string_it_used(self):
        model = _model_cfg("deepseek-r1-distill-qwen-1.5b")
        src = _template_source("deepseek-r1-distill-qwen-1.5b")
        rt = render_prompt_runtime(model, USER_MSGS, src)
        assert rt.template_sha256 == sha256_hex(src)
        assert rt.template_locked is False
        assert rt.renderer_version == TEMPLATES_VERSION
        # same template + same kwargs as the locked path -> same text
        assert rt.text == render_prompt(model, USER_MSGS, PROJECT_ROOT).text

    def test_runtime_renders_custom_template_string(self):
        model = ModelCfg(
            name="unlocked",
            repo="r", revision="r", tokenizer_revision="r", license="l",
            gated=False, gated_note=None, reasoning_mode="no_thinking_channel",
            template_kwargs={"bos_token": "BOS"},
        )
        template_str = "{{ bos_token }}{{ messages[0].content }}"
        rt = render_prompt_runtime(model, USER_MSGS, template_str)
        assert rt.text == "BOSHello?"
        assert rt.template_sha256 == sha256_hex(template_str)
        assert rt.template_locked is False


class TestTemplateEnvironment:
    def test_env_mirrors_transformers_compile_jinja_template(self):
        assert TEMPLATE_ENV.trim_blocks is True
        assert TEMPLATE_ENV.lstrip_blocks is True
        assert TEMPLATE_ENV.keep_trailing_newline is False
        assert TEMPLATE_ENV.undefined is jinja2.StrictUndefined

    def test_tojson_filter_matches_transformers(self):
        value = {"b": 1, "a": [1, 2], "s": "x"}
        expected = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        assert TEMPLATE_ENV.filters["tojson"](value) == expected
