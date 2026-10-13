"""Transformers backend: the real cluster implementation.

Protocol (identical for every model configuration, recorded merged with model
defaults): BF16, no quantization, eval mode, transformers backend, batch 1,
do_sample=True, temperature=0.6, top_p=0.95, top_k=0, repetition_penalty=1.0,
num_beams=1, one generation per item per seed.

Details that matter for the evaluation contract:
  * the prompt is the vendored (sha256-locked) chat template applied to the
    rendered user message; for gated models without a locked template the
    tokenizer's own template string is rendered through the same code path
    and recorded unlocked,
  * raw generated token ids and raw decoded text (special tokens included)
    are stored; cleanup happens later in the parsing layer,
  * a deadline logits processor aborts decoding at the wall-clock deadline;
    that is an infrastructure failure (never scored as partial output),
  * reaching max_new_tokens is a NORMAL evaluated outcome (finish_reason
    "length"), not an error,
  * per-item torch seeds come from make_item_seed(run_seed, item_id), so a
    rerun of one item reproduces it exactly regardless of run order.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import GENERATION_PARAMS, ModelCfg
from .backend import DeadlineExceeded, GenerationBackend, GenerationResult, InfrastructureError, make_item_seed
from .templates import RenderedTemplate, render_prompt, render_prompt_runtime

TRANSFORMERS_BACKEND_VERSION = "transformers-backend-v1"

# GenerationConfig keys that are per-item protocol overrides (dtype/quantization/
# backend/batch_size are backend-level and recorded separately).
_PROTOCOL_GEN_KEYS = (
    "do_sample", "temperature", "top_p", "top_k",
    "repetition_penalty", "num_beams", "use_cache",
)


class DeadlineLogitsProcessor:
    """Raises DeadlineExceeded when decoding runs past the item deadline."""

    def __init__(self, deadline_monotonic: float, deadline_seconds: float):
        self.deadline = deadline_monotonic
        self.deadline_seconds = deadline_seconds

    def __call__(self, input_ids, scores):
        elapsed = time.monotonic() - (self.deadline - self.deadline_seconds)
        if time.monotonic() > self.deadline:
            raise DeadlineExceeded(self.deadline_seconds, elapsed)
        return scores


@dataclass
class _ModelBundle:
    model: Any
    tokenizer: Any
    context_limit: Optional[int]


class TransformersBackend(GenerationBackend):
    name = "transformers"

    def cache_identity(self) -> Dict[str, str]:
        return {"name": self.name, "version": TRANSFORMERS_BACKEND_VERSION}

    def __init__(
        self,
        snapshot_dir: str | Path,
        model_cfg: ModelCfg,
        root: Path,
        device: str = "auto",
        attn_implementation: Optional[str] = None,
    ):
        import torch

        self.snapshot_dir = Path(snapshot_dir)
        self.model_cfg = model_cfg
        self.root = Path(root)
        self.device = device
        self._attn = attn_implementation
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError(
                "the evaluation protocol requires bfloat16; this machine's GPU "
                "does not support it (run on the cluster)"
            )
        self._bundle: Optional[_ModelBundle] = None

    # -- loading ------------------------------------------------------------

    def _load(self) -> _ModelBundle:
        if self._bundle is None:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            t0 = time.monotonic()
            pretrained_config = AutoConfig.from_pretrained(str(self.snapshot_dir))
            if getattr(pretrained_config, "model_type", None) == "gemma3":
                # Gemma 3 4B is a conditional-generation multimodal
                # checkpoint even in this text-only experiment.  Loading it
                # through AutoModelForCausalLM is unsupported.
                from transformers import Gemma3ForConditionalGeneration
                model_cls = Gemma3ForConditionalGeneration
            else:
                model_cls = AutoModelForCausalLM
            model = model_cls.from_pretrained(
                str(self.snapshot_dir),
                config=pretrained_config,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation=self._attn,
                low_cpu_mem_usage=True,
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot_dir))
            limit = self._resolve_context_limit(model.config)
            self._bundle = _ModelBundle(model=model, tokenizer=tokenizer, context_limit=limit)
        return self._bundle

    def _resolve_context_limit(self, model_config: Any) -> Optional[int]:
        source = self.model_cfg.context_limit_source or ""
        node = model_config
        try:
            for part in source.split("."):
                if part == "model_config":
                    continue
                node = getattr(node, part)
            return int(node)
        except Exception:
            return None

    # -- helper surface used by the runner ----------------------------------

    def tokenize(self, prompt_text: str) -> List[int]:
        bundle = self._load()
        rendered = self._render(prompt_text, bundle.tokenizer)
        return bundle.tokenizer(rendered.text, add_special_tokens=False).input_ids

    def eos_strings(self) -> Optional[List[str]]:
        """End-of-sequence token strings used to trim generated text."""
        bundle = self._load()
        eos = getattr(bundle.tokenizer, "eos_token", None)
        if not eos:
            return None
        return [str(eos)]

    @property
    def context_limit(self) -> Optional[int]:
        bundle = self._load()
        return bundle.context_limit

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        item_id: str,
        seed: int,
        prompt_text: str,
        max_new_tokens: int,
        deadline_seconds: float,
    ) -> GenerationResult:
        import torch

        bundle = self._load()
        model, tokenizer = bundle.model, bundle.tokenizer

        rendered = self._render(prompt_text, tokenizer)
        input_ids = tokenizer(rendered.text, add_special_tokens=False).input_ids
        prompt_tokens = list(input_ids)

        item_seed = make_item_seed(seed, item_id)
        torch.manual_seed(item_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(item_seed)

        protocol = {k: GENERATION_PARAMS[k] for k in _PROTOCOL_GEN_KEYS}
        gen_cfg = getattr(model, "generation_config", None)
        model_defaults = dict(gen_cfg.to_diff_dict()) if gen_cfg is not None else {}
        effective = dict(model_defaults)
        effective.update(protocol)
        merged_record = {
            "protocol_overrides": protocol,
            "model_defaults_diff": model_defaults,
            "effective": effective,
            "dtype": GENERATION_PARAMS["dtype"],
            "quantization": GENERATION_PARAMS["quantization"],
            "batch_size": GENERATION_PARAMS["batch_size"],
            "backend_version": TRANSFORMERS_BACKEND_VERSION,
        }

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        deadline = time.monotonic() + deadline_seconds
        processor = DeadlineLogitsProcessor(deadline, deadline_seconds)

        t0 = time.monotonic()
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=torch.tensor([input_ids], dtype=torch.long, device=model.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=protocol["do_sample"],
                    temperature=protocol["temperature"],
                    top_p=protocol["top_p"],
                    top_k=protocol["top_k"],
                    repetition_penalty=protocol["repetition_penalty"],
                    num_beams=protocol["num_beams"],
                    use_cache=protocol["use_cache"],
                    pad_token_id=pad_token_id,
                    logits_processor=[processor],
                )
        except torch.cuda.OutOfMemoryError as e:
            raise InfrastructureError("oom", f"CUDA OOM for {item_id}: {e}") from e
        except DeadlineExceeded:
            raise
        except RuntimeError as e:
            raise InfrastructureError("crash", f"runtime error for {item_id}: {e}") from e
        wall = time.monotonic() - t0

        generated_ids = out[0, len(input_ids) :].tolist()
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        finish = "length" if len(generated_ids) >= max_new_tokens else "eos"

        return GenerationResult(
            item_id=item_id,
            seed=seed,
            item_seed=item_seed,
            prompt_text=rendered.text,
            prompt_token_ids=prompt_tokens,
            generated_token_ids=generated_ids,
            output_text=raw_text,
            finish_reason=finish,
            n_prompt_tokens=len(prompt_tokens),
            n_generated_tokens=len(generated_ids),
            wall_seconds=wall,
            backend=self.name,
            tokenizer_used=self.model_cfg.name,
            merged_generation_params=merged_record,
            extra={
                "template": rendered.to_dict(),
                "model_revision": self.model_cfg.revision,
            },
        )

    def _render(self, prompt_text: str, tokenizer: Any) -> RenderedTemplate:
        messages = [{"role": "user", "content": prompt_text}]
        if self.model_cfg.template_sha256 or self.model_cfg.template_file:
            return render_prompt(self.model_cfg, messages, self.root)
        template_str = getattr(tokenizer, "chat_template", None)
        if not template_str:
            raise InfrastructureError(
                "crash",
                f"tokenizer for {self.model_cfg.name!r} has no chat_template and "
                "no vendored template is configured",
            )
        return render_prompt_runtime(self.model_cfg, messages, str(template_str))

    def close(self) -> None:
        self._bundle = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
