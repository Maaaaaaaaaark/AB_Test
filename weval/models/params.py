"""N_text: unique text-model parameter count from a checkpoint snapshot.

Reads only safetensors HEADERS (keys, shapes, dtypes) plus the small
config.json - the full weights are never loaded, so this works anywhere.

Rules (auditable; the full breakdown is saved):
  * every tensor counts once,
  * tensors matched by the configuration's param_exclusions are excluded
    (Gemma: vision_tower.* and multi_modal_projector.* - the text model is
    what runs for our text-only tasks),
  * when config.json says tie_word_embeddings=true AND the embedding and
    LM-head tensors have the same shape, they are the same underlying
    parameter and are counted ONCE (the convention transformers'
    get_num_parameters uses); a shape mismatch means the checkpoint is
    actually untied despite the flag, so both are counted,
  * a tensor name repeated across shard files (malformed checkpoint) is
    counted once, defensively,
  * sharded checkpoints (index.json) are summed across shards.
"""

from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PARAMS_VERSION = "params-v2"


@dataclass
class ParamAudit:
    n_text: int
    n_total_in_files: int
    n_excluded: int  # excluded parameter elements (numel), not tensor keys
    n_excluded_tensors: int
    tie_word_embeddings: Optional[bool]
    tied_dedup_applied: bool
    per_file: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    excluded_keys: List[str] = field(default_factory=list)
    tied_keys: List[str] = field(default_factory=list)
    version: str = PARAMS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "n_text": self.n_text,
            "n_total_in_files": self.n_total_in_files,
            "n_excluded": self.n_excluded,
            "n_excluded_tensors": self.n_excluded_tensors,
            "tie_word_embeddings": self.tie_word_embeddings,
            "tied_dedup_applied": self.tied_dedup_applied,
            "per_file": self.per_file,
            "excluded_keys": self.excluded_keys,
            "tied_keys": self.tied_keys,
        }


def _safetensors_files(snapshot_dir: Path) -> List[Path]:
    single = snapshot_dir / "model.safetensors"
    if single.exists():
        return [single]
    index = snapshot_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index, encoding="utf-8") as f:
            idx = json.load(f)
        files = sorted({Path(p) for p in idx["weight_map"].values()})
        paths = [snapshot_dir / p for p in files]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"sharded checkpoint files missing: {missing}")
        return paths
    raise FileNotFoundError(f"no model.safetensors (or index) under {snapshot_dir}")


def _tie_info(config: Dict[str, Any]) -> Optional[bool]:
    if "tie_word_embeddings" in config:
        return bool(config["tie_word_embeddings"])
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "tie_word_embeddings" in text_config:
        return bool(text_config["tie_word_embeddings"])
    return None


def count_text_parameters(
    snapshot_dir: str | Path,
    model_config: Optional[Dict[str, Any]] = None,
    param_exclusions: Optional[List[str]] = None,
    detail: bool = False,
) -> ParamAudit:
    """Count unique text-model parameters from safetensors headers."""
    snapshot_dir = Path(snapshot_dir)
    exclusions = param_exclusions or []
    if model_config is None:
        cfg_path = snapshot_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                model_config = json.load(f)
        else:
            model_config = {}
    tie = _tie_info(model_config)

    files = _safetensors_files(snapshot_dir)
    audit = ParamAudit(
        n_text=0,
        n_total_in_files=0,
        n_excluded=0,
        n_excluded_tensors=0,
        tie_word_embeddings=tie,
        tied_dedup_applied=False,
    )

    # First pass: headers only. Collect shapes so the tied-pair rule can
    # compare the embedding and LM-head tensors across shard boundaries.
    from safetensors import safe_open

    per_file: Dict[str, Dict[str, Any]] = {}
    key_shapes: Dict[str, List[int]] = {}
    for fpath in files:
        with safe_open(str(fpath), framework="pt") as f:
            keys = list(f.keys())
            shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
        per_file[str(fpath.name)] = {"keys": keys, "shapes": shapes}
        for k, s in shapes.items():
            key_shapes.setdefault(k, s)

    tied_skip: set = set()
    tied_keys: List[str] = []
    if tie:
        emb_suffix = "model.embed_tokens.weight"
        for emb_key in sorted(k for k in key_shapes if k.endswith(emb_suffix)):
            prefix = emb_key[: -len(emb_suffix)]
            lm_key = prefix + "lm_head.weight"
            emb = key_shapes[emb_key]
            lm = key_shapes.get(lm_key)
            if lm is not None and emb == lm:
                # Same underlying parameter (tied): the pair counts once.
                # Prefix-aware matching supports both text-only checkpoints
                # and multimodal layouts such as language_model.*.
                tied_keys.extend([emb_key, lm_key])
                tied_skip.add(lm_key)
                audit.tied_dedup_applied = True
    audit.tied_keys = tied_keys

    seen_counted: set = set()
    for fname, fdata in per_file.items():
        file_total = 0
        file_counted = 0
        file_excluded = 0
        file_excluded_tensors = 0
        for key in fdata["keys"]:
            numel = math.prod(fdata["shapes"][key])
            # Raw checkpoint total includes text, excluded modalities, tied
            # copies, and even defensive duplicate-key copies across shards.
            audit.n_total_in_files += numel
            file_total += numel
            if any(fnmatch.fnmatch(key, pat) for pat in exclusions):
                audit.n_excluded += numel
                audit.n_excluded_tensors += 1
                file_excluded += numel
                file_excluded_tensors += 1
                if detail:
                    audit.excluded_keys.append(key)
                continue
            if key in tied_skip:
                continue
            if key in seen_counted:
                # same tensor name in more than one shard file (malformed
                # checkpoint): count the parameter once, defensively
                audit.tied_keys.append(key)
                audit.tied_dedup_applied = True
                continue
            seen_counted.add(key)
            audit.n_text += numel
            file_counted += numel
        audit.per_file[fname] = {
            "numel_total": file_total,
            "numel_counted": file_counted,
            "numel_excluded": file_excluded,
            "n_excluded_tensors": file_excluded_tensors,
        }

    return audit
