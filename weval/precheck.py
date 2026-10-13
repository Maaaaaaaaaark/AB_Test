"""Cluster precheck: what is (not) satisfied before GPU jobs run.

Checks, in order of importance:
  1. HF token: boolean only (present in HF_TOKEN / huggingface token cache).
     Tokens are NEVER printed, hashed or saved.
  2. Gated access: fetch model config.json + tokenizer files for the gated
     repos (Gemma, Llama) and record whether the server returns the files
     (granted) or an error JSON (not granted) - this is a hard fact per repo.
  3. Template locks: sha256 the vendored jinja templates and propose the
     exact models.yaml patch that pins them (template_sha256).
  4. Hardware: torch importable, CUDA present, bf16 supported.

The precheck writes artifacts/precheck/precheck.json and prints a human
summary. Nothing here runs any model inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigBundle, sha256_hex

PRECHECK_VERSION = "precheck-v1"


@dataclass
class PrecheckReport:
    version: str = PRECHECK_VERSION
    checked_at_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    hf_token: Dict[str, Any] = field(default_factory=dict)
    gated_access: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "checked_at_utc": self.checked_at_utc,
            "hf_token": self.hf_token,
            "gated_access": self.gated_access,
            "templates": self.templates,
            "hardware": self.hardware,
            "problems": self.problems,
        }


def check_hf_token() -> Dict[str, Any]:
    """Boolean-only check. Never record or print any token value."""
    import os

    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return {"present": True, "source": "environment variable"}
    # huggingface_hub caches a token under ~/.cache/huggingface/token
    import huggingface_hub

    try:
        tok = huggingface_hub.get_token()
    except Exception:
        tok = None
    if tok:
        return {"present": True, "source": "huggingface token cache"}
    return {"present": False, "source": None}


def check_gated_access(cfg: ConfigBundle) -> Dict[str, Dict[str, Any]]:
    """Fetch config.json for every gated repo and record granted/denied.

    An HTTP 401/403 (or the HF error-JSON wrapper) is recorded as a factual
    "not granted" for that repo at this revision - it is not retried here.
    """
    import huggingface_hub

    out: Dict[str, Dict[str, Any]] = {}
    for name, m in cfg.models.items():
        if not m.gated:
            continue
        entry: Dict[str, Any] = {"repo": m.repo, "revision": m.revision}
        try:
            p = huggingface_hub.hf_hub_download(
                m.repo, "config.json", revision=m.revision,
            )
            entry["granted"] = True
            entry["config_json_sha256"] = sha256_hex(Path(p).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            entry["granted"] = False
            entry["error_kind"] = type(e).__name__
            entry["error_short"] = str(e)[:300]
        out[name] = entry
    return out


def _download_chat_template(model, destination: Path) -> str:
    """Fetch the tokenizer template at the pinned revision and vendor it."""
    import huggingface_hub

    p = huggingface_hub.hf_hub_download(
        model.repo, "tokenizer_config.json", revision=model.tokenizer_revision,
    )
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    template = raw.get("chat_template")
    if isinstance(template, dict):
        template = template.get("default")
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer_config.json has no unambiguous chat_template")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template, encoding="utf-8")
    return template


def check_templates(cfg: ConfigBundle) -> Dict[str, Dict[str, Any]]:
    """Vendor missing pinned templates, hash them, and propose YAML locks."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, m in cfg.models.items():
        if not m.template_file:
            out[name] = {"template_file": None, "note": "configuration must declare a vendored template path"}
            continue
        path = cfg.root / m.template_file
        if not path.exists():
            try:
                _download_chat_template(m, path)
            except Exception as e:  # gated access or malformed tokenizer config
                out[name] = {
                    "template_file": str(m.template_file),
                    "note": "file missing and download failed",
                    "error_kind": type(e).__name__,
                    "error_short": str(e)[:300],
                }
                continue
        actual = sha256_hex(path.read_text(encoding="utf-8"))
        entry = {
            "template_file": str(m.template_file),
            "sha256_actual": actual,
            "sha256_pinned": m.template_sha256,
        }
        entry["match"] = m.template_sha256 == actual if m.template_sha256 else None
        if not m.template_sha256:
            entry["proposed_yaml"] = (
                f"  {name}:\n    ...\n    template_file: {m.template_file}\n"
                f"    template_sha256: {actual}"
            )
        out[name] = entry
    return out


def check_hardware() -> Dict[str, Any]:
    """torch/CUDA/bf16 facts. Safe to run on the cluster before any job."""
    out: Dict[str, Any] = {"torch_importable": False}
    try:
        import torch  # type: ignore

        out["torch_importable"] = True
        out["torch_version"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device_count"] = torch.cuda.device_count()
            out["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        else:
            out["bf16_supported"] = False
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def run_precheck(cfg: ConfigBundle, out_dir: Path) -> PrecheckReport:
    rep = PrecheckReport()
    rep.hf_token = check_hf_token()
    if not rep.hf_token.get("present"):
        rep.problems.append(
            "no HF token found - the gated repos (google/gemma-3-4b-it, "
            "meta-llama/Llama-3.2-3B-Instruct) will NOT download; set HF_TOKEN "
            "(or `huggingface-cli login`) on the cluster"
        )
    rep.gated_access = check_gated_access(cfg)
    for name, e in rep.gated_access.items():
        if e.get("granted") is False:
            rep.problems.append(
                f"gated access NOT granted for {e['repo']}@{e['revision']}: "
                f"{e.get('error_kind')}"
            )
    rep.templates = check_templates(cfg)
    for name, t in rep.templates.items():
        if t.get("error_kind"):
            rep.problems.append(
                f"template unavailable for {name}: {t.get('error_kind')} - "
                f"{t.get('error_short')}"
            )
        if t.get("match") is False:
            rep.problems.append(
                f"template lock mismatch for {name}: pinned {t['sha256_pinned']} "
                f"but file hashes {t['sha256_actual']}"
            )
    rep.hardware = check_hardware()
    hw = rep.hardware
    if hw.get("torch_importable") and hw.get("cuda_available") and not hw.get("bf16_supported"):
        rep.problems.append(
            "CUDA present but bf16 NOT supported on the visible GPUs - the "
            "generation protocol requires bf16; run on bf16-capable hardware"
        )
    if not hw.get("torch_importable"):
        rep.problems.append(
            "torch is not importable in this environment - generation cannot run here"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "precheck.json", "w", encoding="utf-8") as f:
        json.dump(rep.to_dict(), f, indent=2, ensure_ascii=False)
    return rep


def print_precheck(rep: PrecheckReport) -> None:
    print(f"precheck v{rep.version} at {rep.checked_at_utc}")
    print()
    print("HF token:", "present" if rep.hf_token.get("present") else "NOT FOUND",
          f"({rep.hf_token.get('source')})")
    print()
    print("Gated access (config.json fetch):")
    for name, e in sorted(rep.gated_access.items()):
        status = "granted" if e.get("granted") else "NOT granted"
        extra = f" - {e.get('error_kind')}" if not e.get("granted") else ""
        print(f"  {name}: {status}{extra}")
    print()
    print("Template locks:")
    for name, t in sorted(rep.templates.items()):
        if not t.get("template_file"):
            continue
        if t.get("sha256_pinned"):
            print(f"  {name}: {'MATCH' if t.get('match') else 'MISMATCH'} (pinned {t['sha256_pinned'][:12]}...)")
        elif t.get("sha256_actual"):
            print(f"  {name}: UNLOCKED - proposed pin:")
            print(f"      template_sha256: {t['sha256_actual']}")
        elif t.get("note"):
            print(f"  {name}: {t['note']} ({t.get('error_kind', 'unknown error')})")
    print()
    hw = rep.hardware
    if not hw.get("torch_importable"):
        print("Hardware: torch NOT importable")
    else:
        print(f"Hardware: torch {hw.get('torch_version')}, cuda={hw.get('cuda_available')}, "
              f"bf16={hw.get('bf16_supported')}, devices={hw.get('devices')}")
    print()
    if rep.problems:
        print("PROBLEMS:")
        for p in rep.problems:
            print(f"  - {p}")
    else:
        print("No blocking problems found.")
