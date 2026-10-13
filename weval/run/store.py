"""Prediction store: per-item atomic JSON records + progress state.

Cache key components (all present in the record, all re-validated on resume):
model revision, chat template, rendered prompt, item content, generation
settings, seed, parser and scorer versions. Any mismatch invalidates the
cached record and the item is regenerated (the stale record is overwritten
atomically; the invalidation is logged, never silent).

Layout under artifacts/predictions/{plan}/:
    {config}/{task}/{seed}/{item_id}.json   prediction record
    {config}/{task}/{seed}/status.json      per-seed progress counters
    {config}/{task}/progress.log            append-only human log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import canonical_json, fingerprint

PREDICTION_RECORD_VERSION = 1


def item_fingerprint(item: Dict[str, Any]) -> str:
    """Canonical-JSON sha256 of the manifest item (gold included - the item
    snapshot identifies the exact content the prompt was built from)."""
    return fingerprint(item)


def record_fingerprints(item: Dict[str, Any], rendered_text: str, model_revision: str, template_sha256: Optional[str], gen_params_version: str, parser_version: str, scorer_version: str) -> Dict[str, str]:
    return {
        "item": item_fingerprint(item),
        "prompt": fingerprint(rendered_text),
        "model_revision": model_revision,
        "template_sha256": template_sha256 or "unlocked",
        "gen_params_version": gen_params_version,
        "parser_version": parser_version,
        "scorer_version": scorer_version,
    }


def safe_item_id(item_id: str) -> str:
    # item ids are dataset-controlled; sanitize path separators defensively
    return item_id.replace("/", "_").replace("\\", "_").replace(":", "_")


class PredictionStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def record_path(self, config: str, task: str, seed: int, item_id: str) -> Path:
        return self.base_dir / config / task / str(seed) / f"{safe_item_id(item_id)}.json"

    def status_path(self, config: str, task: str, seed: int) -> Path:
        return self.base_dir / config / task / str(seed) / "status.json"

    def progress_log_path(self, config: str, task: str) -> Path:
        return self.base_dir / config / task / "progress.log"

    def load(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def is_valid(self, record: Optional[Dict[str, Any]], expected_fingerprints: Dict[str, str]) -> bool:
        """A cached record is reusable only when every fingerprint matches."""
        if record is None:
            return False
        if record.get("record_version") != PREDICTION_RECORD_VERSION:
            return False
        got = record.get("fingerprints", {})
        return all(got.get(k) == v for k, v in expected_fingerprints.items())

    def save_atomic(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def log(self, config: str, task: str, line: str) -> None:
        path = self.progress_log_path(config, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def save_status(self, config: str, task: str, seed: int, status: Dict[str, Any]) -> None:
        self.save_atomic(self.status_path(config, task, seed), status)

    def load_status(self, config: str, task: str, seed: int) -> Dict[str, Any]:
        return self.load(self.status_path(config, task, seed)) or {}
