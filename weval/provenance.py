"""Dataset provenance records.

Every prepared dataset file gets a provenance record: source URL, release or
revision, original split, record IDs, file hashes, license and filtering
reasons. These records are part of the data manifest fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import canonical_json


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class SourceFile:
    local_path: str
    url: str
    sha256: str
    size_bytes: int
    note: str = ""


@dataclass
class DatasetProvenance:
    name: str
    source_url: str
    version_or_revision: str
    license: str
    original_split: str
    n_records: int
    acquired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    record_id_examples: List[str] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    filtering_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    verified_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetProvenance":
        return cls(**d)

    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


def save_provenance(path: Path, prov: DatasetProvenance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov.to_dict(), f, indent=2, ensure_ascii=False)


def load_provenance(path: Path) -> DatasetProvenance:
    with open(path, encoding="utf-8") as f:
        return DatasetProvenance.from_dict(json.load(f))
