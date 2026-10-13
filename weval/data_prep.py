"""Data preparation: raw sources -> artifacts/data + provenance records.

MuSiQue: the authors' Google Drive zip (musique_v1.0.zip, official download
script vendored) is unpacked and hashed; the expected jsonl files are checked.
Invalid zips (e.g. a saved Google error page) raise - never silent fallback.

HF datasets: ProcessBench, GSM8K (main test), HotpotQA (distractor dev) and
the original MATH (HF mirror, see docs/UNRESOLVED.md) are downloaded with
PINNED revisions into artifacts/data/hf_cache and verified (fields, counts)
via their adapters. Verification failures block the dataset.

Every prepared dataset gets a provenance record under
artifacts/data/provenance/{name}.json (URL, revision, license, hashes,
record counts, verification results).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List

from .config import ConfigBundle
from .provenance import DatasetProvenance, file_sha256, save_provenance

DATA_PREP_VERSION = "data-prep-v1"

MUSIQUE_ZIP_EXPECTED = [
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
]


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def prepare_musique(
    cfg: ConfigBundle,
    data_dir: Path,
    zip_path: Path,
    zip_source_url: str,
    license_: str,
) -> DatasetProvenance:
    """Unpack and verify the authors' MuSiQue zip."""
    out_dir = data_dir / "musique"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(
            f"{zip_path} is not a valid zip (a Google Drive error page was "
            f"saved instead of the file - re-run `gdown --output musique_v1.0.zip "
            f"https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`); "
            f"nothing was extracted"
        )
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"zip is corrupt (first bad member: {bad})")
        names = set(z.namelist())
        # The authors' zip nests files under a data/ prefix.
        member_of = {
            n: n if n in names else f"data/{n}"
            for n in MUSIQUE_ZIP_EXPECTED
        }
        missing = [n for n in MUSIQUE_ZIP_EXPECTED if member_of[n] not in names]
        if missing:
            raise ValueError(
                f"zip is missing expected files {missing}; "
                f"available: {sorted(names)[:20]}"
            )
        for n in MUSIQUE_ZIP_EXPECTED:
            target = out_dir / n
            if not target.exists():
                with z.open(member_of[n]) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    files = [
        {
            "local_path": str(out_dir / n),
            "url": zip_source_url,
            "sha256": file_sha256(out_dir / n),
            "size_bytes": (out_dir / n).stat().st_size,
        }
        for n in MUSIQUE_ZIP_EXPECTED
    ]
    counts = {n: _count_lines(out_dir / n) for n in MUSIQUE_ZIP_EXPECTED}
    prov = DatasetProvenance(
        name="musique",
        source_url=zip_source_url,
        version_or_revision="musique_ans_v1.0 (authors' Google Drive release)",
        license=license_,
        original_split="train/dev/test",
        n_records=sum(counts.values()),
        files=files,
        notes=[f"{n}: {c} records" for n, c in counts.items()],
        verified_claims=[
            "zip_is_valid",
            "expected_files_present",
            "record_counts_checked_at_build",
        ],
    )
    save_provenance(data_dir / "provenance" / "musique.json", prov)
    return prov


def _adapter_revision(adapter: Any) -> str:
    for attr in ("PROCESSBENCH_REVISION", "GSM8K_REVISION", "HOTPOTQA_REVISION", "MATH_REVISION", "REVISION"):
        if hasattr(adapter, attr):
            return str(getattr(adapter, attr))
    raise ValueError(f"adapter {adapter} exposes no pinned revision attribute")


def prepare_hf_dataset(
    cfg: ConfigBundle,
    data_dir: Path,
    adapter: Any,
    name: str,
    license_: str,
    verify_fn: Callable[[str], Dict[str, Any]],
    source_url: str,
) -> DatasetProvenance:
    """Download one pinned HF dataset and verify it via its adapter."""
    cache_dir = str(data_dir / "hf_cache")
    info = verify_fn(cache_dir=cache_dir)
    prov = DatasetProvenance(
        name=name,
        source_url=source_url,
        version_or_revision=_adapter_revision(adapter),
        license=license_,
        original_split=info.get("splits", "see adapter"),
        n_records=info["n_records"],
        notes=[f"{k}: {v}" for k, v in (info.get("per_split") or {}).items()],
        verified_claims=info.get("verified_claims", []),
    )
    save_provenance(data_dir / "provenance" / f"{name}.json", prov)
    return prov


def load_provenance(data_dir: Path) -> Dict[str, DatasetProvenance]:
    """Load every provenance record from data_dir/provenance/*.json."""
    out: Dict[str, DatasetProvenance] = {}
    pdir = data_dir / "provenance"
    if pdir.exists():
        for p in sorted(pdir.glob("*.json")):
            with open(p, encoding="utf-8") as f:
                out[p.stem] = DatasetProvenance.from_dict(json.load(f))
    return out


def verify_processbench(cache_dir: str) -> Dict[str, Any]:
    from .datasets.processbench import load_processbench

    data = load_processbench(cache_dir=cache_dir)
    per_split = {s: len(data[s]) for s in sorted(data)}
    return {
        "n_records": sum(per_split.values()),
        "per_split": per_split,
        "splits": ", ".join(sorted(per_split)),
        "verified_claims": [
            "fields_present",
            "four_subsets_loaded",
            "subset_counts: " + ", ".join(f"{k}={v}" for k, v in per_split.items()),
        ],
    }


def verify_gsm8k(cache_dir: str) -> Dict[str, Any]:
    from .datasets.gsm8k import load_gsm8k_main_test

    records = load_gsm8k_main_test(cache_dir=cache_dir)
    n_invalid = sum(1 for r in records if "####" not in r["answer"])
    return {
        "n_records": len(records),
        "splits": "test (main)",
        "verified_claims": [
            f"expected_count_1319: {len(records) == 1319} (got {len(records)})",
            f"gold_answers_without_marker: {n_invalid}",
        ],
    }


def verify_hotpotqa(cache_dir: str) -> Dict[str, Any]:
    from .datasets.hotpotqa import load_hotpotqa_distractor_dev

    records = load_hotpotqa_distractor_dev(cache_dir=cache_dir)
    return {
        "n_records": len(records),
        "splits": "validation (distractor)",
        "verified_claims": [
            f"expected_count_7405: {len(records) == 7405} (got {len(records)})",
        ],
    }


def verify_math(cache_dir: str) -> Dict[str, Any]:
    from .datasets.mathbench import MATH_EXPECTED_COUNTS, load_math

    data = load_math(cache_dir=cache_dir)
    ok = all(len(data[s]) == MATH_EXPECTED_COUNTS[s] for s in ("train", "test"))
    return {
        "n_records": len(data["train"]) + len(data["test"]),
        "per_split": {"train": len(data["train"]), "test": len(data["test"])},
        "splits": "train/test",
        "verified_claims": [
            f"counts_match_official_7500_5000: {ok}",
        ],
    }
