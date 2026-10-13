"""Compute-cost metrics.

Cproxy = 2 * N_text * (Tin + Tout) per item, where
  * N_text  = unique text-model parameter count from the actual checkpoint
              (weval/models/params.py: safetensors headers, tied weights
              counted once, Gemma vision tower excluded),
  * Tin     = post-native-template input tokens INCLUDING the prefilled
              prefix (full prompt token ids),
  * Tout    = ALL generated tokens (reasoning + final answer + eos).

Cweighted = null unless sourced prices exist (configs/prices.yaml with
retrieval date + source URL). It is NEVER zeroed or invented.

Efficiency_compute = mean(score_0_to_1) / mean(Cproxy / 1e12), computed only
within the same task (never across tasks).

Positive and negative costs are reported separately; a 50/50 mix is an
ASSUMPTION for workload planning, not a prevalence claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

COMPUTE_VERSION = "compute-metrics-v1"


@dataclass
class PriceSource:
    usd_per_1m_input: float
    usd_per_1m_output: float
    source_url: str
    retrieved_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usd_per_1m_input": self.usd_per_1m_input,
            "usd_per_1m_output": self.usd_per_1m_output,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }


def load_prices(root: Path) -> Optional[PriceSource]:
    """Sourced prices only. Absent file => None (Cweighted stays null)."""
    path = Path(root) / "configs" / "prices.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    required = {"usd_per_1m_input", "usd_per_1m_output", "source_url", "retrieved_at"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"configs/prices.yaml is missing {sorted(missing)}; refusing to guess prices")
    return PriceSource(
        usd_per_1m_input=float(raw["usd_per_1m_input"]),
        usd_per_1m_output=float(raw["usd_per_1m_output"]),
        source_url=str(raw["source_url"]),
        retrieved_at=str(raw["retrieved_at"]),
    )


def cproxy_per_item(n_text: int, tin: int, tout: int) -> float:
    """2 * N_text * (Tin + Tout)."""
    return 2.0 * n_text * (tin + tout)


def aggregate_cproxy(
    per_item: List[Dict[str, Any]],
    kind_key: str = "kind",
) -> Dict[str, Any]:
    """Sum/mean of Cproxy, split by item kind when present (positive/negative
    costs are reported separately)."""
    out: Dict[str, Any] = {"n_items": len(per_item), "sum": 0.0, "mean": None}
    kinds: Dict[str, Dict[str, Any]] = {}
    for it in per_item:
        c = it.get("cproxy")
        if c is None:
            continue
        out["sum"] += c
        k = str(it.get(kind_key, "na"))
        if k not in kinds:
            kinds[k] = {"sum": 0.0, "n": 0}
        kinds[k]["sum"] += c
        kinds[k]["n"] += 1
    if per_item:
        n_valid = sum(1 for it in per_item if it.get("cproxy") is not None)
        out["mean"] = out["sum"] / n_valid if n_valid else None
        out["n_with_cproxy"] = n_valid
    for k, v in kinds.items():
        v["mean"] = v["sum"] / v["n"] if v["n"] else None
    out["by_kind"] = kinds
    return out


def cweighted(input_tokens: int, output_tokens: int, price: Optional[PriceSource]) -> Optional[float]:
    """USD cost from sourced prices; None when no prices are available."""
    if price is None:
        return None
    return (
        input_tokens / 1_000_000 * price.usd_per_1m_input
        + output_tokens / 1_000_000 * price.usd_per_1m_output
    )


def efficiency_compute(scores: List[float], cproxies: List[float]) -> Optional[float]:
    """mean(score_0_to_1) / mean(Cproxy / 1e12). Same-task-only comparison."""
    if not scores or not cproxies or len(scores) != len(cproxies):
        return None
    mean_score = sum(scores) / len(scores)
    mean_cproxy_t = sum(cproxies) / len(cproxies) / 1e12
    if mean_cproxy_t == 0:
        return None
    return mean_score / mean_cproxy_t
