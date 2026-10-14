"""Human review of MuSiQue local items (musique_negative_review.json).

The review file is a JSON list; every parent gets one entry with TWO
independent decisions:

  * the paired NEGATIVE (review_status / human_reviewed / ...): with the
    support paragraph replaced, is the question really unanswerable?
    Candidate pairs start "proposed"; build-excluded parents "excluded".
  * the POSITIVE (positive_review_status / positive_human_reviewed / ...):
    does the support paragraph, read on its own, answer the resolved question
    with the gold answer? Every positive starts "proposed".

Each entry carries the support paragraph (and the negative's replacement
paragraph) so the reviewer does not have to open the manifests.
Approval is a deliberate, item-level human action:

    weval review approve --parent 2hop__10017_18974 --note "negative is clearly unanswerable"
    weval review approve --parent 2hop__10017_18974 --target positive
    weval review exclude --parent 2hop__10017_18974 --target positive --note "paragraph does not state the relation"

There is no bulk-approve command. Builds re-read the file: approved items
re-enter manifests as review_status=approved; human-excluded items stay out.
A decision is honored only while the item content it was made on is
unchanged (content fingerprint). Formal stages (pilot, selection, holdout)
run approved items only, so review must precede them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REVIEW_VERSION = "review-v2"
VALID_DECISIONS = ("approve", "exclude", "reset")
VALID_TARGETS = ("negative", "positive")


class ReviewError(RuntimeError):
    pass


def _load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ReviewError(f"review file not found: {path}; run `weval build` first")
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ReviewError(f"{path} is not a review-file list")
    return entries


def _save(path: Path, entries: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _find_entry(entries: List[Dict[str, Any]], parent_id: str) -> Optional[Dict[str, Any]]:
    for e in entries:
        if e.get("parent_id") == parent_id:
            return e
    return None


def list_entries(path: Path, only: Optional[str] = None) -> List[Dict[str, Any]]:
    """only: proposed | approved | excluded | all (None = all)."""
    entries = _load(path)
    if only == "proposed":
        return [e for e in entries if e.get("review_status") == "proposed"]
    if only == "approved":
        return [e for e in entries if e.get("review_status") == "approved"]
    if only == "excluded":
        return [e for e in entries if e.get("review_status") == "excluded"]
    return entries


def _apply_positive_decision(
    entry: Dict[str, Any], decision: str, note: Optional[str], reviewed_by: str
) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if decision == "approve":
        entry["positive_review_status"] = "approved"
        entry["positive_human_reviewed"] = True
        entry["positive_reviewed_by"] = reviewed_by
        entry["positive_review_note"] = note
        entry["positive_excluded_reason"] = None
        entry["positive_reviewed_at_utc"] = now
    elif decision == "exclude":
        entry["positive_review_status"] = "excluded"
        entry["positive_human_reviewed"] = True
        entry["positive_reviewed_by"] = reviewed_by
        entry["positive_review_note"] = note
        entry["positive_excluded_reason"] = note
        entry["positive_reviewed_at_utc"] = now
    else:  # reset
        entry["positive_review_status"] = "proposed"
        entry["positive_human_reviewed"] = False
        entry["positive_reviewed_by"] = None
        entry["positive_review_note"] = None
        entry["positive_excluded_reason"] = None
        entry.pop("positive_reviewed_at_utc", None)


def apply_decision(
    path: Path,
    parent_id: str,
    decision: str,
    note: Optional[str] = None,
    reviewed_by: str = "human",
    target: str = "negative",
) -> Dict[str, Any]:
    if decision not in VALID_DECISIONS:
        raise ReviewError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
    if target not in VALID_TARGETS:
        raise ReviewError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    if decision == "exclude" and not note:
        raise ReviewError("--note is required for exclusion (recorded verbatim)")
    entries = _load(path)
    entry = _find_entry(entries, parent_id)
    if entry is None:
        raise ReviewError(f"parent {parent_id!r} not in review file ({len(entries)} entries)")
    if target == "positive":
        if not entry.get("positive_item_id"):
            raise ReviewError(f"parent {parent_id!r} has no positive item to review")
        _apply_positive_decision(entry, decision, note, reviewed_by)
        _save(path, entries)
        return entry
    if decision == "approve":
        if not entry.get("negative_item_id"):
            raise ReviewError(f"parent {parent_id!r} has no pair candidate (excluded at build: "
                              f"{entry.get('excluded_reason')}); nothing to approve")
        entry["review_status"] = "approved"
        entry["human_reviewed"] = True
        entry["reviewed_by"] = reviewed_by
        entry["review_note"] = note
        entry["reviewed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elif decision == "exclude":
        entry["review_status"] = "excluded"
        entry["human_reviewed"] = True
        entry["reviewed_by"] = reviewed_by
        entry["review_note"] = note
        entry["excluded_reason"] = entry.get("excluded_reason") or note
        entry["reviewed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    else:  # reset
        entry["review_status"] = "proposed" if entry.get("negative_item_id") else "excluded"
        entry["human_reviewed"] = False
        entry["reviewed_by"] = None
        entry["review_note"] = None
        entry.pop("reviewed_at_utc", None)
    _save(path, entries)
    return entry


def review_summary(path: Path) -> Dict[str, int]:
    entries = _load(path)
    out: Dict[str, int] = {"total": len(entries)}
    for key in ("proposed", "approved", "excluded"):
        out[key] = sum(1 for e in entries if e.get("review_status") == key)
    out["excluded_at_build"] = sum(
        1 for e in entries if e.get("review_status") == "excluded" and not e.get("human_reviewed")
    )
    out["human_excluded"] = sum(
        1 for e in entries if e.get("review_status") == "excluded" and e.get("human_reviewed")
    )
    return out


def positive_review_summary(path: Path) -> Dict[str, int]:
    """Counts of POSITIVE decisions (review_summary covers negatives)."""
    entries = _load(path)
    out: Dict[str, int] = {"total": sum(1 for e in entries if e.get("positive_item_id"))}
    for key in ("proposed", "approved", "excluded"):
        out[key] = sum(
            1 for e in entries
            if e.get("positive_item_id") and (e.get("positive_review_status") or "proposed") == key
        )
    out["with_stale_decisions"] = sum(1 for e in entries if e.get("stale_decisions"))
    return out


def print_summary(path: Path, only: Optional[str] = None) -> None:
    s = review_summary(path)
    ps = positive_review_summary(path)
    print(f"review file: {path}")
    print(f"  negatives: {s['total']} entries (proposed {s['proposed']}, approved {s['approved']}, "
          f"excluded {s['excluded']} = at-build {s['excluded_at_build']} + human {s['human_excluded']})")
    print(f"  positives: {ps['total']} (proposed {ps['proposed']}, approved {ps['approved']}, "
          f"excluded {ps['excluded']}); entries with stale (dropped) decisions: {ps['with_stale_decisions']}")
    if only is not None or s["total"] <= 40:
        for e in list_entries(path, only):
            flags = [k for k, v in (e.get("checks") or {}).items() if v]
            print(
                f"  {e['parent_id']}: neg={e.get('review_status')} pos={e.get('positive_review_status', 'proposed')}"
                + (f" by {e.get('reviewed_by')}" if e.get("human_reviewed") else "")
                + (f" note={e.get('review_note')!r}" if e.get("review_note") else "")
                + (f" excluded={e.get('excluded_reason')!r}" if e.get("excluded_reason") and not e.get("review_note") else "")
                + (f" flags={flags}" if flags else "")
                + f" | Q: {e.get('question_resolved', '')[:70]}"
                + f" | syntax={e.get('syntax', {}).get('has_relation') and 'relation' or 'natural'}"
            )
