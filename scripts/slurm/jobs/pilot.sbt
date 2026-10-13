#!/usr/bin/env bash
# Pilot plan on GPUs: 10 MuSiQue TRAIN parents, 5 ProcessBench per subset,
# 10 whole-question references, seed 0, all 5 configs.
# Gate: require_approved_negatives - pre-flight below refuses to queue the job
# with zero approved pairs (save the queue slot; the runner re-checks too).
# Suggested --time: 04:00:00 with 1 GPU (5 configs x ~130 items).
#SBATCH --nodes=1
#SBATCH --ntasks=1
# (account/partition/gres/time/mem are passed by scripts/slurm/submit.sh)
set -euo pipefail
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$SLURM_DIR/env.sh" ]]; then source "$SLURM_DIR/env.sh"; fi
WEVAL_ROOT="$(cd "${WEVAL_ROOT:-$SLURM_DIR/../..}" && pwd)"
cd "$WEVAL_ROOT"
PY="${WEVAL_PYTHON:-python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WEVAL_ROOT

"$PY" - <<'EOF'
import json, os, sys
from pathlib import Path
root = Path(os.environ["WEVAL_ROOT"])
review = json.loads((root / "artifacts" / "manifests" / "musique_negative_review.json").read_text(encoding="utf-8"))
approved = [e for e in review
            if e.get("review_status") == "approved" and e.get("human_reviewed") and e.get("negative_item_id")]
if not approved:
    sys.exit("pre-flight gate failed: no approved negative pairs - run `weval review approve --parent <id>` first")
print(f"pre-flight ok: {len(approved)} approved pairs")
EOF

"$PY" -m weval.cli run --plan pilot --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto
"$PY" -m weval.cli report --plan pilot
"$PY" -m weval.cli estimate --source pilot --target selection || true
