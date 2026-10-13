#!/usr/bin/env bash
# FORMAL SELECTION - submit ONLY after the user explicitly approves it.
# 100 MuSiQue DEV parents + approved negatives, 25 ProcessBench per subset,
# seeds 0,1,2, all 5 configs.
# Gates (pre-flight below, re-checked by the runner):
#   - thresholds.yaml status == confirmed (user confirmation IS the gate),
#   - all config templates locked (weval precheck),
#   - approved negative pairs present.
# Suggested --time: 12:00:00 with 1 GPU (5 configs x ~300 items x 3 seeds);
# consider more GPUs / longer wall time after the pilot estimate.
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

import yaml
from pathlib import Path

root = Path(os.environ["WEVAL_ROOT"])
problems = []
thr = yaml.safe_load((root / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))
if thr.get("status") != "confirmed":
    problems.append(
        "thresholds.yaml status is not 'confirmed' - formal selection is gated. "
        "Review the pilot report, confirm the thresholds, then resubmit."
    )
review = json.loads((root / "artifacts" / "manifests" / "musique_negative_review.json").read_text(encoding="utf-8"))
approved = [e for e in review
            if e.get("review_status") == "approved" and e.get("human_reviewed") and e.get("negative_item_id")]
if not approved:
    problems.append("no approved negative pairs - run `weval review approve --parent <id>` first")
models = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))
unlocked = [n for n, m in models["models"].items() if not m.get("template_sha256")]
if unlocked:
    problems.append(f"templates not locked for {unlocked} - run `weval precheck` on the cluster first")
if problems:
    sys.exit("pre-flight gate failed:\n - " + "\n - ".join(problems))
print(f"pre-flight ok: thresholds confirmed, {len(approved)} approved pairs, templates locked")
EOF

"$PY" -m weval.cli run --plan selection --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto
"$PY" -m weval.cli report --plan selection
