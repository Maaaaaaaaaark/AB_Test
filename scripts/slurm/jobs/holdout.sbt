#!/usr/bin/env bash
# FORMAL HOLDOUT - submit ONLY after the user explicitly approves it.
# 100 disjoint MuSiQue DEV parents + negatives, 25 ProcessBench per subset,
# seeds 0,1,2.
# The runner reads the selection report and runs ONLY the nominated
# candidate(s) from the holdout plan - holdout can never re-pick.
# Same gates as selection (thresholds confirmed, templates locked, approved
# negatives). Suggested --time: 06:00:00 with 1 GPU.
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

import yaml

root = Path(os.environ["WEVAL_ROOT"])
problems = []
thr = yaml.safe_load((root / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))
if thr.get("status") != "confirmed":
    problems.append("thresholds.yaml status is not 'confirmed'")
sel = root / "artifacts" / "reports" / "selection" / "report.json"
if not sel.exists():
    problems.append("selection report missing - run selection and its report first")
else:
    rec = json.loads(sel.read_text(encoding="utf-8")).get("recommendation") or {}
    inner = rec.get("recommendation") or {}
    nominated = sorted({inner.get("evidence_worker"), inner.get("math_checker")} - {None})
    if not nominated:
        problems.append("selection report nominates no candidate")
    else:
        print(f"holdout will run for nominated config(s): {nominated}")
if problems:
    sys.exit("pre-flight gate failed:\n - " + "\n - ".join(problems))
EOF

"$PY" -m weval.cli run --plan holdout --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto
"$PY" -m weval.cli report --plan holdout
