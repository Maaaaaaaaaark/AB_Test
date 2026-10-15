#!/usr/bin/env bash
# Re-parse + re-score an existing plan's predictions and write
# artifacts/reports/<plan>/report.json + report.md. CPU-only, cheap.
# Select the plan via REPORT_PLAN:
#   REPORT_PLAN=pilot bash scripts/slurm/submit.sh report
# Suggested --time: 00:30:00.
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

PLAN="${REPORT_PLAN:?set REPORT_PLAN=smoke|pilot|selection|holdout|reference|math_solve}"
"$PY" -m weval.cli report --plan "$PLAN"
