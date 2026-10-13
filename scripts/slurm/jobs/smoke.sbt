#!/usr/bin/env bash
# Smoke plan on GPUs: 2 musique parents (with negatives), 2 processbench per
# subset, 2 whole-question references, seed 0, all 5 configs.
# Followed by the report and the smoke->pilot cost extrapolation.
# Suggested --time: 02:00:00 with 1 GPU (5 configs x ~60 items).
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

"$PY" -m weval.cli run --plan smoke --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto
"$PY" -m weval.cli report --plan smoke
"$PY" -m weval.cli estimate --source smoke --target pilot || true
