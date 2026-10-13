#!/usr/bin/env bash
# Descriptive whole-question references: 100 HotpotQA distractor, 100 GSM8K
# main test, 100 MATH test, seed 0, all 5 configs.
# References NEVER decide the worker choice and are never used to tune
# prompts. Suggested --time: 06:00:00 with 1 GPU.
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

"$PY" -m weval.cli run --plan reference --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto
"$PY" -m weval.cli report --plan reference
