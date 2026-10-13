#!/usr/bin/env bash
# Cluster readiness: token presence (boolean only - never printed), gated
# access for Gemma/Llama, template locks, torch/CUDA/bf16.
# Writes artifacts/precheck/precheck.json and prints a human summary.
# No GPU needed, no model weights needed. Suggested --time: 00:20:00.
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
# HF_TOKEN is exported by env.sh when the user set it; never printed here.

"$PY" -m weval.cli precheck
