#!/usr/bin/env bash
# Submit a weval cluster job. All Slurm specifics come from env.sh (or the
# variables below); nothing is hard-coded in the job templates.
#
# usage:
#   bash scripts/slurm/submit.sh <job> [extra sbatch args...]
#
# jobs (templates in scripts/slurm/jobs/):
#   precheck          cluster readiness: token presence, gated access,
#                     template locks, torch/CUDA/bf16
#   snapshots         download model weights at the pinned revisions
#   inspect-params    N_text per config from safetensors headers (no GPU)
#   smoke             smoke plan on GPUs (seed 0)
#   pilot             pilot plan on GPUs (seed 0, approved negatives required)
#   selection         FORMAL selection (gated: confirmed thresholds, locked
#                     templates, approved negatives) - submit only after the
#                     user explicitly approves it
#   holdout           FORMAL holdout for the selection-nominated candidate
#   reference         descriptive whole-question references: HotpotQA,
#                     GSM8K, MATH (seed 0)
#   math_solve        standalone single-model math solving (GSM8K + MATH,
#                     seeds 0,1,2; descriptive, nominates nothing). One job
#                     per model: WEVAL_CONFIGS=<config> (see the template)
#   report            re-parse + re-score and write report.json/report.md
#
# examples:
#   bash scripts/slurm/submit.sh smoke
#   SBATCH_TIME=06:00:00 bash scripts/slurm/submit.sh pilot
#   bash scripts/slurm/submit.sh pilot --gres=gpu:2
#   WEVAL_CONFIGS=gemma-3-4b-it bash scripts/slurm/submit.sh math_solve
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SLURM_DIR/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SLURM_DIR/env.sh"
fi

JOB="${1:?usage: submit.sh <job> [extra sbatch args...]}"
shift || true
# per-config jobs (WEVAL_CONFIGS, exported to the job by sbatch) get their own
# job name / log file
JOB_TAG="$JOB${WEVAL_CONFIGS:+-${WEVAL_CONFIGS//,/+}}"

TEMPLATE="$SLURM_DIR/jobs/${JOB}.sbt"
if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: no job template for '$JOB' (looked for $TEMPLATE)" >&2
  echo "available: $(cd "$SLURM_DIR/jobs" && ls *.sbt | sed 's/\.sbt$//' | tr '\n' ' ')" >&2
  exit 1
fi

LOG_DIR="${SBATCH_LOG_DIR:-${WEVAL_ROOT:-$HOME}/logs}"
mkdir -p "$LOG_DIR"

echo "submitting $JOB_TAG ($TEMPLATE)"
sbatch \
  ${SBATCH_ACCOUNT:+--account="$SBATCH_ACCOUNT"} \
  ${SBATCH_PARTITION:+--partition="$SBATCH_PARTITION"} \
  ${SBATCH_GPUS:+--gres="gpu:${SBATCH_GPUS}"} \
  ${SBATCH_CPUS:+--cpus-per-task="$SBATCH_CPUS"} \
  ${SBATCH_MEM:+--mem="$SBATCH_MEM"} \
  ${SBATCH_TIME:+--time="$SBATCH_TIME"} \
  --job-name="weval-${JOB_TAG}" \
  --output="$LOG_DIR/weval-${JOB_TAG}-%j.out" \
  "$@" \
  "$TEMPLATE"
