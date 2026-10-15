#!/usr/bin/env bash
# Standalone single-model math solving (descriptive; nominates nothing).
# 200 GSM8K main-test problems (simple random) + 200 MATH test problems
# (40 per Level 1-5), seeds 0,1,2, the 4 candidate configs. Samples are frozen
# by `weval build` (artifacts/manifests/{gsm8k,math}_math_solve.json) and
# exclude every ProcessBench problem, the reference/pilot problems and the
# GSM8K prompt exemplar (artifacts/manifests/math_solve_exclusions.json).
#
# One job per model (recommended - Qwen thinking-on dominates the cost):
#   for c in qwen3-4b-thinking-off qwen3-4b-thinking-on gemma-3-4b-it llama-3.2-3b-instruct; do
#     WEVAL_CONFIGS=$c SBATCH_TIME=24:00:00 bash scripts/slurm/submit.sh math_solve
#   done
#   # after ALL jobs finished:
#   REPORT_PLAN=math_solve bash scripts/slurm/submit.sh report
# All configs in one job: leave WEVAL_CONFIGS unset (the report is then
# written at the end of the job).
# Resubmitting the same command resumes: finished items are reused.
# Suggested --time: from `weval estimate --source pilot --target math_solve`.
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

for m in artifacts/manifests/gsm8k_math_solve.json artifacts/manifests/math_math_solve.json; do
  if [[ ! -f "$m" ]]; then
    echo "missing $m - run \`weval build\` first" >&2
    exit 1
  fi
done

echo "math_solve configs: ${WEVAL_CONFIGS:-<all in plan>}"
"$PY" -m weval.cli run --plan math_solve --backend transformers \
    --snapshots "$WEVAL_ROOT/models" --device auto \
    ${WEVAL_CONFIGS:+--configs "$WEVAL_CONFIGS"}

if [[ -z "${WEVAL_CONFIGS:-}" ]]; then
  "$PY" -m weval.cli report --plan math_solve
else
  echo "per-config job done; write the report after all jobs finished:"
  echo "  REPORT_PLAN=math_solve bash scripts/slurm/submit.sh report"
fi
