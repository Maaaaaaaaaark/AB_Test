#!/usr/bin/env bash
# N_text per configuration from safetensors HEADERS only (no GPU, no weights
# loaded). Writes artifacts/model_params/<config>.json - the auditable
# parameter-count breakdown used by Cproxy.
# No GPU needed. Suggested --time: 00:30:00.
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

for cfg in qwen3-4b-thinking-off qwen3-4b-thinking-on gemma-3-4b-it \
           llama-3.2-3b-instruct deepseek-r1-distill-qwen-1.5b; do
  "$PY" -m weval.cli inspect-params --config "$cfg" \
      --snapshot-dir "$WEVAL_ROOT/models/$cfg"
done
