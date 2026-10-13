# Copy to env.sh next to this file and edit. submit.sh sources env.sh.
# NO account, partition, GPU type or path is hard-coded in any job template.
# Values here are defaults for the sbatch overrides - every one of them can
# also be passed directly to submit.sh (see its usage line).

# --- Slurm ----------------------------------------------------------------
#SBATCH_ACCOUNT="CHANGE_ME"        # billing account (required on most clusters)
#SBATCH_PARTITION="gpu"            # partition with GPUs
#SBATCH_GPUS="1"                   # GPUs per job (5-config runs share 1 GPU
#                                  #   sequentially; increase to parallelize -
#                                  #   the runner is single-process per item)
#SBATCH_CPUS="8"
#SBATCH_MEM="64G"
#SBATCH_TIME="01:00:00"            # default; per-job suggested times are in
#                                  #   the templates' comments
#SBATCH_LOG_DIR="$WEVAL_ROOT/logs" # sbatch -o target (created automatically)

# --- Project ---------------------------------------------------------------
WEVAL_ROOT="${WEVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# --- Python ----------------------------------------------------------------
# The venv on the cluster (created with `uv pip install -e .[inference]`).
# Leave unset to use whatever python is active when sbatch runs the job.
#WEVAL_PYTHON="$WEVAL_ROOT/.venv/bin/python"

# --- Hugging Face ----------------------------------------------------------
# Where model/dataset caches live (point at a large shared disk).
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME

# HF token for the gated repos (Gemma, Llama). This project NEVER prints,
# hashes or stores it - it is only exported into the job environment.
# Alternatives: `huggingface-cli login` once on the login node (token then
# lives in $HF_HOME/token), or set HF_TOKEN below.
#HF_TOKEN="hf_..."                 # DO NOT commit env.sh with a token inside
