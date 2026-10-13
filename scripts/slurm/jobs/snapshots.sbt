#!/usr/bin/env bash
# Download model weights at the PINNED revisions into models/<config>/.
# Gated repos (Gemma, Llama) require the licenses accepted on the HF account
# and HF_TOKEN set in env.sh (or `huggingface-cli login` once). The token is
# never printed. snapshot_download resumes partial downloads.
# No GPU needed. Suggested --time: 04:00:00 (5 snapshots, ~5-8 GB each).
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
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

root = Path(os.environ["WEVAL_ROOT"])
cfg = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))
seen = {}
for name, m in cfg["models"].items():
    dest = root / "models" / name
    key = (m["repo"], m["revision"], m.get("tokenizer_revision"))
    canonical = seen.get(key)
    if canonical is not None:
        # Multiple evaluation configurations may intentionally use the exact
        # same checkpoint (Qwen thinking on/off differs only in template
        # kwargs).  Store the weights once and expose the second config via a
        # relative symlink so project relocation does not break it.
        if dest.is_symlink():
            if dest.resolve() != canonical.resolve():
                raise RuntimeError(f"{dest} points to {dest.resolve()}, expected {canonical.resolve()}")
        elif dest.exists():
            if any(dest.iterdir()):
                raise RuntimeError(
                    f"refusing to replace non-empty duplicate snapshot directory {dest}; "
                    "move it aside and rerun"
                )
            dest.rmdir()
            dest.symlink_to(os.path.relpath(canonical, dest.parent), target_is_directory=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(canonical, dest.parent), target_is_directory=True)
        print(f"{name}: reusing identical snapshot from {canonical}")
        continue
    seen[key] = dest
    if (dest / "snapshot_complete").exists():
        print(f"{name}: snapshot present, skipping")
        continue
    print(f"{name}: downloading {m['repo']} @ {m['revision']} -> {dest}")
    snapshot_download(repo_id=m["repo"], revision=m["revision"], local_dir=dest)
    (dest / "snapshot_complete").write_text("revision=" + m["revision"], encoding="utf-8")
print("all snapshots present")
EOF
