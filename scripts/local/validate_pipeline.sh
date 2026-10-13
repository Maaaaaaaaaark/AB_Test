#!/usr/bin/env bash
# Local (no-GPU, no-torch) end-to-end pipeline validation on the dev machine:
# build manifests -> scripted smoke run -> report -> worked examples.
# The scripted backend proves manifests, gating, persistence, scoring and
# reports; its outputs are NOT model outputs (see docs/UNRESOLVED.md #7).
#
# usage: bash scripts/local/validate_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${WEVAL_PYTHON:-$ROOT/.venv/Scripts/python.exe}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY (run: uv venv .venv && uv pip install -e . --no-deps)" >&2
  exit 1
fi
export PYTHONIOENCODING=utf-8

echo "== build (skipped if manifests already exist; run `weval build` to rebuild) =="
"$PY" -m weval.cli build || echo "build failed or already built - continuing (see above)"

echo "== scripted smoke run =="
"$PY" -m weval.cli run --plan smoke --backend scripted --default-text "NOT FOUND"

echo "== smoke report =="
"$PY" -m weval.cli report --plan smoke

echo "== worked examples =="
"$PY" -m weval.cli worked-examples --plan smoke

echo "== review summary =="
"$PY" -m weval.cli review list

echo "== tests =="
"$PY" -m pytest tests -q

echo "all local steps finished - see artifacts/reports/smoke/report.md and docs/worked_examples.md"
