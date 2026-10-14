# weval - worker-model evaluation (evidence QA + math checking)

A clean, standalone evaluation project for **frozen** models. No training, no
fine-tuning, no existing framework dependency. It measures whether small
instruction-tuned models can serve as two kinds of "worker":

1. **Evidence worker** - local QA with abstention on a custom MuSiQue-derived
   task (answer from 5 provided paragraphs, or exactly `NOT FOUND`; includes
   human-reviewed paired negatives).
2. **Math checker** - ProcessBench step-error detection.

Five frozen configurations, four models (every repo/revision/tokenizer/
chat-template is pinned):

| config | repo | thinking |
|---|---|---|
| `qwen3-4b-thinking-off` | `Qwen/Qwen3-4B` | disabled (declared switch) |
| `qwen3-4b-thinking-on` | `Qwen/Qwen3-4B` | optional (`<think>...</think>`) |
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | no thinking channel |
| `llama-3.2-3b-instruct` | `meta-llama/Llama-3.2-3B-Instruct` | no thinking channel |
| `deepseek-r1-distill-qwen-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | prefilled thinking |

Tasks: `musique_local` (positive + paired negatives), `processbench` (4 subsets
reported separately), and whole-question references `gsm8k`, `math`, `hotpotqa`.

**Where things run:** this Windows machine is the development machine only.
All GPU inference runs on a Slurm cluster (`scripts/slurm/`). Local runs use a
scripted backend for pipeline validation - nothing here downloads weights or
runs a model locally.

Full protocol (official vs project-defined, per task): [docs/PROTOCOL.md](docs/PROTOCOL.md).
Open items needing human decisions: [docs/UNRESOLVED.md](docs/UNRESOLVED.md).
Worked examples (exactly what the model sees): [docs/worked_examples.md](docs/worked_examples.md).

---

## Repository layout

```
configs/            models.yaml (pinned repos/revisions/templates), tasks.yaml,
                    splits.yaml, thresholds.yaml, plans/{smoke,pilot,selection,holdout,reference}.yaml
vendor/             vendored official evaluation code (verbatim, pinned; see vendor/VENDORED.md)
weval/              package: adapters, prompt renderers, parsers, scorers, metrics, runner
artifacts/          data/ (verified datasets + provenance), manifests/, predictions/,
                    reports/, precheck/, estimates/, model_params/
scripts/slurm/      cluster job templates + env override file
scripts/local/      local (no-GPU) convenience scripts
tests/              pytest suite (no torch, no network)
docs/               PROTOCOL.md, UNRESOLVED.md, worked_examples.md
```

## Setup

### Development machine (this machine)

Requires Python 3.11 and `uv`. No torch, no GPU, no HF token needed for the
local steps.

```bash
uv venv .venv
uv pip install -e . --no-deps        # base deps only
uv pip install -e .[dev] --no-deps   # + pytest
```

(The local venv has no pip by design - use `uv pip`.)

### Cluster (inference)

```bash
uv venv .venv
uv pip install -e .[inference] --no-deps   # adds torch/transformers/accelerate
cp scripts/slurm/env.example.sh scripts/slurm/env.sh   # then edit: partition,
                                                       # account, HF_HOME
```

Transfer the whole project (configs + code + `artifacts/data` +
`artifacts/manifests`, roughly 1 GB) to the cluster, e.g. `rsync -a`.

## Local workflow (data + manifests, no GPU)

```bash
# 1. one-time: download the MuSiQue zip from the authors' Google Drive release
python -m gdown --output _recon/musique_v1.0.zip \
  "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

# 2. verify MuSiQue + download/verify the pinned HF datasets (provenance saved)
weval data-prep

# 3. build all manifests (leakage checks, splits, negative-pair candidates)
weval build

# 4. pipeline smoke test with the scripted backend (no model)
weval run --plan smoke --backend scripted --default-text "NOT FOUND"
weval report --plan smoke
weval worked-examples --plan smoke

# 5. human review (REQUIRED before pilot): each parent has TWO decisions.
#    The review file shows the support paragraph and the negative's
#    replacement paragraph, so no manifest needs to be opened.
#    negative: with the support paragraph replaced, is it really unanswerable?
weval review list --only proposed
weval review approve --parent <parent_id>
weval review exclude --parent <parent_id> --note "<reason>"
#    positive: does the support paragraph, read on its own, answer the
#    resolved question with the gold answer?
weval review approve --parent <parent_id> --target positive
weval review exclude --parent <parent_id> --target positive --note "<reason>"

# Freeze the review decisions into the manifests used by the runner.
weval build

# 6. run the test suite
python -m pytest tests -q
```

`weval run` with the scripted backend exists to validate manifests, gating,
persistence and reports. Its outputs are clearly not model outputs.

## Cluster workflow (inference)

```bash
# 0. on every node / after any env change
weval precheck                       # token presence (boolean only), gated
                                     # access for Gemma/Llama, template locks,
                                     # torch/CUDA/bf16 -> artifacts/precheck/

# The first successful precheck downloads the pinned Gemma/Llama tokenizer
# chat templates into configs/templates/ and prints their exact sha256 values.
# Copy those two hashes into configs/models.yaml, then rerun precheck. Formal
# selection/holdout remains blocked until both hashes are pinned and MATCH.

# 1. download model snapshots (once; the gated licenses must be accepted first)
bash scripts/slurm/submit.sh snapshots   # snapshot_download at the pinned
                                         # revisions -> models/<config>/
weval inspect-params --config qwen3-4b-thinking-off --snapshot-dir models/qwen3-4b-thinking-off
# ... per config (Qwen3 on/off share one snapshot) -> artifacts/model_params/
# Repeat inspect-params for every configuration. Missing N_text prevents a
# winner from being nominated; it is never treated as zero compute.

# 2. smoke on GPUs: 2 items/task/config, seed 0
bash scripts/slurm/submit.sh smoke
weval report --plan smoke
weval estimate --source smoke --target pilot

# 3. pilot: 10 musique parents + 5 processbench/subset + 10 references
#    (requires approved negative pairs - see UNRESOLVED.md #3)
bash scripts/slurm/submit.sh pilot
weval report --plan pilot               # informational: smoke/pilot/reference
                                        # reports never nominate a worker
weval estimate --source pilot --target selection

# 4. FORMAL SELECTION - gated, never auto-launched:
#    - thresholds.yaml confirmed by the user
#    - templates locked (precheck)
#    - approved negative pairs
#    only after explicit user approval:
bash scripts/slurm/submit.sh selection
weval report --plan selection          # per-seed + mean, clustered CIs,
                                       # qualification table, Pareto, Cproxy

# 5. holdout: runs ONLY the selection-nominated candidate, never re-picks
bash scripts/slurm/submit.sh holdout
weval report --plan holdout            # pass/fail of the selection nominee

# 6. descriptive whole-question references (seed 0, never decide the worker)
bash scripts/slurm/submit.sh reference
```

Slurm scripts read `scripts/slurm/env.sh` (partition, account, GPUs, time,
paths). No account names are hard-coded anywhere.

## Configuration

- `configs/models.yaml` - repos/revisions pinned (fetched from the HF API
  2026-10-13), reasoning modes, `template_sha256` locks. Gated models lock
  their templates at the first cluster precheck.
- `configs/tasks.yaml` - instructions (identical for every config), budgets
  (MuSiQue 2048 new tokens; ProcessBench/whole-question 4096), scorers.
- `configs/plans/*.yaml` - per-stage item counts, seeds, and the three gates:
  `require_approved_negatives`, `require_locked_templates`,
  `require_locked_thresholds`.
- `configs/splits.yaml` - stage sizes, per-stage seeds (selection/holdout
  0,1,2; everything else 0), frozen split seeds.
- `configs/thresholds.yaml` - qualification thresholds (draft until the user
  confirms them; confirmation gates formal selection).

## Generation settings (identical across configs)

BF16, no quantization, eval mode, transformers, batch 1; `do_sample=True`,
`temperature=0.6`, `top_p=0.95`, `top_k=0`, `repetition_penalty=1.0`; one
generation per item per seed with stable order-independent per-item seeds.
The record stores the full merged parameter set. Token-limit termination is an
evaluated outcome; timeout/OOM are infra failures with at most 2 retries
(wrong answers are never retried). Raw token IDs and raw text are stored
before cleanup.

## Metrics summary

- MuSiQue: answerable F1 (official EM/F1 max over golds), false-abstention
  rate, reviewed negative correct-abstention; relation vs natural
  subquestions reported separately.
- ProcessBench: official per-subset error acc / correct acc / F1, plus the
  project macro-averages used for qualification.
- GSM8K: official `####` extraction (first match), extraction failure rate
  reported; relaxed parser diagnostic only. MATH: boxed extraction +
  vendored equivalence (missing gold boxed = unscorable, counted). HotpotQA:
  official answer F1.
- Cost: `Cproxy = 2 * N_text * (Tin + Tout)` (N_text from checkpoint
  safetensors headers, tied weights once); `Cweighted` null without sourced
  prices; `Efficiency_compute = mean(score) / mean(Cproxy/1e12)` per task;
  positive/negative costs separate (50:50 mix is a declared assumption).
- Statistics: per seed + cross-seed mean, paired config diffs, 95% bootstrap
  CIs clustered by parent problem group (pairs/variants/seeds together).

## Data provenance

Every downloaded dataset gets a provenance record under `artifacts/data/`
(URL, pinned revision, sha256, license, record counts, verified claims,
filtering reasons). See the per-dataset `provenance.json` files and
`docs/UNRESOLVED.md` (#2 documents the MATH source-substitution chain).

## Testing

```bash
python -m pytest tests -q
```

Covers: reference resolution, leakage, aliases, strict distractors, review
file, negative screening, official-scorer parity, reasoning parsing, missing
answer scoring, runner resume/fingerprints/retries, splits, templates, cost
params, seeds, renderers. No torch, no network.

## Policy

- **Never auto-launch a full evaluation.** Smoke/pilot may run when hardware
  and permissions allow; selection/holdout/reference only after explicit
  user approval and the gates above.
- **Never print or store HF tokens.** Precheck records token presence as a
  boolean only.
- **Never substitute models or datasets silently.** Changes require a new
  pinned config entry; datasets require provenance + verification.
