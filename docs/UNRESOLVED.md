# UNRESOLVED items

Everything in this file is a gap that blocks a specific later step (not the
whole project). Each item says exactly what is missing and what unblocks it.
Items are resolved by an explicit human decision or cluster action - never
silently.

Status legend: `OPEN` = blocks something, `PENDING-APPROVAL` = waiting for the
user's explicit sign-off, `RESOLVED` = done (kept for the record).

---

## 1. Gated model access: Gemma and Llama (blocks GPU runs for 2 of 5 configs)

- **Status:** OPEN
- **What is missing:** `google/gemma-3-4b-it` and `meta-llama/Llama-3.2-3B-Instruct`
  both report `gated=manual` on the Hub. Their `config.json` and tokenizer files
  return 401/403 without authentication, so their chat templates could not be
  pinned locally (`template_sha256: null` in `configs/models.yaml`).
- **What it blocks:** `weval precheck` will report them as not granted; the smoke,
  pilot, selection, holdout and reference GPU runs for these two configurations.
- **To resolve:**
  1. On the HF account used on the cluster: accept the Gemma license
     (https://huggingface.co/google/gemma-3-4b-it) and the Llama 3.2 community
     license (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
  2. Put a token for that account on the cluster (env var `HF_TOKEN`, or
     `huggingface-cli login`). The token is never printed, hashed or stored by
     this project.
  3. Run `weval precheck` on the cluster. It verifies the gated files are
     actually retrievable and proposes the exact `models.yaml` patch that pins
     `template_sha256` for these two models (templates lock on first precheck).
  4. Formal stages require `require_locked_templates: true`, so selection /
     holdout / reference are gated until this lock is in place.

## 2. MATH dataset source substitution (awaits explicit user sign-off)

- **Status:** PENDING-APPROVAL
- **What happened (full chain, nothing guessed):**
  1. The original source URL for `hendrycks/competition_math` returns **403**
     (recorded 2026-10-13).
  2. The `hendrycks/competition_math` dataset on the Hub reports
     `disabled=True` via the API and its resolve endpoints return 401
     "Invalid username or password" even for public requests; `datasets>=5`
     raises `DatasetNotFoundError` for it (legacy script-based dataset).
  3. `HuggingFaceH4/MATH` was evaluated and **rejected**: it is a partial
     subset (746 train / 546 test, not 7500 / 5000).
  4. Adopted: **`EleutherAI/hendrycks_math`**, pinned revision
     `21a5633873b6a120296cce3e2df9d5550074f4a3` (`weval/datasets/mathbench.py`).
     Verified at download time: 7 categories (algebra, counting_and_probability,
     geometry, intermediate_algebra, number_theory, prealgebra, precalculus)
     x train/test = **7500 / 5000** records, fields
     `{problem, solution, level, type}` matching the original MATH fields.
- **What it blocks:** treating the `math` reference task as settled. The data is
  downloaded, verified and provenance-recorded; only the explicit human
  acknowledgment is missing.
- **To resolve:** read this entry, then confirm in one sentence (reply to this
  document's author or edit the status line). After sign-off, flip this entry to
  RESOLVED and remove the "mirror" wording from `weval/cli.py` data-prep output
  if desired.

## 3. Human review of MuSiQue positives AND paired negatives (blocks pilot/selection/holdout MuSiQue)

- **Status:** OPEN (builder v2 regenerates every item: earlier decisions, if
  any, no longer match the content and are moved to `stale_decisions`)
- **What is missing:** every entry of
  `artifacts/manifests/musique_negative_review.json` now carries two
  decisions, both starting `proposed`:
  - negative (`review_status`): with the support paragraph replaced, is the
    question really unanswerable from the five visible paragraphs?
  - positive (`positive_review_status`): does the support paragraph, read on
    its own, answer the resolved question with the gold answer? (Added after
    the first smoke run: one positive's paragraph mentioned the answer
    string without stating the relation, another never named the entity.)
  Each entry shows the support paragraph and the replacement paragraph.
  No entry is ever auto-approved.
- **What it blocks:** `pilot`, `selection` and `holdout` set
  `require_approved_negatives: true` and `require_approved_positives: true`;
  selection/holdout refuse to run with any unreviewed MuSiQue item, pilot
  skips unreviewed items (counted). Smoke is not blocked.
- **To resolve:**
  ```
  weval review list --only proposed
  weval review approve --parent <id>                      # negative
  weval review exclude --parent <id> --note "<why>"
  weval review approve --parent <id> --target positive    # positive
  weval review exclude --parent <id> --target positive --note "<why>"
  ```
  Excluding requires a written note. `reset` returns a decision to
  `proposed`. Approval is the human's judgment; this tool only records it.
- Start with the 10 pilot parents and record how many positives had to be
  excluded: that rate decides whether MuSiQue sub-questions are a usable
  source for the evidence-worker test. Excluded positives reduce N; raise
  `stages.selection.musique_parents` if you need >= 100 approved positives.
- After reviewing, run `weval build` again: approved items re-enter the
  manifests as `review_status: approved`, human-excluded items drop out and
  are counted (`review:<reason>`, `positive_review:<reason>`), and decisions
  survive rebuilds as long as the item content is unchanged.

## 4. No sourced prices -> Cweighted is null

- **Status:** OPEN
- **What is missing:** `configs/prices.yaml` does not exist, so there are no
  sourced USD-per-token prices.
- **What it blocks:** the `Cweighted` (cost in USD) column. Reports print
  `Cweighted = null` with a note - it is **never** zeroed or invented.
  `Cproxy` (compute-proxy tokens) and `Efficiency_compute` are unaffected.
- **To resolve:** add `configs/prices.yaml` with sourced prices
  (provider, price per 1M input/output tokens, source URL, retrieval date),
  or leave null deliberately.

## 5. Cluster environment unknowns (block nothing until arrival)

- **Status:** OPEN
- **What is missing:** the cluster's hardware (GPU model/memory, bf16 support),
  Slurm partition/account names, and how `artifacts/` travels from this Windows
  machine to the cluster.
- **What it blocks:** writing concrete `#SBATCH` values (scripts ship with
  env-driven overrides, see `scripts/slurm/env.example.sh`), and the
  pilot-based time/cost extrapolation.
- **To resolve:** run `weval precheck` on the cluster (hardware section reports
  torch/CUDA/bf16); set the env vars in `scripts/slurm/env.example.sh`; transfer
  the project directory (including `artifacts/data` and `artifacts/manifests`,
  ~1 GB) with rsync/scp.

## 6. `inspect-params` requires a cluster model snapshot

- **Status:** OPEN
- **What is missing:** `N_text` (unique text-model parameters per checkpoint)
  comes from safetensors headers of an actual downloaded snapshot. No model
  weights exist on this machine (by design - no inference locally).
  `artifacts/model_params/` is empty.
- **What it blocks:** the `n_text` column of the compute table and therefore
  `Cproxy`/`Efficiency_compute` (they stay null until it runs).
- **To resolve:** on the cluster, after precheck, run per config:
  `weval inspect-params --config <name> --snapshot-dir models/<name>`
  (5 configs total; the two Qwen3 configs share one snapshot).

## 7. All current prediction records are scripted, not model outputs

- **Status:** OPEN (by design)
- **What is missing:** `artifacts/predictions/smoke/**` was produced by the
  scripted backend to validate the pipeline. No real model has generated a
  token in this project yet; no real inference was (or should be) run on this
  machine.
- **What it blocks:** interpreting any smoke number as a model result. The
  smoke report exists to prove the pipeline end-to-end.
- **To resolve:** run the smoke plan on the cluster with
  `--backend transformers` and re-run `weval report`.

## 8. Thresholds are draft (blocks formal selection by design)

- **Status:** OPEN (by design - user confirmation IS the gate)
- **What is missing:** `configs/thresholds.yaml` has
  `status: draft_unconfirmed`.
- **What it blocks:** `selection` and `holdout` plans
  (`require_locked_thresholds: true`). Smoke and pilot run regardless.
- **To resolve:** after reviewing the pilot results, the user confirms the
  thresholds (set `status: confirmed` with `confirmed_by`/`confirmed_at` -
  or change the values first, then confirm). After confirmation the values
  may not change again once selection scores have been inspected.

## 9. GSM8K Socratic sub-task deferred

- **Status:** RESOLVED (deferred by spec)
- Whole-question GSM8K (main test, `####` format) is implemented and is the
  only GSM8K condition in this project. The Socratic-answer variant was
  deferred in the original specification and is not implemented.

## 10. ProcessBench schema vs its README

- **Status:** RESOLVED (adapter matches the real data; documented in
  docs/PROTOCOL.md)
- The ProcessBench README mentions `final_answer` / `reference_answer`
  columns; the pinned revision
  (`3bdcd5371ed567559a78f559c01c13a6deee7604`) actually contains
  `{id, generator, problem, steps, final_answer_correct, label}`. The adapter
  follows the data. `final_answer_correct` is carried as informational context
  only; official scoring uses `label` only.
