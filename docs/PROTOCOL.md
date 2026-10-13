# Protocol: official vs project-defined, task by task

One inventory, one rule: **official where official exists** (data, prompt,
extraction, scoring - vendored verbatim with pinned source revisions), and
**every project-defined deviation is listed here with its reason**. Nothing
else is allowed to differ without a new entry in this file.

Vendored sources (see `vendor/VENDORED.md` for exact revision pins):

| vendor dir | source | used for |
|---|---|---|
| `vendor/musique/` | MuSiQue official repo `evaluate_v1.0.py` + `metrics/` | answer normalization, EM/F1 |
| `vendor/processbench/` | QwenLM/ProcessBench repo (`critique_template.txt`, `run_eval.py`) | critique prompt, extraction, scoring |
| `vendor/gsm8k/` | openai/grade-school-math | `####` answer extraction regex |
| `vendor/hotpotqa/` | HotpotQA official evaluation | answer normalization, EM/F1 |
| `vendor/math/` | hendrycks MATH `math_equivalence.py` | boxed-answer equivalence heuristic |

---

## MuSiQue (local evidence task)

**Official (verbatim):**
- Data: MuSiQue-Ans v1.0 dev/train (downloaded from the authors' Google Drive
  release; provenance recorded with URL, license cc-by-4.0, record counts).
- Answer scoring: official `AnswerMetric` - `normalize_answer`, token-level
  `compute_f1`, `compute_exact`, max over all gold answers.
- `X >> relation` subquestion syntax is kept exactly as in the data.

**Project-defined (this is a new task design, not the official MuSiQue setup):**
1. *One subquestion per parent.* Each parent contributes one subquestion
   (chosen with a per-parent seeded pick; the first hop is avoided whenever at
   least two eligible subquestions exist). Hop-position and question-format
   distributions are recorded in the manifest.
2. *`#N` references are resolved inline* with the canonical prerequisite
   answer (from the decomposition), deterministically - never rewritten by an
   LLM, never left as `#N` tokens.
3. *Five visible paragraphs:* the annotated supporting paragraph plus four
   non-supporting paragraphs of the same original example. Strict rule: a
   paragraph supporting ANY other subquestion of the same parent is never used
   as a distractor (it would leak intermediate evidence). Parents with an
   insufficient pool are excluded and counted.
4. *Abstention as a first-class label:* the instruction requires the model to
   answer or output exactly `NOT FOUND`. A positive item answered with
   NOT FOUND is a false abstention (scored 0, flagged); a negative item
   answered with exactly NOT FOUND scores 1.
5. *Gold for positives:* the subquestion's canonical answer, plus the parent
   answer and parent aliases at the terminal step (aliases are used only when
   the canonical subanswer equals the parent answer under official
   normalization; the alias source is recorded per item).
6. *Paired negatives (project-constructed):* the supporting paragraph is
   replaced by a 5th non-supporting paragraph at the same slot; the
   instruction is identical ("output exactly NOT FOUND" is the only expected
   answer). Built pairs are screened for answer/alias leakage (into titles,
   paragraph text, or prerequisite answers). Ambiguous yes/no and numeric
   negatives are excluded by policy unless explicitly reviewed. **Only
   human-approved pairs** (review file) enter the formal abstention
   evaluation; unreviewed pairs are dev-only and reported separately.
   Human decisions survive rebuilds (the review file is regenerated on every
   build, but `human_reviewed` entries keep their decision fields), and
   human-excluded pairs are reported as `review:<reason>` exclusions in the
   build coverage instead of silently disappearing.

## ProcessBench

**Official (verbatim):**
- Data at pinned revision `3bdcd5371ed567559a78f559c01c13a6deee7604`
  (counts 400/1000/1000/1000 per subset verified).
- Critique prompt: official template, applied through the model's native chat
  template.
- Extraction: the **last** `\boxed{...}` on the FULL generated text -> int
  (`-1` allowed); non-numeric boxed content = invalid format.
- Scoring: `pred == label`; error-solution accuracy (label != -1),
  correct-solution accuracy (label == -1), subset F1 = 2ab/(a+b).
- The 4 subset scores are reported separately, always.

**Project-defined:**
1. *Partition.* The official release has no train/test split, so selection and
   holdout partitions are project-defined: whole problem groups (a problem and
   all its step-level variants) move together. Group order = first occurrence
   in the official order, then a deterministic shuffle seeded with
   `split_seeds.processbench_group_order` (same seed -> identical partition;
   recorded in the manifest), and the error/no-error ratio is preserved
   approximately per subset (recorded in the manifest). Debug, selection and
   holdout consume consecutive whole groups of that order; pilot is selected
   from the debug groups. Stage selection (`_select_items`) also keeps groups
   whole for pilot/selection/holdout; only the smoke screen applies a flat
   per-subset cap that may cut a group mid-group (dev-only output).
2. *Macro-average.* The macro F1 and the macro correct/error accuracies
   (equal-weight mean over the 4 subsets) are project aggregations used for
   the math-checker qualification; the per-subset official scores are always
   shown next to them.
3. *Schema note:* the ProcessBench README mentions `final_answer` and
   `reference_answer` columns; the pinned revision does not contain them
   (actual: `id, generator, problem, steps, final_answer_correct, label`).
   The adapter follows the data. `final_answer_correct` is recorded as
   informational context and never used in scoring.

## GSM8K (whole-question reference)

**Official:** main test split; answer extraction by the official regex
(`#### (\-?[0-9\.\,]+)`), **first** match, `INVALID_ANS` on malformed
`####` answers.

**Project-defined:**
1. *Instruction* (custom): "put your final numeric answer on a line by itself
   in the format `#### <number>`" - the `####` format is required by the
   official extractor.
2. *Relaxed parser* exists as a secondary diagnostic only (it is reported how
   many official-extraction failures it would rescue); the official
   extraction failure rate is reported as a primary metric. The relaxed
   parser never determines a score.
3. Socratic variant: deferred (see UNRESOLVED.md).

## MATH (whole-question reference)

**Official:** original MATH test split (see UNRESOLVED.md #2 for the
source-substitution record); the official `math_equivalence.py` heuristic,
vendored verbatim, comparing the extracted predicted boxed answer against the
gold boxed answer.

**Project-defined:**
1. *Extraction is separate from equivalence* (`\boxed{...}` extraction, then
   the official checker). A missing predicted boxed answer is incorrect
   (never retried, never matched).
2. *A missing gold boxed answer makes the item unscorable* - counted and
   reported explicitly, never silently dropped. Two missing answers are
   NEVER treated as a match.
3. The checker is explicitly not claimed to be a full symbolic solver; scores
   are only as good as the official heuristic.
4. MATH-500 (HuggingFaceH4/MATH-500) is forbidden by spec; the 5000-record
   original test split is used (7500/5000 counts verified).

## HotpotQA (whole-question reference)

**Official:** distractor dev split, full question + all 10 paragraphs;
answer normalization and token F1 vendored from the official evaluation.

**Project-defined:** answer-only scope - supporting-facts prediction and the
joint EM/F1 are not part of this project, so only the answer metrics are
computed (the vendored functions are used verbatim for the parts that apply).

---

## Generation protocol (identical for all configurations)

- BF16, no quantization, eval mode, transformers backend, batch size 1.
- `do_sample=True, temperature=0.6, top_p=0.95, top_k=0,
  repetition_penalty=1.0` (the record stores the FULL merged parameter set,
  i.e. project values merged over model defaults).
- One generation per item per seed. Per-item seeds are derived from
  (run seed, item id) with a stable, order-independent formula - reordering
  the manifest never changes an item's seed.
- Seeds 0/1/2 for formal local runs - every seed is reported plus the mean;
  never best-of-N, never majority voting. Seed 0 for smoke/pilot/reference.
- Token budgets (safety budgets, not targets): MuSiQue 2048 new tokens;
  ProcessBench and whole-question tasks 4096. Identical across configs;
  8192 would be a separate budget condition and is not used.
- Context length is checked per item BEFORE generation (common eligibility
  rule); an item that does not fit is recorded as ineligible, never silently
  truncated.
- Raw token IDs and raw text are stored before any cleanup.
- Prompt rendering uses the vendored, sha256-pinned native chat template with
  transformers' exact jinja environment (trim/lstrip blocks,
  `keep_trailing_newline=False`, StrictUndefined, `tojson`), and passes
  `tools=None` / `documents=None` exactly as transformers'
  `render_with_template` does - the Qwen3 template's `{% if tools %}` branch
  depends on it.

## Parsing of reasoning models

- The reasoning region (`<think>...</think>` or equivalent) is parsed per
  reasoning mode (`prefilled_thinking`, `optional_thinking`,
  `thinking_disabled`, `no_thinking_channel` - see `configs/models.yaml`).
- An answer is never graded from inside the reasoning region.
- An OPEN reasoning region at end of generation = missing final answer
  (scored 0 with status `missing_final`, never guessed).
- DeepSeek-R1 prefill: the template prefills `<|Assistant|><think>\n`
  (chat-template detail, pinned); the closing `</think>` without an opening
  tag is handled explicitly.
- No hidden-reasoning token counts are invented; `Tout` counts all generated
  tokens (reasoning, answer and EOS) as actually produced.
- Token-limit termination (`finish_reason == length`) is an evaluated
  outcome: the truncated text is parsed and scored as-is.
- Timeout / OOM / crash = infrastructure failure: no partial output is
  scored, the item is retried at most 2 times (same item, same seed), and
  a wrong answer is NEVER retried. Unresolved failures leave the comparison
  explicitly incomplete (report lists them).

## Cost and efficiency

- `Tin` = input tokens after the native chat template is applied (including
  any prefilled prefix). `Tout` = all generated tokens (reasoning + answer +
  EOS).
- `Cproxy = 2 * N_text * (Tin + Tout)` per item. `N_text` = unique
  text-model parameters from the actual checkpoint (safetensors headers
  only; tied weights counted once: when `tie_word_embeddings=true` and the
  embedding and LM-head tensors have equal shapes they are one parameter;
  embeddings and LM head included; Gemma's vision tower and
  multi_modal_projector excluded). The auditable breakdown is saved by
  `weval inspect-params`.
- `Cweighted` = null without sourced prices; never zero, never invented.
- `Efficiency_compute = mean(score in [0,1]) / mean(Cproxy / 1e12)`,
  computed within the same task only.
- Positive and negative costs are reported separately. The 50:50
  positive/negative mix in the evidence-worker role comparison is a declared
  assumption for cost comparison only - not an estimate of deployment
  prevalence.

## Statistics

- Every metric is reported per seed and as the cross-seed mean.
- Paired config differences are computed on identical items.
- 95% bootstrap CIs are clustered: MuSiQue resamples whole parents (a
  positive and its paired negative, all variants and seeds, move together);
  ProcessBench macro metrics resample groups preserving the subset structure.
- Never best-of-N.

## Splits and leakage

- MuSiQue: train -> debug pool (pilot), dev -> selection (first 100 eligible
  parents) + holdout (next 100), disjoint by construction and verified.
- ProcessBench: project-defined selection/holdout by whole problem groups
  (above), pilot from groups excluded from selection.
- Whole-question references: frozen 100-item subsets per dataset (seed 0,
  fixed ordering); problems used in local debug/selection are excluded and
  the exclusions are reported.
- Cross-dataset check: exact matches under normalization (digits and
  operators preserved; whitespace adjacent to a math operator is
  insignificant, so `"x + 2 = 5"` and `"x+2=5"` match, while other
  whitespace runs collapse to one space and never merge tokens) and
  near-duplicates (Jaccard >= 0.95 token overlap) are flagged between all
  datasets.
- The dedup/leakage checks are hygiene, NOT a no-contamination claim.

## Selection rules (in config, user-confirmed before selection)

- Evidence worker: answerable F1 >= 0.80 AND false-abstention <= 0.10 AND
  reviewed negative correct-abstention >= 0.90.
- Math checker: ProcessBench macro-F1 >= 0.60 AND macro correct-solution
  acc >= 0.50 AND macro error-solution acc >= 0.50.
- Among qualified configs: lowest mean Cproxy on the role workload; the
  quality-cost Pareto frontier is reported; no winner is forced (borderline =
  provisional).
- The holdout is evaluated only for the selection-nominated candidate and may
  never be used to re-pick.
- Thresholds live in `configs/thresholds.yaml`; `draft_unconfirmed` values
  gate formal selection - the user confirming the file IS the gate.
