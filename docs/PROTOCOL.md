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
   insufficient pool are excluded and counted. The four distractors are a
   **seeded sample** of that pool and the five paragraphs are shown in a
   **seeded random order** (`split_seeds.musique_paragraph_order`, per item,
   recorded in `gold.paragraph_plan`). Builder v1 took the four lowest-idx
   distractors in idx order, which put the support paragraph in the last slot
   almost always; the support-slot distribution is now reported per build.
4. *Abstention as a first-class label:* the instruction requires the model to
   answer or output exactly `NOT FOUND`. A positive item answered with
   NOT FOUND is a false abstention (scored 0, flagged); a negative item
   answered with NOT FOUND scores 1. The answer is extracted with the
   shared answer-line rule (see "Short-answer extraction" below).
   *NOT FOUND matching (revised after the pilot, frozen before selection):*
   the extracted answer is lower-cased, `_`/`-` become spaces, surrounding
   whitespace, quotes, brackets, markdown and punctuation are stripped, and
   the result must equal exactly `not found`. In the pilot, Qwen3 wrote
   `NOT_FOUND` on 11 of 14 negative replies and the exact-match rule scored
   them as answers (abstention 0/7 and 1/7 instead of 6/7 and 6/7). There is
   still no substring or synonym matching: free-text refusals ("The provided
   information is insufficient ...") remain answer attempts. Variants are
   recorded in `diagnostics.not_found_variant`.
5. *Gold for positives:* the subquestion's canonical answer, plus the parent
   answer and parent aliases at the terminal step (aliases are used only when
   the canonical subanswer equals the parent answer under official
   normalization; the alias source is recorded per item).
6. *Paired negatives (project-constructed):* the supporting paragraph is
   replaced by a 5th (seeded) non-supporting paragraph at the same slot; the
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

7. *Positive screening and review (project-defined).* A subquestion is
   eligible only if one of its gold answers appears literally in its support
   paragraph (title or text). This is necessary, not sufficient: the first
   smoke run showed positives whose support paragraph mentions the answer
   string without stating the asked relation, and positives whose paragraph
   never names the entity in the question. Every positive is therefore
   human-reviewed ("does the support paragraph, read on its own, answer the
   resolved question with the gold answer?"); pilot, selection and holdout
   run **approved positives only** (`require_approved_positives`). Without
   this, false-abstention would penalize models that stay faithful to the
   paragraphs and reward answering from memory.
8. *Review decisions are bound to content.* Each decision (positive and
   negative separately) stores a fingerprint of the item it was made on
   (question, gold, visible paragraphs in order). A rebuild that changes the
   item drops the decision into `stale_decisions` and the item returns to
   `proposed`.

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
1. *Instruction* (custom) with **one fixed worked example** in the official
   `#### <number>` format (GSM8K main train, first record, calculator
   annotations removed; any identical test question is excluded at build).
   In the first smoke run the zero-shot instruction produced 0% official
   accuracy for every configuration because no model wrote `####`; the
   example demonstrates the format the official extractor requires.
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
5. *Gold extraction fix (adapter math-adapter-v3, 2026-10-15).* Up to v2 the
   gold answer was taken with the regex `\\boxed\{([^}]*)\}`, which stops at
   the first `}`: `\boxed{\frac{1}{2}}` gave the gold `\frac{1`, so every
   correct fraction / root / exponent answer was scored wrong. v3 takes the
   LAST `\boxed{...}` / `\fbox{...}` with nested braces matched, as the
   official harness does (`last_boxed_only_string` + `remove_boxed`), and
   accepts the `\boxed 5` form. Item selection does not depend on the gold,
   so no manifest changes membership; reports re-score from the manifests, so
   re-running `weval build` + `weval report` corrects earlier MATH numbers
   (pilot and reference) without new generations.
6. *Loose equivalence (diagnostic only, `math-loose-v1`).* The official
   checker compares normalized strings, so `0.25` vs `\frac{1}{4}` or
   `2\pi` vs `\pi\cdot 2` count as wrong. A looser second opinion (wider
   normalization, numeric evaluation of simple expressions, element-wise
   tuples, choice letters) is stored per record and reported as
   `loose_accuracy_diagnostic` with the disagreeing cases listed for spot
   checks. It never changes the score.

## Single-model math solving (plan `math_solve`, descriptive)

**Purpose.** How well each configuration solves math problems ON ITS OWN:
one model, one problem, no other agent and no checker. It complements the
ProcessBench checking task (judging someone else's solution) and is fully
decoupled from worker selection: the plan nominates nothing, has no
thresholds and no holdout, and its results never feed the selection rules.

**Data (frozen by `weval build`, before any output is seen):**
- GSM8K: official main test split (1,319): 200 problems, simple random.
- MATH: original test split (5,000): stratified by level, 40 problems from
  each of Level 1-5 (200). Records with another level value are excluded and
  counted. Subject counts are recorded in the manifest and the report.
- Exclusions (exact after `normalize_for_matching`, or near-duplicate with
  token Jaccard >= 0.95 - here the near-duplicate flag DOES exclude): every
  ProcessBench problem (all subsets, all records), every problem of the
  GSM8K / MATH reference subsets (they contain the pilot items), and the
  GSM8K prompt's worked example. Candidates that duplicate an already drawn
  problem are skipped too. Every skipped candidate is listed with its reason
  in `artifacts/manifests/math_solve_exclusions.json`.
- Item ids are `test-<index in the pinned split>`; sampler
  `math-solve-sampler-v1`, seed `split_seeds.math_solve_order`.
- The manifest records both the raw 5,000-question level counts and the level
  counts after the external overlap exclusions above. Headline weights use the
  latter because excluded questions have zero probability of entering this
  experiment.

**Run:** the 4 candidate configurations, seeds 0,1,2, the same prompts and
budgets as everywhere else (GSM8K 4096, MATH 8192 new tokens), official
scoring. Runs can be split per configuration (`--configs`), records land in
one store, the report covers the whole plan.

**Report:**
- GSM8K accuracy (official), extraction failures, missing final, length stops.
- MATH accuracy, headline = **level-weighted**: the per-level accuracies
  weighted by the eligible post-exclusion pool's level proportions (the
  equal-allocation sample over-represents the easy levels). The plain sample mean is shown next to
  it. Per-level accuracy, subject counts, answer failures, length stops, and
  the loose-equivalence diagnostic with examples.
- CIs: problems are the clusters (all seeds of a problem move together);
  the level-weighted CI resamples problems within each level (stratified).
- Paired config differences on identical problems; for MATH both the plain
  and the level-weighted difference.
- Mean output tokens per configuration and task.

## HotpotQA (whole-question reference)

**Official:** distractor dev split, full question + all 10 paragraphs;
answer normalization and token F1 vendored from the official evaluation.

**Project-defined:** answer-only scope - supporting-facts prediction and the
joint EM/F1 are not part of this project, so only the answer metrics are
computed (the vendored functions are used verbatim for the parts that apply).
The short answer is extracted with the answer-line rule below (builder v1
scored the whole reply, so full-sentence answers diluted F1 and EM was 0 for
every configuration). The official normalization does not map digits to
number words ("5" vs "five" is a miss); kept as official.

## Short-answer extraction (MuSiQue local, HotpotQA)

The instruction asks for a short answer and, if anything precedes it, a last
line `Answer: <short answer>`; the prompt ends with an `Answer:` cue.
From the answer channel (after any reasoning region): (1) the LAST
`Answer: X` line (markdown decoration tolerated; empty X -> next non-empty
line); else (2) a single-line reply is the answer; else (3) invalid format:
no answer is extracted, the item scores 0 and is reported in the
answer-format counts (Answer-line / single-line / inline-marker / invalid /
missing).

*Inline marker (revised after the pilot, frozen before selection):* if the
extracted text still contains an `Answer:` marker after other text, the text
after the LAST marker is the answer when it is non-empty, otherwise the text
before it. In the pilot Gemma wrote `Belgium Answer:` and
`8 April 2013 Answer: 8 April 2013` (the prompt's trailing `Answer:` cue plus
the answer-line instruction); scoring the whole line gave EM 1/6 instead of
4/6. Such replies are counted as `inline_marker` (a format deviation that is
reported, not hidden).

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
  ProcessBench, GSM8K and HotpotQA 4096; MATH 8192 (raised from 4096 after
  the pilot, where Qwen3 thinking-on left 4/10 problems without a final
  answer at 4096; MATH is a reference task and never decides selection).
  Identical across configs within a task.
- Context length is checked per item BEFORE generation (common eligibility
  rule); an item that does not fit is recorded as ineligible, never silently
  truncated.
- Raw token IDs and raw text are stored before any cleanup.
- BOS: prompts are tokenized with `add_special_tokens=False`, so a template
  that renders `{{ bos_token }}` (Gemma 3, Llama 3.2, R1-Distill) gets the
  pinned BOS string (`template_kwargs.bos_token`, tokenizer fallback). The
  backend asserts that the first prompt token is exactly one BOS and aborts
  the run otherwise (`PromptContractError`, never retried, never scored).
  The first smoke run rendered Gemma and Llama prompts without BOS.
- Stop strings: trailing stop strings are trimmed before parsing. The set is
  tokenizer `eos_token` + the decoded `generation_config.eos_token_id`s + the
  declared `stop_strings` in `configs/models.yaml` (Gemma stops on
  `<end_of_turn>`, which is not its `eos_token`). Re-scoring merges the
  declared strings in, so old records are parsed correctly too.
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
  the exclusions are reported. The existing reference plan remains immutable
  and still contains HotpotQA, GSM8K and MATH. `math_solve` is a separate plan;
  adding it does not require rerunning reference.
- math_solve samples: see "Single-model math solving" above.
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
- Only the selection stage nominates (report mode `selection_nomination`).
  The holdout report gives pass/fail of the selection nominee of each role
  against the same thresholds (mode `holdout_check`); a failing nominee is
  reported as failing, never replaced. Smoke, pilot and reference reports
  show the candidate metrics next to the thresholds for information only and
  never nominate (mode `informational_no_nomination`), even with confirmed
  thresholds. (Fixed in report-v4: before it, a confirmed thresholds file
  made the pilot report print nominations.)
- Thresholds live in `configs/thresholds.yaml`; `draft_unconfirmed` values
  gate formal selection - the user confirming the file IS the gate.
  Confirmed on 2026-10-14, before any selection run, with the pre-pilot
  values unchanged.
- Candidates for selection/holdout (decided after the pilot, before
  selection): Qwen3-4B thinking off and on, Gemma-3-4B-it, Llama-3.2-3B.
  DeepSeek-R1-Distill-Qwen-1.5B was dropped: in the pilot it failed both
  roles (evidence F1 0.40, negative abstention 1/7; ProcessBench macro F1
  0.00 with 6/21 length stops). It remains in the descriptive reference plan.
