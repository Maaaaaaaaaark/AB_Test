# Vendored official evaluation code

Every file under `vendor/` is copied **verbatim** from the official repository
at a pinned commit. Nothing is modified (the only addition is the empty
`metrics/__init__.py` so the vendored MuSiQue package imports unchanged).
Each file's SHA-256 (computed 2026-10-13, listed below) is recorded; the test
suite verifies the copies are byte-identical at install time.

| File | Source | Commit | SHA-256 | Used for |
|---|---|---|---|---|
| `musique/metrics/answer.py` | https://github.com/StonyBrookNLP/musique `metrics/answer.py` | `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | `10368F619B4D5EF5D83748C05A96C0AFD332A14AB5C010740C98D58DFAEFE974` | Official MuSiQue answer EM/F1 (normalize / compute_exact / compute_f1 / max-over-golds) |
| `musique/metrics/metric.py` | https://github.com/StonyBrookNLP/musique `metrics/metric.py` | `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | `C858D1BFDA2F0B005065E87A402CD2F82154EB7EED5916845AE130759CC3A299` | Base class imported by `answer.py` |
| `musique/evaluate_v1.0.py` | https://github.com/StonyBrookNLP/musique `evaluate_v1.0.py` | `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | `F5FE66AE61DBEA5172CBA9D428D9924A5811F3457EDC945FC1D81369C30E74B7` | Reference: official eval orchestration (we use only the answer metric; support/sufficiency metrics are not applicable to our local task) |
| `processbench/run_eval.py` | https://github.com/QwenLM/ProcessBench `code/run_eval.py` | `e8024636bcabdf8bd514440551b531d3f90dd18b` | `66D09FC7A3D20D46F166D4BBA4E04835F3D7473E1534B6908CD9045AEFC7B704` | Reference: official prompt, extraction (`\boxed{}` -> int), and scoring (error acc, correct acc, F1 per subset) |
| `processbench/critique_template.txt` | https://github.com/QwenLM/ProcessBench `code/templates/critique_template.txt` | `e8024636bcabdf8bd514440551b531d3f90dd18b` | `0E7DBA24BFAA9DEA379907E11B8DC701C6CD6E74453AB2FFC9B1D63237CC4434` | Official ProcessBench task prompt (used verbatim) |
| `hotpotqa/hotpot_evaluate_v1.py` | https://github.com/hotpotqa/hotpot `hotpot_evaluate_v1.py` | `3635853403a8735609ee997664e1528f4480762a` | `D35FC91A6DB21D791DBDDA11DAF3856E9359F5701D54E3EEFBA20D88FECC02C0` | Official HotpotQA normalization, answer EM/F1. We use the answer part only; supporting-facts evaluation is not applicable (no supporting-fact predictions in this project) - documented intentional difference |
| `gsm8k/grade_school_math_dataset.py` | https://github.com/openai/grade-school-math `grade_school_math/dataset.py` | `3101c7d5072418e28b9008a6636bde82a006892c` | `C22B81BCCAAEBFD1123BE296E8B3C3DCA3DDC704C46B70BF4A50603BB627E197` | Official GSM8K extractor (`ANS_RE`, `INVALID_ANS`, `is_correct`) |
| `math/math_equivalence.py` | https://github.com/hendrycks/math `modeling/math_equivalence.py` | `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | `C4101B1F51A2BB65665194AECD9F761D668C0C248A0350F196C2950E87488527` | Official MATH answer equivalence checker (adapted scoring: we never treat two missing answers as a match - guarded in our wrapper) |

## Intentional differences from the official evaluation protocols

All differences are recorded in `docs/PROTOCOL.md`. Summary:

1. **MuSiQue**: we run a custom *local* task (one subquestion, 5 visible
   paragraphs, prerequisite answers resolved) scored with the official answer
   EM/F1 only. The official support-F1, group-answer-sufficiency and
   group-support-sufficiency metrics are not computed (they are defined for the
   full benchmark, not for our local task).
2. **ProcessBench**: same prompt, extraction and scoring as `run_eval.py`.
   Differences: 5 model configurations, our sample sizes (5/25/25 per subset),
   temperature 0.6 (official: 0.0), three independent single-response seeds
   (official: 1 response; voting mode is not used), Transformers backend
   (official: vLLM), max 4096 new tokens (official: 8192). This is a repeated
   single-response evaluation, NOT a reproduction of the paper's full
   protocol.
3. **HotpotQA**: answer EM/F1 with the official normalization/functions.
   Supporting-facts / joint metrics are not computed because this project does
   not predict supporting facts (documented adapted score).
4. **GSM8K**: official `ANS_RE`/`is_correct` logic is the primary scorer. The
   prompt asks for `#### <number>` (our controlled instruction). A relaxed
   last-number parser exists only as a secondary diagnostic; it never replaces
   the primary score. Extraction failures are reported separately.
5. **MATH**: vendored `math_equivalence.py` is the equivalence checker.
   Answer extraction (`\boxed{}`) is a separate step; two missing answers are
   never treated as a match. The checker is an equivalence heuristic, not a
   full symbolic solver (we do not claim otherwise).
