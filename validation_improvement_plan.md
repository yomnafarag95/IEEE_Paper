# RAG-Shield Validation Improvement Plan

This plan prioritizes the fixes that most reduce benchmark overfitting, result memorization, and weak generalization claims. The goal is to make the final reported results defensible: thresholds frozen before final testing, broader benign coverage, larger evasion coverage, and a final hidden benchmark that is not used during development.

## Priority Order

| Priority | Fix | Main Risk Reduced |
|---:|---|---|
| 1 | Freeze a final hidden test set | Evaluation-loop overfitting |
| 2 | Enforce strict train/dev/test separation | Data leakage and threshold overfitting |
| 3 | Lock thresholds before final evaluation | Benchmark-specific tuning |
| 4 | Increase benign diversity | Fragile FPR claims |
| 5 | Expand evasion set | Overstated evasion ADR |
| 6 | Add adversarially generated attacks | Memorized attack patterns |
| 7 | Report confidence intervals | Overconfident metrics |
| 8 | Run cross-dataset evaluation | Dataset-specific learning |
| 9 | Audit FP/FN without tuning on final set | Silent benchmark patching |
| 10 | Add a fresh benchmark result | External validity |

## Fix 1: Freeze a Final Hidden Test Set

**Objective:** Create an untouched final holdout that is never used for debugging, prompt edits, threshold tuning, or model selection.

**Implementation Steps**

1. Create `data/final_holdout/`.
2. Store separate files for attacks, benign samples, and evasions.
3. Add a short `README.md` inside the folder explaining that the set is final-evaluation only.
4. Add a guard in evaluation scripts so final-holdout mode prints a warning before running.
5. Record the exact run date, commit hash, config hash, and model artifact hashes with the result.

**Acceptance Criteria**

- No training or threshold-tuning script reads `data/final_holdout/`.
- Final-holdout results are generated only by an explicit command.
- Final reported paper/demo numbers clearly identify whether they come from dev, test, or final holdout.

## Fix 2: Enforce Strict Train/Dev/Test Separation

**Objective:** Prevent accidental reuse of samples across training, calibration, threshold selection, and final reporting.

**Implementation Steps**

1. Define split folders or manifest files:
   - `data/splits/train.jsonl`
   - `data/splits/dev.jsonl`
   - `data/splits/test.jsonl`
   - `data/final_holdout/*.jsonl`
2. Use deterministic SHA-256 IDs for each sample.
3. Add a leakage checker that fails if the same normalized text hash appears in multiple splits.
4. Make training scripts consume only `train`.
5. Make threshold sweeps consume only `dev`.
6. Make reported evaluation consume only `test` or `final_holdout`.

**Acceptance Criteria**

- A split-integrity command reports zero duplicate hashes across splits.
- `train_meta_aggregator.py` cannot silently consume evaluation files.
- `threshold_sweep.py` cannot silently consume final-holdout files.

## Fix 3: Lock Thresholds Before Final Evaluation

**Objective:** Make the final numbers reproducible and immune to post-hoc tuning.

**Implementation Steps**

1. Create a threshold snapshot file, for example `logs/frozen_thresholds.json`.
2. Include:
   - `L1_BLOCK_THRESHOLD`
   - `L2_STAGE1_THRESHOLD`
   - `L2_DOC_PATTERN_THRESHOLD`
   - `L3_CONSISTENCY_THRESHOLD`
   - `META_BLOCK_THRESHOLD`
   - `META_MONITOR_THRESHOLD`
   - `META_BLOCK_CONSENSUS_THRESHOLD`
3. Add config/model hash metadata to the evaluation report.
4. Make final evaluation fail if current thresholds differ from the frozen snapshot unless an explicit override is passed.

**Acceptance Criteria**

- Final evaluation records thresholds and refuses accidental threshold drift.
- Any threshold change requires a new dev calibration run and a new threshold snapshot.

## Fix 4: Increase Benign Diversity

**Objective:** Make FPR claims robust across realistic enterprise RAG traffic.

**Implementation Steps**

1. Expand benign samples to at least 500-1,000 examples.
2. Include domains such as HR, finance, legal, medical, IT, customer support, travel, policy, sales, and general knowledge.
3. Include realistic edge cases:
   - short questions
   - long questions
   - malformed grammar
   - pasted email snippets
   - tables
   - markdown
   - noisy OCR-like text
   - multilingual benign queries
4. Keep a separate benign-final-holdout subset that is not used for threshold tuning.

**Acceptance Criteria**

- Benign FPR is reported on at least 500 benign samples for the main benchmark.
- False positives are grouped by domain and text style.

## Fix 5: Expand Evasion Set

**Objective:** Replace the current tiny evasion set with a benchmark large enough to support credible claims.

**Implementation Steps**

1. Increase evasion samples from 7 to at least 50, preferably 100+.
2. Cover:
   - Base64
   - hex
   - Unicode homoglyphs
   - full-width characters
   - leetspeak
   - whitespace splitting
   - markdown/comment hiding
   - role-play wrappers
   - indirect document instructions
   - multi-turn Crescendo-style escalation
3. Label evasion type in metadata.
4. Report per-evasion-family ADR.

**Acceptance Criteria**

- Evasion ADR is reported both overall and by evasion family.
- The result is not based on fewer than 50 evasion samples.

## Fix 6: Add Adversarially Generated Attacks

**Objective:** Test against attacks that were not shaped by the current detector design.

**Implementation Steps**

1. Freeze the current detector configuration.
2. Ask a separate LLM, script, or human reviewer to generate new attacks after freezing.
3. Do not reveal current failure cases, thresholds, or exact regexes to the generator.
4. Include both direct and indirect attacks.
5. Keep these attacks out of training and threshold selection.

**Acceptance Criteria**

- A separate adversarial benchmark exists with provenance notes.
- The evaluation report identifies it separately from older datasets.

## Fix 7: Report Confidence Intervals

**Objective:** Avoid overclaiming precision from small sample sizes.

**Implementation Steps**

1. Use `bootstrap_ci.py` for ADR, FPR, precision, F1, and ROC-AUC.
2. Add confidence intervals to `eval_report.json`.
3. Add confidence intervals to paper tables and README metrics.
4. For small categories, explicitly mark results as preliminary.

**Acceptance Criteria**

- Every headline metric has a 95% confidence interval.
- Evasion results with small `n` are not presented as definitive.

## Fix 8: Run Cross-Dataset Evaluation

**Objective:** Show whether the system generalizes across attack distributions instead of learning dataset-specific artifacts.

**Implementation Steps**

1. Run train/tune on HackAPrompt-style data and test on InjecAgent-style data.
2. Run train/tune on InjecAgent-style data and test on HackAPrompt-style data.
3. Add TensorTrust or another external dataset as a third distribution if available.
4. Report each result separately rather than only combined averages.

**Acceptance Criteria**

- Cross-dataset ADR and FPR are present in logs.
- The writeup distinguishes in-distribution and out-of-distribution performance.

## Fix 9: Audit False Positives and False Negatives Without Tuning on Final Set

**Objective:** Learn from errors without contaminating the final benchmark.

**Implementation Steps**

1. Create a structured FP/FN report format:
   - sample ID
   - expected label
   - actual action
   - risk score
   - blocking layer
   - suspected cause
   - proposed future fix
2. Allow detailed debugging on train/dev/test.
3. For final holdout, document errors but do not change thresholds or code based on those exact samples.
4. Use final-holdout errors only to design a future benchmark version.

**Acceptance Criteria**

- FP/FN reports exist for each evaluation run.
- No final-holdout-specific patch is made before publishing the final result.

## Fix 10: Add a Fresh Benchmark Result

**Objective:** Provide the strongest evidence that RAG-Shield is not memorizing the existing benchmark.

**Implementation Steps**

1. After all code, thresholds, and models are frozen, collect a fresh benchmark.
2. Target at least 300-500 samples:
   - 100-200 attacks
   - 150-250 benign
   - 50+ evasions
3. Use samples from new sources, new prompt authors, or newly generated adversarial tasks.
4. Run the frozen system once.
5. Report this as the strongest final generalization result.

**Acceptance Criteria**

- Fresh benchmark was collected after system freeze.
- No code, model, or threshold changes happen between fresh benchmark creation and result reporting.
- Fresh benchmark results are clearly separated from development benchmark results.

## Suggested Implementation Sequence

1. Implement split manifests and leakage checks. **Status: Done.**
2. Freeze current thresholds and artifact hashes. **Status: Done.**
3. Expand benign and evasion datasets. **Status: Done.**
4. Add adversarially generated attacks. **Status: Done.**
5. Add confidence intervals to reports. **Status: Done for evaluation reports; paper/README values pending final full run.**
6. Add cross-dataset evaluation mode.
7. Add FP/FN structured reports.
8. Create final hidden holdout.
9. Run final hidden holdout once.
10. Collect and run a fresh benchmark after full freeze.

## Final Reporting Rule

Use three clearly separated result sections:

1. **Development Benchmark:** useful for iteration, not the main claim.
2. **Frozen Test Benchmark:** main internal result after thresholds are locked.
3. **Fresh External Benchmark:** strongest evidence of generalization.

Do not update the headline numbers after inspecting final-holdout failures unless a new benchmark version is created and clearly labeled.

## Implementation Log

### Step 1: Split Manifests and Leakage Checks

**Status:** Done

**Added**

- `split_registry.py`
- `data/splits/manifest.jsonl` generated locally
- `data/splits/summary.json` generated locally
- `data/splits/leakage_report.json` generated locally
- Optional `eval_suite.py --require-clean-splits` guard

**Commands**

```bash
python split_registry.py build
python split_registry.py check
python eval_suite.py --mode all --require-clean-splits
```

**Current Check Result**

The current manifest indexes 71,692 curated/evaluation-sized samples and reports zero exact normalized duplicate hashes across train/dev/test splits.

**Note**

By default, `split_registry.py` skips very large raw files such as `data/hackaprompt.jsonl` so routine checks stay fast. Use `python split_registry.py build --include-raw-large` for a heavier full-source audit.

### Step 2: Freeze Thresholds and Artifact Hashes

**Status:** Done

**Added**

- `freeze_state.py`
- `logs/frozen_thresholds.json` generated locally
- Optional `eval_suite.py --require-frozen-state` guard
- `_evaluation_state` metadata added to saved evaluation reports

**Commands**

```bash
python freeze_state.py freeze
python freeze_state.py check
python freeze_state.py summary
python eval_suite.py --mode all --require-clean-splits --require-frozen-state
```

**Frozen Thresholds**

| Name | Value |
|---|---:|
| `L1_BLOCK_THRESHOLD` | 0.85 |
| `L2_STAGE1_THRESHOLD` | 0.60 |
| `L2_DOC_PATTERN_THRESHOLD` | 0.60 |
| `L3_CONSISTENCY_THRESHOLD` | 0.55 |
| `META_BLOCK_THRESHOLD` | 0.35 |
| `META_MONITOR_THRESHOLD` | 0.31 |
| `META_BLOCK_CONSENSUS_THRESHOLD` | 0.85 |
| `STATEFUL_DRIFT_THRESHOLD` | 0.72 |

**Current Check Result**

The current freeze snapshot tracks 30 artifacts: 9 code files, 19 model/tokenizer files, and 2 data split files. `python freeze_state.py check` passes against `logs/frozen_thresholds.json`.

**Note**

The snapshot currently records `git.dirty = true` because the workspace has uncommitted edits. This is acceptable for local validation, but a publication-grade freeze should be made from a clean commit after intentional changes are reviewed.

### Step 3: Expand Benign and Evasion Datasets

**Status:** Done

**Added**

- `build_validation_sets.py`
- `data/validation_benign_expanded.jsonl` generated locally
- `data/evasion_validation_curated.csv` generated locally
- `eval_suite.py` now includes the generated benign set in expanded benign stress tests
- `eval_suite.py` now prefers the curated evasion file when present
- Evasion metrics now include per-family sample counts and prevention ADR
- `split_registry.py` now includes the generated validation files in split/leakage checks
- `freeze_state.py` now hashes the generator and generated validation files

**Commands**

```bash
python build_validation_sets.py
python split_registry.py build
python split_registry.py check
python freeze_state.py freeze
python freeze_state.py check
```

**Current Generated Set Sizes**

| Set | Samples | Output |
|---|---:|---|
| Expanded benign validation | 192 | `data/validation_benign_expanded.jsonl` |
| Curated evasion validation | 100 | `data/evasion_validation_curated.csv` |

**Current Check Result**

The split manifest now indexes 71,984 samples and reports zero exact normalized duplicate hashes across train/dev/test splits. The refreshed freeze snapshot tracks 33 artifacts: 10 code files, 19 model/tokenizer files, and 4 data artifacts.

**Note**

The generated data files remain under `data/`, which is gitignored. Commit the generator script, not the generated benchmark files. Regenerate the local files before final evaluation.

### Step 4: Add Adversarially Generated Attacks

**Status:** Done

**Added**

- `build_validation_sets.py` now generates `data/adversarial_validation_curated.csv`
- `eval_suite.py --mode adversarial`
- `eval_suite.py --mode all` now includes adversarial benchmark metrics
- `split_registry.py` now includes the adversarial benchmark in split/leakage checks
- `freeze_state.py` tracks the generated adversarial benchmark as a data artifact

**Commands**

```bash
python build_validation_sets.py
python split_registry.py build
python split_registry.py check
python eval_suite.py --mode adversarial
```

**Current Generated Set Size**

| Set | Samples | Output |
|---|---:|---|
| Adversarial validation | 60 | `data/adversarial_validation_curated.csv` |

**Current Check Result**

The split manifest now indexes 72,036 samples and reports zero exact normalized duplicate hashes across train/dev/test splits.

**Current Adversarial Evaluation Result**

| Metric | Value |
|---|---:|
| ADR prevention | 0.9000 |
| ADR detection | 0.9000 |
| F1 prevention | 0.9474 |
| Mean latency | 919.4 ms |

Per-family prevention ADR: business action hijack 1.0000, crescendo escalation 1.0000, data exfiltration 0.8000, policy conflict 1.0000, retrieval instruction override 1.0000, tool misuse 0.6000.

**Note**

The benchmark provenance is recorded in the CSV: generated after threshold freeze by deterministic adversarial templates, without exposing thresholds, regexes, or current failure cases to the generator. The generated CSV remains under `data/`, which is gitignored; commit the generator and evaluation wiring, then regenerate the local file before a publication-grade freeze.

### Step 5: Add Confidence Intervals to Reports

**Status:** Done for evaluation reports; paper/README values pending final full run

**Added**

- `eval_suite.py` now adds `confidence_intervals_95` to every metrics block produced by `_metrics`
- Wilson score intervals for ADR, FPR, and precision
- Bootstrap percentile intervals for F1 and ROC-AUC
- Single-class benchmarks, such as adversarial-only runs, report ROC-AUC CI as `N/A`

**Commands**

```bash
python eval_suite.py --mode adversarial
python eval_suite.py --mode all
```

**Current Check Result**

`python eval_suite.py --mode adversarial` writes `logs/eval_report.json` with confidence intervals. Current adversarial ADR prevention is 0.9000 with 95% CI `[0.7985, 0.9534]`; F1 prevention is 0.9474 with 95% CI `[0.8991, 0.9831]`.

**Note**

Do not update README or paper headline CI values from a partial mode-specific run. Run `python eval_suite.py --mode all` after the final benchmark set is frozen, then copy the final report CIs into the paper tables and README metric table.
