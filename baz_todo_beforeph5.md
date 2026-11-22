# TODO Before Phase 5

Use this checklist to close the remaining gaps identified in the project-plan review before starting the full evaluation and ablation phase. Checked items have supporting files or observed run evidence; unchecked items remain incomplete.

**Current status:** The RUHSOLD Task 1 SVM baseline is complete. A Gemini API smoke response succeeded, but classification inference hit the free-tier RPM limit and no LLM evaluation predictions or metrics were saved. CSAP remains planned work. The Task 2 validation split is a duplicate of Task 2 training and is excluded.

## Required Implementation Work

- [ ] Add an Akhter et al. / RUTUBE data loader alongside the existing RUHSOLD loader.
- [x] Confirm the local RUHSOLD source, Task 1/Task 2 TSV schemas, labels, and supplied split files (Task 2 validation duplication noted).
- [ ] Add dataset-quality checks for missing values, duplicates, class balance, and script distribution.
- [ ] Produce auditable train/validation/test artifacts with documented split policy and random seed.
- [ ] Add dataset summary output with row counts, label counts, text-column names, and preprocessing decisions.
- [x] Run the Task 1 SVM baseline and save test predictions, classification report, confusion matrix, and run metadata under `results/`.

- [ ] Add LLM experiment logging for model, endpoint/provider, temperature, token limits, prompt version, prompt length, latency, request status, and failures.
- [ ] Log malformed or ambiguous LLM responses and apply a pre-declared evaluation policy to them.
- [ ] Add response parsing compatible with the paper's `Abusive` and `Not abusive` labels, checking `Not abusive` before `Abusive`.
- [ ] Document the mapping between the paper labels and the project's binary labels if the internal labels remain `offensive` and `normal`.
- [ ] Add bounded response caching or resumable output handling so interrupted API runs do not repeat unnecessary requests.
- [ ] Add a focused test suite for preprocessing, SVM evaluation, prompt construction, response parsing, retry behavior, and artifact writing.

## Data Preparation

- [x] Obtain local RUHSOLD coarse-grained Task 1 train/validation/test files under `data/2020paper/roman_urdu_hate_speech-main/`.
- [ ] Obtain the Akhter et al. source data or the RUTUBE sample used for the comparison.
- [ ] Prepare the approximately 300-example custom spelling-variation test set.
- [ ] Record the custom-set construction rules, label policy, English mix-in policy, and quality-control process.
- [ ] Check for exact and near-duplicate leakage between demonstrations, training data, and evaluation data.
- [ ] Freeze the few-shot examples and alignment dictionaries before the full run.
- [ ] Verify that the N=0, N=5, and N=15 alignment settings use the intended mappings.

## API and Security Setup

- [x] Select Gemini API; `gemini-3.1-flash-lite` is configured in the local `.env` and returned a successful one-word smoke response.
- [ ] Create and test the selected provider credentials locally.
- [x] Store API credentials locally in `.env`; `.gitignore` excludes `.env`. Never print or commit the key.
- [ ] Add a `.gitignore` entry for `.env`, `.venv/`, Python caches, downloaded data, and generated outputs where appropriate.
- [ ] Confirm that no API key appears in source code, notebooks, logs, or committed files.
- [ ] Run one small real API smoke test and record provider, model, quota limits, and response format.
- [ ] Confirm the expected free-tier rate limits and set a conservative request delay if needed.

## Experiment Readiness

- [ ] Decide the exact LLM model and provider for the primary experiment.
- [ ] Fix temperature, maximum output tokens, timeout, retry count, and random-seed policy.
- [ ] Confirm the primary metric is Macro-F1 and define how failed or malformed responses count.
- [ ] Confirm the evaluation domains: clean Urdu, standard Roman Urdu, and custom spelling-variation Roman Urdu.
- [ ] Confirm that latency and prompt length will be recorded for every prompting condition.
- [x] Create `results/` outputs with descriptive RUHSOLD Task 1 baseline filenames.
- [ ] Make sure the full experiment can be resumed after an API interruption.

## Final Gate Before Phase 5

- [ ] All required datasets are available and schema checks pass.
- [x] Task 1 SVM baseline runs and writes metrics and predictions; see `results/ruhsold_task1_svm_metrics.json` and `results/ruhsold_task1_svm_predictions.csv`.
- [ ] Zero-shot and standard few-shot prompts complete successfully on a small sample.
- [ ] Cross-Script Alignment Prompting completes successfully for N=0, N=5, and N=15.
- [ ] The selected API's responses are parsed consistently into the two project labels.
- [ ] A dry run produces metrics, latency records, prompts/configuration metadata, and failure logs.
- [ ] The project is ready for the complete Phase 5 evaluation and ablation run.
