# TODO Before Phase 5

Use this checklist to close the remaining gaps identified in the project-plan review before starting the full evaluation and ablation phase.

## Required Implementation Work

- [ ] Add an Akhter et al. / RUTUBE data loader alongside the existing RUHSOLD loader.
- [ ] Confirm the actual RUHSOLD dataset identifier, configuration, schema, and available splits.
- [ ] Add dataset-quality checks for missing values, duplicates, class balance, and script distribution.
- [ ] Produce auditable train/validation/test artifacts with documented split policy and random seed.
- [ ] Add dataset summary output with row counts, label counts, text-column names, and preprocessing decisions.
- [ ] Extend the SVM workflow to save predictions, classification reports, confusion matrices, and configuration metadata.
- [ ] Add LLM experiment logging for model, endpoint/provider, temperature, token limits, prompt version, prompt length, latency, request status, and failures.
- [ ] Log malformed or ambiguous LLM responses and apply a pre-declared evaluation policy to them.
- [ ] Add response parsing compatible with the paper's `Abusive` and `Not abusive` labels, checking `Not abusive` before `Abusive`.
- [ ] Document the mapping between the paper labels and the project's binary labels if the internal labels remain `offensive` and `normal`.
- [ ] Add bounded response caching or resumable output handling so interrupted API runs do not repeat unnecessary requests.
- [ ] Add a focused test suite for preprocessing, SVM evaluation, prompt construction, response parsing, retry behavior, and artifact writing.

## Data Preparation

- [ ] Obtain or download the RUHSOLD coarse-grained dataset.
- [ ] Obtain the Akhter et al. source data or the RUTUBE sample used for the comparison.
- [ ] Prepare the approximately 300-example custom spelling-variation test set.
- [ ] Record the custom-set construction rules, label policy, English mix-in policy, and quality-control process.
- [ ] Check for exact and near-duplicate leakage between demonstrations, training data, and evaluation data.
- [ ] Freeze the few-shot examples and alignment dictionaries before the full run.
- [ ] Verify that the N=0, N=5, and N=15 alignment settings use the intended mappings.

## API and Security Setup

- [ ] Choose the free provider: Gemini or Groq.
- [ ] Create and test the selected provider credentials locally.
- [ ] Store credentials only in a local `.env` file or environment variables.
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
- [ ] Confirm output locations under `results/` and the naming convention for each run.
- [ ] Make sure the full experiment can be resumed after an API interruption.

## Final Gate Before Phase 5

- [ ] All required datasets are available and schema checks pass.
- [ ] The SVM baseline runs on the selected train/test split and writes its artifacts.
- [ ] Zero-shot and standard few-shot prompts complete successfully on a small sample.
- [ ] Cross-Script Alignment Prompting completes successfully for N=0, N=5, and N=15.
- [ ] The selected API's responses are parsed consistently into the two project labels.
- [ ] A dry run produces metrics, latency records, prompts/configuration metadata, and failure logs.
- [ ] The project is ready for the complete Phase 5 evaluation and ablation run.
