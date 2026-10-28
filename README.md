# Project Progress Log: Up Till Now

## Project

**Course:** CS 5316: NLP Theory and Applications  
**Category:** Category 2 - Re-implementation with a Novel Extension  
**Topic:** Cross-Script Offensive Language Detection for Roman Urdu and standard Urdu  
**Team:** Group 18  
**Status:** Phases 0 through 4 completed; Phase 5 not started

## Project Objective

The project re-implements prompt-based offensive-language detection for Urdu and Roman Urdu based on:

> Iffat Maab, Usman Haider, and Junichi Yamagishi. "Prompt-driven Detection of Offensive Urdu Language using Large Language Models." EACL 2026, pages 4302-4327.

The proposed extension is **Cross-Script Alignment Prompting**. The method injects a small Roman Urdu to standard Urdu alignment dictionary into the prompt so that the model can interpret spelling variants, abbreviations, and slang before classifying a comment.

## Phase 0: Planning and Documentation

### Completed

Created `project_plan.md` in the project root.

The plan documents:

- Motivation and the Roman Urdu/Urdu script gap.
- The central research question.
- Hypotheses.
- Re-implementation scope.
- TF-IDF SVM, zero-shot, and few-shot baselines.
- Cross-Script Alignment Prompting.
- N=0, N=5, and N=15 ablation levels.
- RUHSOLD, Akhter et al., and custom spelling-variation data.
- Macro-F1, precision, recall, latency, and prompt-size evaluation.
- Error-analysis taxonomy.
- Execution roadmap.
- Risks, controls, deliverables, and success criteria.

## Phase 1: Environment Setup

### Completed

Created `requirements.txt` containing:

- `datasets`
- `scikit-learn`
- `pandas`
- `urduhack`
- `requests`
- `python-dotenv`

Created the following directory structure:

```text
data/
    raw/
    processed/
configs/
notebooks/
results/
src/
    data/
    models/
```

The declared dependencies were installed in the configured Python 3.14 environment. No PyTorch or TensorFlow dependency was added.

## Phase 2: Data Pipeline

### Completed

Created [src/data/preprocess.py](src/data/preprocess.py).

The module provides:

- RUHSOLD loading through Hugging Face `datasets`.
- Default dataset name `roman_urdu_hate_speech`.
- Default configuration `Coarse_Grained`.
- Flexible text-column discovery for common names such as `text`, `tweet`, `comment`, and `sentence`.
- Flexible label-column discovery for common names such as `label`, `labels`, `class`, and `category`.
- URL removal.
- Mention removal.
- Repeated-character reduction.
- Whitespace normalization.
- Arabic Unicode variant normalization to Urdu characters.
- Binary label normalization:
  - Offensive = 1.
  - Normal = 0.
- Hugging Face `Dataset` and `DatasetDict` split preparation.

### Correction Found During Later Audit

The label normalizer originally rejected labels such as `non-offensive` because hyphens were converted to spaces before lookup. The accepted normalized forms were corrected to include `non offensive` and `non hate`.

The correction was validated with direct edge-case tests.

## Phase 3: Classical Baseline

### Completed

Created [src/models/baseline_svm.py](src/models/baseline_svm.py).

The module provides:

- Character-level `TfidfVectorizer`.
- Character n-gram range `(2, 5)`.
- `LinearSVC` classifier.
- Configurable SVM `C` value.
- Optional class weighting.
- Training function.
- Prediction function.
- Input-length, text-type, and binary-label validation.
- Structured scikit-learn classification report.
- Per-class precision, recall, and F1.
- Macro and weighted averages.

A small balanced training/evaluation smoke test passed with Macro-F1 output successfully produced.

## Phase 4: LLM Inference and Prompting

### Completed

Created [src/models/llm_prompter.py](src/models/llm_prompter.py).

The module provides:

- Zero-shot prompt generation.
- Four-example few-shot prompt generation.
- Cross-Script Alignment Prompting.
- Configurable dictionary size through `n_dictionary_examples`.
- Default alignment ablation support for N=0, N=5, and N=15.
- Roman Urdu to Urdu-script alignment examples.
- Environment-based configuration through `python-dotenv`.
- OpenAI-compatible request payloads, including services such as Groq.
- Gemini-style request payloads.
- Request timeouts.
- Bounded retries for rate-limit and server errors.
- Deterministic parsing of `normal` and `offensive` labels.
- Mockable HTTP sessions for testing without a real API request.

A mocked API smoke test passed, including:

- Prompt generation.
- N=5 alignment injection.
- Label parsing.
- OpenAI-compatible response extraction.

No real API request was made.

## Research Paper Verification

After the initial implementation, the authoritative paper and the authors' official repository were checked:

- Paper: https://aclanthology.org/2026.eacl-long.201/
- Official code: https://github.com/nii-yamagishilab/offensive-urdu-language-detection

Important details confirmed from the paper:

- RUHSOLD coarse-grained data contains 9,992 examples.
- The binary labels are Abusive and Not abusive.
- The binary few-shot setup uses four examples: two abusive and two non-abusive.
- RUHSOLD is evaluated with English, Roman Urdu, and Urdu prompts.
- RUTUBE and USADC are evaluated using English prompts.
- RUTUBE uses a balanced 10,000-example sample from the Akhter et al. corpus.
- The paper primarily evaluates local open-weight Llama, Qwen, and Lughaat models.
- GPT-4 is included as a proprietary comparison, not as the only possible route.
- The official repository provides local-model prompting code, prompt templates, label configurations, and incremental output handling.

## Validation Completed

The following checks have passed:

- Markdown diagnostics for the project plan and requirements file.
- Python syntax compilation for the Phase 2 module.
- Phase 2 preprocessing smoke test.
- Phase 3 SVM training and evaluation smoke test.
- Phase 3 required vectorizer configuration check.
- Phase 4 prompt and mocked API smoke test.
- Editor diagnostics for all implemented Python modules.
- Post-audit label-normalization edge-case test.
- Post-audit SVM regression smoke test.

## Current Project State

The core Phases 0 through 4 are implemented. Before Phase 5, one refinement should be made for a more faithful reproduction:

1. Configure and smoke-test a free Gemini or Groq API endpoint.
2. Align prompt wording and binary labels more closely with the paper: `Abusive` and `Not abusive`.
3. Improve response parsing to prioritize `Not abusive` before matching `Abusive`, as recommended by the paper's answer-cleansing procedure.
4. Verify the actual RUHSOLD dataset identifier and schema before a full run.
5. Obtain or prepare the Akhter/RUTUBE data and the custom 300-example spelling-variation set.

These refinements are prerequisites for a clean, reproducible Phase 5 evaluation. Phase 5 has not been started, and no full dataset experiment or real LLM inference has been run yet.

## Cost and Resource Plan

The project can be completed using free services without paying for API usage, subject to provider quotas:

- Dataset download and SVM experiments are free.
- Gemini or Groq free tiers can be used for LLM inference. They require API keys but do not necessarily require paid billing.
- Free-tier quotas, rate limits, model availability, and provider policies may change, so the selected provider and limits must be recorded in the experiment metadata.
- A local Ollama model remains an optional fallback if hosted quotas are insufficient.
- API keys must never be committed to the repository.

The current client supports OpenAI-compatible endpoints such as Groq and Gemini-style endpoints. Credentials should be supplied through environment variables or a local `.env` file:

- Groq/OpenAI-compatible: `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`
- Gemini: `GEMINI_ENDPOINT`, `GEMINI_API_KEY`, `GEMINI_MODEL`

## Files Created So Far

```text
project_plan.md
requirements.txt
src/data/preprocess.py
src/models/baseline_svm.py
src/models/llm_prompter.py
up_til_now.md
```

## Next Planned Step

Configure and smoke-test the selected Gemini or Groq endpoint, complete the paper-compatible answer parsing refinement, and then proceed to Phase 5: evaluation, comparison, logging, and N=0/N=5/N=15 ablation experiments.
