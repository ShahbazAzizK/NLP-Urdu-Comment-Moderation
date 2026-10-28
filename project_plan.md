# Cross-Script Offensive Language Detection

## Course Project

**Course:** CS 5316: NLP Theory and Applications  
**Category:** Category 2 - Re-implementation with a Novel Extension  
**Anchor paper:** Maab, Haider, and Yamagishi, "Prompt-driven Detection of Offensive Urdu Language," EACL 2026  
**Project team:** Group 18

## 1. Motivation

Tech-facilitated abuse on Pakistani social media is frequently expressed in Roman Urdu, Urdu-English code-switching, and informal chat spelling. The same word may appear in many forms, such as `nahi`, `nai`, or `nhi`. These variations create a script and normalization gap for moderation systems trained primarily on standardized Urdu or English text.

Prompt-based large language models (LLMs) provide a practical alternative to task-specific fine-tuning, especially for low-resource languages. However, a prompt that performs well on standard Urdu may lose accuracy on noisy Roman Urdu because the model must first infer transliteration, slang, abbreviations, and spelling variation before deciding whether a comment is offensive.

This project investigates whether a lightweight prompt intervention can reduce that gap. The intervention uses a small phonetic and lexical alignment dictionary to help the model interpret Roman Urdu variants in the context of offensive-language classification. The project emphasizes measurable NLP performance rather than user-interface development.

## 2. Problem Statement

This project will re-implement the principal prompting baselines from the anchor paper and compare them with a novel **Cross-Script Alignment Prompting** method.

The central research question is:

> Does augmenting an LLM classification prompt with Roman Urdu to standard Urdu alignment examples improve offensive-language detection on noisy, code-switched social media text compared with standard few-shot prompting?

The study will compare performance across clean Urdu, ordinary Roman Urdu, and deliberately stress-tested spelling-variation data. It will also measure the cost of the intervention through latency and prompt size.

### Hypotheses

1. Few-shot prompting will outperform zero-shot prompting on at least some language/script splits.
2. A character-level TF-IDF SVM will provide a strong, inexpensive classical lower bound on Roman Urdu data.
3. Cross-Script Alignment Prompting will improve Macro-F1, especially on spelling-variation examples, relative to standard few-shot prompting.
4. Increasing the number of dictionary examples will eventually produce diminishing returns or higher inference cost.

## 3. Re-Implementation Scope

The re-implementation will reproduce the comparison between conventional supervised classification and prompt-driven LLM classification as closely as possible within the available datasets and API-based inference constraints.

The implementation will document:

- Dataset versions, source URLs, licenses, and preprocessing decisions.
- Model/API name, model version, temperature, token limits, and request settings.
- Exact prompt templates and demonstration examples.
- Train, validation, and test split policy.
- Random seeds and experiment configuration.
- Any deviations from the anchor paper caused by unavailable models, data, or code.

The project will not use PyTorch or TensorFlow. Classical model training and metric computation will remain isolated to scikit-learn, while LLM inference will use an HTTP API.

## 4. Baselines

### 4.1 TF-IDF with Linear SVM

A supervised classical baseline will use a scikit-learn pipeline consisting of:

- Character-level `TfidfVectorizer`.
- Character n-gram range of 2 through 5.
- `LinearSVC` classifier.

The SVM will be trained on the designated training split and evaluated on each compatible test split. Character n-grams are appropriate for Roman Urdu because they can capture recurring spelling fragments without requiring a fixed word vocabulary.

This model establishes a reproducible lower bound and helps distinguish gains from prompting from gains that can already be achieved through robust character features.

### 4.2 Zero-Shot LLM Prompting

The zero-shot prompt will provide the classification task, label definitions, and the input text without labeled demonstrations. The model will be instructed to return a constrained label so that predictions can be parsed consistently.

This baseline measures the model's prior capability without task-specific examples in the prompt.

### 4.3 Standard Few-Shot LLM Prompting

The few-shot prompt will include three to five labeled examples selected from the standard task data. The examples will demonstrate the expected output format and the distinction between offensive and normal text, but will not include explicit cross-script alignment instructions or a mapping dictionary.

Example selection, ordering, and count will be fixed for the main comparison and recorded in the experiment configuration.

## 5. Novel Intervention: Cross-Script Alignment Prompting

The proposed method will extend the few-shot prompt with a compact alignment dictionary containing mappings from frequent Roman Urdu spellings, slang, or abbreviations to standard Urdu script equivalents. For example, a mapping may associate `kch` with the Urdu equivalent of "kuch".

Before classification, the LLM will be instructed to:

1. Inspect the input for Roman Urdu variants, abbreviations, and code-switched terms.
2. Use the supplied mappings as interpretive evidence rather than as an automatic relabeling rule.
3. Mentally align relevant Roman Urdu forms with their standard Urdu equivalents.
4. Classify the original input as `offensive` or `normal`.
5. Return only the required label for automated evaluation, with optional reasoning disabled in the scored output path.

### Ablation levels

The main ablation will inject different numbers of mapping examples:

- **N = 0:** No alignment dictionary; equivalent to the standard few-shot condition.
- **N = 5:** A small dictionary of high-frequency variants.
- **N = 15:** A larger dictionary covering additional spelling and slang variants.

The exact mappings will be versioned and separated from the test labels to avoid leakage. The ablation will compare Macro-F1, precision, recall, prompt length, and inference latency.

## 6. Data Plan

The planned data sources are:

1. **RUHSOLD:** Roman Urdu Hate Speech and Offensive Language Dataset, loaded from Hugging Face using the `roman_urdu_hate_speech` resource and the `Coarse_Grained` configuration where available.
2. **Akhter et al. (2020):** Offensive Language for Urdu and Roman Urdu, including Urdu-script and Roman Urdu social-media examples sourced from the authors' public repository.
3. **Custom spelling-variation set:** Approximately 300 held-out Roman Urdu comments created or transformed by the team to represent common spelling variation, abbreviations, and English mix-ins.

The custom set will be documented with its construction procedure, label policy, transformation rules, and quality-control process. It will be treated as an evaluation stress test, not mixed into model training.

All datasets will be checked for duplicate or near-duplicate examples across splits. Labels will be normalized to a binary scheme:

- `offensive` = 1
- `normal` = 0

The final report will clearly state any differences between the intended dataset sizes and the records that are actually available after loading, cleaning, deduplication, and filtering.

## 7. Evaluation Design

### Primary metric

- **Macro-F1**, because it gives equal weight to offensive and normal classes and is less misleading under class imbalance.

### Secondary metrics

- Macro precision.
- Macro recall.
- Per-class precision, recall, and F1.
- Confusion matrix.
- Inference latency per example and aggregate latency where measurable.
- Prompt token or character length for prompt-based conditions.

### Evaluation domains

Each applicable model will be evaluated separately on:

1. Clean Urdu script.
2. Standard Roman Urdu.
3. Custom spelling-variation Roman Urdu with English mix-ins.

The evaluation will use identical label normalization and prediction parsing across LLM conditions. Malformed or ambiguous LLM responses will be logged and counted using a pre-declared policy rather than silently removed.

### Error analysis

For the best-performing alignment configuration, the team will manually review up to 50 misclassified comments. Errors will be assigned to a documented taxonomy, such as:

- Missed sarcasm or implicit offensiveness.
- English slang or code-switching confusion.
- Transliteration ambiguity.
- Spelling-variation failure.
- Context or pragmatics failure.
- Label ambiguity or annotation noise.
- Prompt-following or output-parsing failure.

## 8. Execution Roadmap

### Phase 0 - Planning and documentation

- Create this project plan.
- Confirm the anchor paper, research question, baselines, intervention, datasets, and evaluation protocol.
- Identify assumptions and risks that must be checked before implementation.

### Phase 1 - Environment setup

- Create `requirements.txt` with the approved Python dependencies.
- Scaffold `data/`, `src/data/`, `src/models/`, `notebooks/`, and related output/configuration directories.
- Confirm Python 3.10+ and install dependencies.
- Configure API credentials through environment variables; never commit secrets.

### Phase 2 - Data pipeline

- Implement dataset loaders for RUHSOLD and the Akhter et al. data.
- Implement URL and mention removal, repeated-character normalization, and Urdu character standardization.
- Normalize binary labels.
- Produce auditable train/validation/test artifacts and dataset summary tables.
- Add checks for missing values, duplicates, class balance, and script distribution.

### Phase 3 - Classical baseline

- Implement the character TF-IDF plus Linear SVM pipeline.
- Train on the selected training split.
- Evaluate on all compatible domains.
- Save predictions, classification reports, confusion matrices, and configuration metadata.

### Phase 4 - LLM inference and prompting

- Implement API request handling with retries, timeouts, rate-limit awareness, and structured logging.
- Implement zero-shot, standard few-shot, and cross-script alignment prompt templates.
- Implement deterministic prediction parsing and malformed-response logging.
- Run a small smoke test before the full evaluation.
- Record model/API settings, prompt versions, latency, and cost-related metadata if available.

### Phase 5 - Evaluation and ablation

- Run the complete comparison across the planned test domains.
- Calculate and log Macro-F1, precision, recall, and per-class metrics.
- Run the N = 0, N = 5, and N = 15 alignment ablations.
- Compare accuracy gains against prompt length and latency.
- Generate tables and plots for the report.

### Phase 6 - Error analysis and final reporting

- Select the optimal configuration using the pre-declared primary metric.
- Review up to 50 errors and apply the error taxonomy.
- Discuss reproducibility, limitations, data quality, API variability, and possible leakage.
- Prepare the final report with introduction, related work, methodology, experiments, results, discussion, and conclusion.
- Prepare the 15-minute presentation/demo and clearly distinguish reproduced components from the novel extension.

## 9. Risks and Controls

- **Dataset availability or schema mismatch:** Inspect actual dataset features before coding against them; record version and fallback source.
- **Paper or implementation mismatch:** Report deviations explicitly instead of claiming exact reproduction when inputs or models differ.
- **LLM nondeterminism:** Use a fixed temperature where supported, repeat selected conditions when feasible, and record model/version details.
- **Prompt leakage:** Keep alignment examples separate from test examples and inspect for duplicate or near-duplicate text.
- **Class imbalance:** Use Macro-F1 and per-class metrics rather than accuracy alone.
- **API failures and budget limits:** Cache responses, use retries with bounded backoff, log failures, and support resumable evaluation.
- **Annotation ambiguity:** Preserve raw text and labels, document uncertain cases, and include qualitative analysis.
- **Custom-set validity:** Use a clear construction protocol and independent review where possible so that the stress test is not only synthetic noise.

## 10. Expected Deliverables

- Reproducible source code organized by data, models, and evaluation responsibilities.
- Versioned preprocessing and prompt configurations.
- Dataset statistics and quality checks.
- Baseline and intervention predictions.
- Metric tables, ablation results, latency measurements, and error taxonomy.
- Final course report and presentation/demo materials.

## 11. Success Criteria

The project will be considered successful if it:

1. Reproduces the baseline comparison with a documented, runnable pipeline.
2. Evaluates all three prompting conditions under the same label and metric policy.
3. Tests the alignment intervention on at least the planned Roman Urdu stress conditions.
4. Reports whether the intervention improves Macro-F1 and under what conditions.
5. Includes an ablation and error analysis that explain both gains and failures.
6. Makes all methodological deviations, data limitations, and API-dependent assumptions explicit.
