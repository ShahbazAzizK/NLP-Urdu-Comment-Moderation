# Group 18 ? Roman Urdu Offensive Language Detection

**Course:** CS 5316: NLP Theory and Applications  
**Project type:** Re-implementation with a proposed novel extension  
**Current checkpoint:** Baseline completed; preliminary local LLM prompting pilot completed; larger validation and final report checks remain.

## Project question

Does adding Roman Urdu?Urdu-script alignment examples to LLM prompts improve offensive-language detection, especially under spelling variation? The proposed method is Cross-Script Alignment Prompting (CSAP).

## What is done, and where

### Project root

The root contains the original project code and the intermediate report source. The original character TF?IDF (2?5) + LinearSVC baseline used the supplied RUHSOLD Task 1 train/test files and achieved **0.8902 accuracy / 0.8894 macro-F1** on the 2,003-example test split. Its metrics and predictions are in `results/ruhsold_task1_svm_metrics.json` and `results/ruhsold_task1_svm_predictions.csv`. The root also contains the original preprocessing and prompt-building modules, project plan, Phase 5 checklist, and report source. The canonical rerunnable experiment notebooks and scripts are in `testing/`.

### `testing/`

`testing/` contains a separate leakage-aware, reproducible experiment package and the local LLM pilot. The no-API experiment removes training examples overlapping evaluation text, then compares matched character SVM with a Roman spelling-normalization probe. Test macro-F1: **0.8924 baseline**, **0.8999 normalization probe**. This probe is not the LLM novelty.

The corrected Ollama pilot used **Qwen 3.5 4B**, temperature 0, on 30 balanced RUHSOLD Task 1 test examples (15 per class), with 30 valid predictions per condition:

| Prompt condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Zero-shot | 0.8333 | 0.8316 |
| Four-example few-shot | 0.8667 | 0.8643 |
| CSAP, N=5 | 0.8667 | 0.8643 |
| CSAP, N=15 | 0.9000 | 0.8990 |

This 30-example pilot is **exploratory**, not conclusive. It was used for preliminary development; a larger evaluation with the mapping list reviewed and frozen is needed before claiming CSAP improves generalization. Full metrics: `testing/results/ollama_prompt_summary.json`; per-request outputs and latency: `testing/results/ollama_prompt_predictions.jsonl`. The summary is versioned `csap-urdu-utf8-v1`; legacy rows are retained in the JSONL but excluded from its scores.

## Data and interpretation caveats

- RUHSOLD Task 1 source labels are `0 = abusive/offensive`, `1 = normal`; supplied split sizes are 7,209 train, 801 validation, and 2,003 test.
- The supplied Task 2 validation TSV duplicates its training TSV and is excluded.
- Local RUHSOLD results are not a full reproduction of the anchor paper?s complete model/data evaluation.
- The spelling-normalization probe uses a manually specified Roman Urdu variant table. It needs Urdu-speaker review.
- Keep the original root baseline numbers separate from the `testing/` experiment numbers; their preprocessing/configurations differ.

## Remaining work

1. Have Urdu speakers verify the Roman Urdu?Urdu alignment entries and few-shot examples; freeze and version them.
2. Run a larger, predeclared, label-balanced local evaluation; record subset selection and avoid using test results to tune the lexicon.
3. Add uncertainty analysis and manual error analysis; report pilot and confirmatory results separately.
4. Obtain/use Akhter et al./RUTUBE or other planned domains only if time and data quality permit; document deviations.
5. Compile `intermediate_report.tex` to IEEE PDF and verify it is no more than three pages including references.
6. Update `baz_todo_beforeph5.md` based on actual completed evidence.

## Reproduce the experiments

From the repository root, run the no-API SVM/normalization experiment:

```powershell
.venv/Scripts/python.exe testing/run_experiments.py
```

Run the local LLM pilot (requires Ollama running and the model already downloaded):

```powershell
.venv/Scripts/python.exe testing/ollama_prompt_eval.py --model qwen3.5:4b --n 30
```

The Ollama runner saves each result and resumes successful requests. Neither experiment uses cloud API calls. See `testing/README.md` for setup details.

## Key files

- `intermediate_report.tex` ? IEEE report source; PDF layout/page count still needs final verification.
- `project_plan.md` ? project scope, methodology, and current phase status.
- `baz_todo_beforeph5.md` ? remaining readiness checklist.
- `testing/README.md` ? experiment details, pilot results, and Ollama instructions.
- `testing/ruhsold_experiments.ipynb` ? no-API experiment notebook.
- `testing/run_experiments.py` ? leakage-aware baseline/normalization experiment.
- `testing/ollama_prompt_eval.py` ? resumable local prompt-model evaluation.
