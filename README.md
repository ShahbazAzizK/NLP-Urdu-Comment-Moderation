# Group 18 ? Roman Urdu Offensive Language Detection

**Course:** CS 5316: NLP Theory and Applications  
**Project type:** Re-implementation with a proposed novel extension  
**Status:** Supervised baseline complete; local prompt-model evaluation planned.

## Project question

Does giving a language model a small set of Roman Urdu?Urdu-script spelling correspondences improve offensive-language classification, especially when Roman Urdu spelling varies? The proposed method is **Cross-Script Alignment Prompting (CSAP)**. The current intermediate report does not claim CSAP results. The checkpoint instructions allow either an initial novelty implementation or a clear account of planned work; our report states a concrete local evaluation plan.

## What is complete

- Local RUHSOLD Task 1 coarse-grained dataset audited. Source labels: `0 = abusive/offensive`, `1 = normal`. Supplied splits: 7,209 train, 801 validation, 2,003 test.
- Original project baseline: character TF?IDF (2?5) + LinearSVC (`C=1`), with test accuracy 0.8902 and macro-F1 0.8894. Metrics and predictions: `results/ruhsold_task1_svm_metrics.json`, `results/ruhsold_task1_svm_predictions.csv`.
- A second, self-contained experiment in `testing/` compares a matched character-SVM baseline with conservative exact-token Roman spelling normalization. It removes train examples whose normalized text overlaps validation/test before fitting. Current test macro-F1: 0.8924 baseline, 0.8999 normalization probe. These are preliminary and do not establish the LLM novelty.
- A Python experiment runner, Jupyter notebook, and saved outputs are in `testing/`.

## Data caveats

The supplied Task 2 validation file duplicates its training file and is excluded. The local experiments use RUHSOLD Task 1, not the anchor paper?s complete set of models/datasets; do not call the local result a full reproduction. The spelling normalization probe uses a manually specified Romanized variant list; review it with Urdu speakers before generalizing its result.

## Planned prompt-model evaluation

Hosted Gemini/Groq attempts were blocked by API quotas. No LLM evaluation metrics were saved. The next planned step is to run zero-shot, few-shot, and CSAP prompts at N=5 and N=15 with a local Ollama model, starting with `qwen3.5:4b`. The resumable runner is `testing/ollama_prompt_eval.py`; it evaluates a deterministic balanced 30-example test pilot and saves each response to `testing/results/ollama_prompt_predictions.jsonl` plus a summary JSON. It makes no cloud/API requests. The local model is optional for the completed SVM work but required to make empirical claims about the prompt-based novelty while hosted APIs are unavailable.

### Local model setup

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download) and leave its local service running.
2. In PowerShell, run `ollama pull qwen3.5:4b`; optionally verify it with `ollama run qwen3.5:4b`.
3. From the project root, run:

   ```powershell
   .venv/Scripts/python.exe testing/ollama_prompt_eval.py --model qwen3.5:4b --n 30
   ```


## Important files

- `intermediate_report.tex` ? IEEE conference-format source; compile and check the final PDF is at most three pages before submission.
- `project_plan.md` ? full scope, evaluation design, risks, and progress state.
- `baz_todo_beforeph5.md` ? evidence-based Phase 5 readiness checklist.
- `notebooks/ruhsold_baseline_and_alignment.ipynb` ? root baseline and prompt scaffold notebook.
- `testing/README.md` ? no-API experiment and Ollama setup details.
- `testing/ruhsold_experiments.ipynb` ? rerunnable SVM/normalization experiment.
- `testing/run_experiments.py` ? no-API runner.
- `testing/ollama_prompt_eval.py` ? optional resumable local LLM runner.

## Run the no-API experiments

From the project root:

```powershell
.venv/Scripts/python.exe testing/run_experiments.py
```

Outputs are written under `testing/results/`. Do not merge metrics from different implementations/configurations; each JSON summary records its own method.
