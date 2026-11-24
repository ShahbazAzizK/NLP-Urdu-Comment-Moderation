# Testing: reproducible RUHSOLD experiments

This folder contains a no-API experiment package for the project proposal. It runs locally using the repository Python environment and the supplied Task 1 TSV files. No LLM download, API key, or GPU is needed for the included experiments.

## Run

From the repository root:

```powershell
.venv/Scripts/python.exe testing/run_experiments.py
```

Or open `ruhsold_experiments.ipynb` in Jupyter and run all cells.

## Design

The runner compares two conditions on the same official split and classifier:

1. **Character SVM baseline:** URL/mention removal, repeated-character reduction, character TF-IDF 2?5, LinearSVC C=1.
2. **Roman spelling normalization probe:** the same pipeline plus exact-token canonicalization from a predeclared variant table.

Label IDs are preserved from the source: 0 = abusive/offensive; 1 = normal. Raw train/validation/test counts are 7,209/801/2,003. Validation is diagnostic; the supplied test split is used for final metrics. Exact normalized duplicate overlap is audited, and training rows overlapping validation or test are removed before fitting (7 rows removed in the current audit).

The spelling condition is a **no-API proxy experiment**, not LLM prompting and not Urdu-script transliteration. It offers a cheap initial check of whether common Roman Urdu spelling variation is worth pursuing. Do not claim it validates the proposal's LLM novelty. Review the variant groups with Urdu speakers before treating any gain as generalizable.

## Current run

On the supplied test set, the matched implementation produced:

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Character SVM baseline | 0.8932 | 0.8924 |
| Roman spelling normalization probe | 0.9006 | 0.8999 |

These are preliminary local results for this implementation. They differ from earlier baseline artifacts because this package uses explicit text cleaning, `sublinear_tf=True`, and removes train/evaluation text overlaps; do not combine numbers across configurations. Results and per-row predictions are written under `testing/results/`.

## What requires a local LLM?

Only empirical testing of the proposal's prompt-based novelty requires a language model. If API quota remains unavailable, a local model served through Ollama can replace the hosted endpoint; a GPU is helpful but not always necessary for a small quantized model. Hardware capability determines which model and speed are practical. The local SVM and normalization probe do not need an LLM.


## Run the proposed LLM prompt experiment locally (optional)

Your 32 GB RAM and RTX 5060 Laptop GPU should handle a small quantized model. Ollama officially lists RTX 50-series support; the Qwen 3.5 4B package is around 3?4 GB, making it the safer first choice for laptop VRAM. A 9B variant is larger (about 6.6?7.6 GB) and may spill into system RAM depending on the laptop GPU's VRAM and context use. Sources: [Ollama GPU support](https://github.com/ollama/ollama/blob/main/docs/gpu.mdx), [Qwen 3.5 model sizes](https://ollama.com/library/qwen3.5).

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download) and keep its local server running.
2. In PowerShell, run `ollama pull qwen3.5:4b`.
3. Smoke-test with `ollama run qwen3.5:4b` and type `Reply exactly: Not abusive`; exit with `/bye`.
4. From the project root run ` .venv/Scripts/python.exe testing/ollama_prompt_eval.py --model qwen3.5:4b --n 30 `.

The runner makes 120 local requests for 30 balanced test examples across zero-shot, few-shot, N=5, and N=15 alignment prompts. It saves each response immediately and resumes completed rows if interrupted. It uses four deterministic training demonstrations, fixed decoding, parses the source labels consistently, and writes summary metrics. Local model results must be labeled as a pilot and not compared as if they were the anchor paper's model. No API calls or paid billing are involved.


## Local prompt pilot (completed)

Ollama `qwen3.5:4b` completed 120 local requests: 30 balanced RUHSOLD Task 1 examples across four prompt conditions. Coverage was 100% for every condition. Macro-F1: zero-shot 0.8316; few-shot 0.8643; alignment N=5 0.8990; alignment N=15 0.8999. Accuracy: 0.8333, 0.8667, 0.9000, and 0.9000, respectively. Treat this as exploratory; confirm on a larger untouched sample and review mappings before drawing conclusions. Full metrics are in `results/ollama_prompt_summary.json`; raw outputs and latency are in `results/ollama_prompt_predictions.jsonl`.


## Visual pilot notebook

Open qwen_pilot_visualization.ipynb to view score, class F1, confusion matrix, and latency SVG charts generated from the saved local pilot predictions. Charts use notebook HTML/SVG and need no matplotlib/seaborn install.
