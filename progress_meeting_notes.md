# Progress meeting notes ? Group 18

## Completed work
- Audited the supplied RUHSOLD Task 1 coarse-grained train/validation/test files. Label 0 is abusive/offensive; label 1 is normal.
- Trained character TF?IDF (2?5) + LinearSVC (C=1.0) on 7,209 training rows and evaluated once on the supplied 2,003-row test set.
- Test accuracy: 0.8902; macro-F1: 0.8894; weighted F1: 0.8900. Confusion matrix, gold/predicted order [abusive/offensive, normal]: `[[808, 125], [95, 975]]`.
- Saved metrics and test predictions in `results/`; baseline notebook: `notebooks/ruhsold_baseline_and_alignment.ipynb`.

## Data caveat
Task 1 validation has 801 examples and was not used for tuning. Task 2 validation duplicates Task 2 training in the supplied files, so Task 2 is excluded. This baseline uses the local RUHSOLD release and is not claimed as a full reproduction of the 2026 anchor study.

## Proposed novelty: planned work
Cross-Script Alignment Prompting (CSAP) adds a small verified Roman Urdu?Urdu mapping list to LLM prompts. Next: verify mappings and few-shot demonstrations; freeze prompt and model settings; compare zero-shot, few-shot, and alignment N=5/N=15 conditions on the same deterministic stratified held-out subset; save responses incrementally; report macro-F1, per-class scores, coverage/failures, latency, and prompt length. Corrected local Qwen pilot completed (prompt version `csap-urdu-utf8-v1`); older placeholder-mapping rows are retained but excluded from current metrics. To avoid dependence on hosted API quotas, planned prompt inference will run locally with Ollama, starting with quantized Qwen 3.5 4B. If API limits constrain sample size, report the evaluation as a pilot.
