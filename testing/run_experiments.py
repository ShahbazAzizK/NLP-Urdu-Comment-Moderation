"""Reproducible, no-API RUHSOLD experiment for Group 18.

Compares the char n-gram SVM baseline with a spelling-canonicalized variant.
Run from the repository root: python testing/run_experiments.py
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/2020paper/roman_urdu_hate_speech-main"
OUT = Path(__file__).resolve().parent / "results"
SEED = 473
LABEL_NAMES = {0: "abusive/offensive", 1: "normal"}

# Conservative, transparent Roman Urdu spelling groups. Canonical forms are
# Latin-script anchors; no Urdu transliteration is claimed. Keep the mapping
# versioned and review expansions before interpreting the ablation.
VARIANTS = {
    "nahi": {"nahi", "nai", "nhi", "nahin"},
    "acha": {"acha", "accha", "achha"},
    "kyun": {"kyun", "kyu", "kiun"},
    "mujhe": {"mujhe", "mujhy", "mujhay", "mje", "mjhe", "mjhy"},
    "hai": {"hai", "hey", "hay"},
    "tumhe": {"tumhe", "tumhay", "tumhy"},
    "aap": {"ap", "aap"},
    "bohat": {"bohat", "bahut", "bhot"},
    "kuch": {"kuch", "kch"},
    "kyunki": {"kyunki", "kyunke", "kyun_kay", "qke", "q_k"},
}
TOKEN_MAP = {variant: canonical for canonical, variants in VARIANTS.items() for variant in variants}
TOKEN_RE = re.compile(r"[^\W_]+(?:['?][^\W_]+)?|[^\w\s]", flags=re.UNICODE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return SPACE_RE.sub(" ", text).strip()

def canonicalize_roman_variants(text: str) -> str:
    """Replace exact whitespace-delimited Roman Urdu tokens conservatively."""
    tokens = TOKEN_RE.findall(clean_text(text))
    return " ".join(TOKEN_MAP.get(tok.lower(), tok) for tok in tokens)

def read_split(split: str) -> tuple[list[str], list[int]]:
    path = DATA / f"task_1_{split}.tsv"
    xs, ys = [], []
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 2 or not row[0].strip() or row[1].strip() not in {"0", "1"}:
                continue
            xs.append(row[0].strip()); ys.append(int(row[1]))
    if not xs: raise ValueError(f"No valid examples in {path}")
    return xs, ys

def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                   lowercase=True, sublinear_tf=True)),
        ("svm", LinearSVC(C=1.0, class_weight=None, random_state=SEED)),
    ])

def duplicate_audit(train: list[str], val: list[str], test: list[str]) -> dict:
    norm = lambda xs: {clean_text(x).casefold() for x in xs}
    a,b,c=norm(train),norm(val),norm(test)
    return {"train_unique":len(a),"validation_unique":len(b),"test_unique":len(c),
            "train_validation_exact_overlap":len(a&b),"train_test_exact_overlap":len(a&c),
            "validation_test_exact_overlap":len(b&c)}

def evaluate_condition(name: str, transform: Callable[[str],str], splits: dict) -> tuple[dict,list[dict]]:
    xt,yt=splits["train"]; xv,yv=splits["validation"]; xe,ye=splits["test"]
    model=make_model(); start=time.perf_counter()
    model.fit([transform(x) for x in xt],yt); fit_seconds=time.perf_counter()-start
    # Validation is reported as a diagnostic. Test remains the final held-out result.
    pv=model.predict([transform(x) for x in xv]); pe=model.predict([transform(x) for x in xe])
    report=classification_report(ye,pe,labels=[0,1],target_names=[LABEL_NAMES[0],LABEL_NAMES[1]],output_dict=True,zero_division=0)
    cm=confusion_matrix(ye,pe,labels=[0,1]).tolist()
    metrics={"condition":name,"model":"char TF-IDF (2-5) + LinearSVC(C=1.0)",
      "preprocessing":"URL/mention removal, elongation reduction, whitespace normalization"+(" + exact-token Roman Urdu variant canonicalization" if name!="baseline" else ""),
      "train_n":len(xt),"validation_n":len(xv),"test_n":len(xe),
      "validation_macro_f1":precision_recall_fscore_support(yv,pv,labels=[0,1],average="macro",zero_division=0)[2],
      "test_accuracy":accuracy_score(ye,pe),"test_macro_precision":report["macro avg"]["precision"],
      "test_macro_recall":report["macro avg"]["recall"],"test_macro_f1":report["macro avg"]["f1-score"],
      "test_weighted_f1":report["weighted avg"]["f1-score"],"class_report":report,
      "confusion_matrix_label_order":[LABEL_NAMES[0],LABEL_NAMES[1]],"confusion_matrix":cm,
      "fit_seconds":fit_seconds}
    predictions=[{"condition":name,"row_id":i,"text":text,"gold_label":int(g),
                  "gold_name":LABEL_NAMES[int(g)],"prediction":int(p),"prediction_name":LABEL_NAMES[int(p)]}
                 for i,(text,g,p) in enumerate(zip(xe,ye,pe))]
    return metrics,predictions

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    splits={s:read_split(s) for s in ("train","validation","test")}
    raw_sizes={s:len(pair[0]) for s,pair in splits.items()}
    audit=duplicate_audit(splits["train"][0],splits["validation"][0],splits["test"][0])
    # Prevent exact normalized-text leakage from evaluation into training. Keep
    # the supplied validation/test rows intact; drop conflicting train copies.
    evaluation_texts={clean_text(x).casefold() for split in ("validation","test") for x in splits[split][0]}
    xtr,ytr=splits["train"]
    kept=[i for i,x in enumerate(xtr) if clean_text(x).casefold() not in evaluation_texts]
    splits["train"] = ([xtr[i] for i in kept],[ytr[i] for i in kept])
    audit["train_rows_removed_for_eval_overlap"] = raw_sizes["train"]-len(kept)
    audit["train_rows_after_leakage_filter"] = len(kept)
    n={s:len(pair[0]) for s,pair in splits.items()}
    dist={s:{str(k):v for k,v in sorted(Counter(pair[1]).items())} for s,pair in splits.items()}
    all_metrics=[]; all_predictions=[]
    for name,fn in (("baseline",clean_text),("roman_variant_normalization",canonicalize_roman_variants)):
        metrics,preds=evaluate_condition(name,fn,splits); all_metrics.append(metrics); all_predictions.extend(preds)
        print(name, "test_macro_f1=",round(metrics["test_macro_f1"],4),"accuracy=",round(metrics["test_accuracy"],4),flush=True)
    summary={"dataset":"RUHSOLD Task 1 coarse-grained; local supplied splits",
      "label_mapping":{"0":LABEL_NAMES[0],"1":LABEL_NAMES[1]},"split_sizes":n,"label_distributions":dist,
      "duplicate_audit_exact_normalized_text":audit,"random_seed":SEED,
      "novelty":"No-API cross-script spelling robustness probe: canonicalize predeclared exact Roman Urdu variant tokens before char n-gram classification. This is a lexical-normalization proxy, not LLM Cross-Script Alignment Prompting and not Urdu-script transliteration.",
      "mapping_groups":{k:sorted(v) for k,v in VARIANTS.items()},"conditions":all_metrics}
    (OUT/"experiment_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"test_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_predictions[0])); w.writeheader(); w.writerows(all_predictions)
    print("saved",OUT)
if __name__=="__main__": main()
