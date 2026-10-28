"""Character n-gram TF-IDF and Linear SVM baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


LABELS = (0, 1)
TARGET_NAMES = ("normal", "offensive")
ClassificationReport = dict[str, Any]


def build_svm_pipeline(
    *,
    c: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
) -> Pipeline:
    """Build the required character TF-IDF and LinearSVC pipeline."""
    if c <= 0:
        raise ValueError("c must be greater than zero")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(2, 5)),
            ),
            (
                "svm",
                LinearSVC(C=c, class_weight=class_weight),
            ),
        ]
    )


def _validate_training_data(
    texts: Sequence[str], labels: Sequence[int]
) -> tuple[list[str], list[int]]:
    """Validate and materialize parallel text and label sequences."""
    materialized_texts = list(texts)
    materialized_labels = list(labels)
    if not materialized_texts:
        raise ValueError("training data must contain at least one example")
    if len(materialized_texts) != len(materialized_labels):
        raise ValueError("texts and labels must have the same length")
    if any(not isinstance(text, str) for text in materialized_texts):
        raise TypeError("all texts must be strings")
    if any(label not in LABELS for label in materialized_labels):
        raise ValueError("labels must contain only 0 (normal) or 1 (offensive)")
    return materialized_texts, materialized_labels


def train_svm(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    c: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
) -> Pipeline:
    """Fit and return a character n-gram TF-IDF SVM classifier."""
    training_texts, training_labels = _validate_training_data(texts, labels)
    model = build_svm_pipeline(c=c, class_weight=class_weight)
    model.fit(training_texts, training_labels)
    return model


def predict_svm(model: Pipeline, texts: Sequence[str]) -> list[int]:
    """Predict binary labels for a sequence of normalized texts."""
    prediction_texts = list(texts)
    if any(not isinstance(text, str) for text in prediction_texts):
        raise TypeError("all texts must be strings")
    return [int(label) for label in model.predict(prediction_texts)]


def evaluate_svm(
    model: Pipeline,
    texts: Sequence[str],
    labels: Sequence[int],
) -> ClassificationReport:
    """Return a scikit-learn classification report for the SVM predictions.

    The returned dictionary includes per-class precision, recall, and F1 as well as
    macro and weighted averages, making it directly usable by later evaluation code.
    """
    evaluation_texts, evaluation_labels = _validate_training_data(texts, labels)
    predictions = predict_svm(model, evaluation_texts)
    return classification_report(
        evaluation_labels,
        predictions,
        labels=list(LABELS),
        target_names=list(TARGET_NAMES),
        output_dict=True,
        zero_division=0,
    )


def classification_report(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    labels: list[int],
    target_names: list[str],
    output_dict: bool,
    zero_division: int,
) -> ClassificationReport:
    """Call scikit-learn's report function through a local typed wrapper."""
    from sklearn.metrics import classification_report as sklearn_classification_report

    return sklearn_classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=zero_division,
    )


__all__ = [
    "ClassificationReport",
    "build_svm_pipeline",
    "evaluate_svm",
    "predict_svm",
    "train_svm",
]