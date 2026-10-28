"""Data loading and normalization utilities for Urdu offensive-language data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET_NAME = "roman_urdu_hate_speech"
DEFAULT_CONFIG_NAME = "Coarse_Grained"
DEFAULT_TEXT_COLUMNS = ("text", "tweet", "comment", "sentence", "content")
DEFAULT_LABEL_COLUMNS = ("label", "labels", "class", "category", "target")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{2,}", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")

# These substitutions cover common Arabic-script variants found in mixed Urdu data.
URDU_CHARACTER_TRANSLATIONS: Mapping[int, str] = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ه": "ہ",
        "ة": "ہ",
        "ۀ": "ہ",
        "ؤ": "و",
        "ئ": "ی",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ـ": "",
    }
)

OFFENSIVE_LABELS = {
    "1",
    "offensive",
    "offense",
    "offence",
    "hate",
    "hateful",
    "abusive",
    "yes",
    "true",
}
NORMAL_LABELS = {
    "0",
    "normal",
    "non-offensive",
    "non offensive",
    "nonoffensive",
    "not offensive",
    "not_offensive",
    "non-hate",
    "non hate",
    "nonhate",
    "clean",
    "no",
    "false",
}


def clean_text(text: str) -> str:
    """Remove URLs and mentions, reduce character elongation, and normalize spaces."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    without_urls = URL_PATTERN.sub(" ", text)
    without_mentions = MENTION_PATTERN.sub(" ", without_urls)
    without_elongation = REPEATED_CHARACTER_PATTERN.sub(r"\1\1", without_mentions)
    return WHITESPACE_PATTERN.sub(" ", without_elongation).strip()


def standardize_urdu_characters(text: str) -> str:
    """Map common Arabic Unicode variants to their Urdu equivalents."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.translate(URDU_CHARACTER_TRANSLATIONS)


def normalize_text(text: str) -> str:
    """Apply text cleaning followed by Arabic-to-Urdu character standardization."""
    return standardize_urdu_characters(clean_text(text))


def normalize_label(label: Any) -> int:
    """Convert common offensive-language labels to 1 (offensive) or 0 (normal)."""
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        if label in (0, 1):
            return label
        raise ValueError(f"Unsupported numeric label: {label}")
    if isinstance(label, float) and label.is_integer():
        return normalize_label(int(label))

    normalized = str(label).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in OFFENSIVE_LABELS:
        return 1
    if normalized in NORMAL_LABELS:
        return 0
    raise ValueError(f"Unsupported label value: {label!r}")


def _find_column(
    columns: list[str], preferred: str | None, candidates: tuple[str, ...], field: str
) -> str:
    """Resolve a requested column or a known default column name."""
    if preferred is not None:
        if preferred not in columns:
            raise ValueError(f"{field} column {preferred!r} is not present in {columns}")
        return preferred

    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not identify a {field} column in {columns}")


def _normalize_record(
    record: dict[str, Any], text_column: str, label_column: str
) -> dict[str, Any]:
    """Normalize one dataset record for use by downstream experiments."""
    return {
        **record,
        "text": normalize_text(record[text_column]),
        "label": normalize_label(record[label_column]),
    }


def prepare_split(
    split: Dataset,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
) -> Dataset:
    """Normalize text and labels in one Hugging Face dataset split."""
    columns = split.column_names
    resolved_text_column = _find_column(
        columns, text_column, DEFAULT_TEXT_COLUMNS, "text"
    )
    resolved_label_column = _find_column(
        columns, label_column, DEFAULT_LABEL_COLUMNS, "label"
    )
    return split.map(
        lambda record: _normalize_record(
            record, resolved_text_column, resolved_label_column
        )
    )


def load_ruhsold(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
    text_column: str | None = None,
    label_column: str | None = None,
    **load_kwargs: Any,
) -> DatasetDict | Dataset:
    """Load and normalize the RUHSOLD coarse-grained Hugging Face dataset.

    Additional keyword arguments are passed to ``datasets.load_dataset`` so callers
    can select revisions, cache locations, or specific splits without changing this
    module's public API.
    """
    dataset = load_dataset(dataset_name, config_name, **load_kwargs)
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split_name: prepare_split(
                    split,
                    text_column=text_column,
                    label_column=label_column,
                )
                for split_name, split in dataset.items()
            }
        )
    return prepare_split(
        dataset,
        text_column=text_column,
        label_column=label_column,
    )


__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_DATASET_NAME",
    "URDU_CHARACTER_TRANSLATIONS",
    "clean_text",
    "load_ruhsold",
    "normalize_label",
    "normalize_text",
    "prepare_split",
    "standardize_urdu_characters",
]