"""Prompt templates and API inference helpers for offensive-language detection."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import requests
from dotenv import load_dotenv


Label = Literal[0, 1]
Provider = Literal["openai_compatible", "gemini"]


@dataclass(frozen=True)
class PromptExample:
    """One labeled demonstration for few-shot classification."""

    text: str
    label: Label


@dataclass(frozen=True)
class AlignmentExample:
    """One Roman Urdu to Urdu-script alignment mapping."""

    roman_urdu: str
    urdu_script: str


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an API-backed text-generation request."""

    endpoint: str
    api_key: str
    model: str
    provider: Provider = "openai_compatible"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 8
    max_retries: int = 3

    @classmethod
    def from_environment(cls, provider: Provider = "openai_compatible") -> "LLMConfig":
        """Build configuration from a local ``.env`` file or process environment."""
        load_dotenv()
        prefix = "GEMINI" if provider == "gemini" else "LLM"
        endpoint = os.getenv(f"{prefix}_ENDPOINT")
        api_key = os.getenv(f"{prefix}_API_KEY")
        model = os.getenv(f"{prefix}_MODEL")
        missing = [
            name
            for name, value in (
                (f"{prefix}_ENDPOINT", endpoint),
                (f"{prefix}_API_KEY", api_key),
                (f"{prefix}_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            provider=provider,
        )


DEFAULT_FEW_SHOT_EXAMPLES = (
    PromptExample("Yeh bohat acha kaam hai.", 0),
    PromptExample("Tum ek badtameez insaan ho.", 1),
    PromptExample("Kal mausam acha tha.", 0),
    PromptExample("Us ne mujhe gaali di.", 1),
)

DEFAULT_ALIGNMENT_EXAMPLES = (
    AlignmentExample("kch", "کچھ"),
    AlignmentExample("nhi", "نہیں"),
    AlignmentExample("nai", "نہیں"),
    AlignmentExample("kr", "کر"),
    AlignmentExample("ap", "آپ"),
    AlignmentExample("aap", "آپ"),
    AlignmentExample("mjhy", "مجھے"),
    AlignmentExample("mujhy", "مجھے"),
    AlignmentExample("tumhy", "تمہیں"),
    AlignmentExample("q k", "کیونکہ"),
    AlignmentExample("kyun k", "کیونکہ"),
    AlignmentExample("acha", "اچھا"),
    AlignmentExample("badtameez", "بدتمیز"),
    AlignmentExample("gaali", "گالی"),
    AlignmentExample("insaan", "انسان"),
)


def _label_name(label: Label) -> str:
    """Convert an integer label to the prompt's public label name."""
    return "offensive" if label == 1 else "normal"


def zero_shot_prompt(text: str) -> str:
    """Create a zero-shot classification prompt for one input text."""
    return (
        "Classify the following social-media text as exactly one label: "
        "offensive or normal. Return only the label, with no explanation.\n\n"
        f"Text: {text}\n"
        "Label:"
    )


def few_shot_prompt(
    text: str,
    examples: Iterable[PromptExample] = DEFAULT_FEW_SHOT_EXAMPLES,
) -> str:
    """Create a few-shot prompt using labeled examples without alignment mappings."""
    demonstrations = "\n".join(
        f"Example text: {example.text}\nLabel: {_label_name(example.label)}"
        for example in examples
    )
    return (
        "Classify social-media text as exactly one label: offensive or normal. "
        "Return only the label, with no explanation.\n\n"
        f"Demonstrations:\n{demonstrations}\n\n"
        f"Text: {text}\n"
        "Label:"
    )


def cross_script_alignment_prompt(
    text: str,
    *,
    dictionary_examples: Iterable[AlignmentExample] = DEFAULT_ALIGNMENT_EXAMPLES,
    n_dictionary_examples: int = 15,
    few_shot_examples: Iterable[PromptExample] = DEFAULT_FEW_SHOT_EXAMPLES,
) -> str:
    """Create a few-shot prompt with N Roman Urdu to Urdu alignment examples."""
    if n_dictionary_examples < 0:
        raise ValueError("n_dictionary_examples must be non-negative")

    alignment_examples = list(dictionary_examples)[:n_dictionary_examples]
    mappings = "\n".join(
        f"Roman Urdu: {example.roman_urdu}\nUrdu: {example.urdu_script}"
        for example in alignment_examples
    )
    alignment_section = mappings or "(No alignment mappings provided.)"
    base_prompt = few_shot_prompt(text, few_shot_examples)
    return (
        "Before classification, inspect Roman Urdu spelling, abbreviations, and slang. "
        "Use the mappings below to align relevant forms with standard Urdu script. "
        "Treat mappings as interpretive evidence, then classify the original text. "
        "Return only offensive or normal, with no explanation.\n\n"
        f"Alignment mappings ({len(alignment_examples)}):\n{alignment_section}\n\n"
        f"{base_prompt}"
    )


def parse_label(response_text: str) -> Label:
    """Parse an API response into the binary project label.

    Exact labels are preferred. A labeled response such as ``Label: offensive`` is
    also accepted; ambiguous responses raise ``ValueError`` for explicit logging.
    """
    normalized = response_text.strip().lower()
    if normalized in {"normal", "0"}:
        return 0
    if normalized in {"offensive", "1"}:
        return 1

    labeled_match = re.search(r"\blabel\s*[:=-]\s*(normal|offensive|0|1)\b", normalized)
    if labeled_match:
        return 1 if labeled_match.group(1) in {"offensive", "1"} else 0
    raise ValueError(f"Could not parse a unique offensive-language label: {response_text!r}")


def _openai_payload(config: LLMConfig, prompt: str) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def _gemini_payload(config: LLMConfig, prompt: str) -> dict[str, Any]:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        },
    }


def _extract_response_text(payload: dict[str, Any], provider: Provider) -> str:
    """Extract generated text from the supported provider response shapes."""
    if provider == "gemini":
        return str(payload["candidates"][0]["content"]["parts"][0]["text"])
    return str(payload["choices"][0]["message"]["content"])


def request_completion(
    prompt: str,
    config: LLMConfig,
    *,
    session: requests.Session | None = None,
) -> str:
    """Send one prompt to the configured API with bounded retries."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if config.max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    client = session or requests.Session()
    headers = {"Content-Type": "application/json"}
    request_kwargs: dict[str, Any] = {
        "headers": headers,
        "timeout": config.timeout_seconds,
    }
    if config.provider == "gemini":
        request_kwargs["params"] = {"key": config.api_key}
        payload = _gemini_payload(config, prompt)
    else:
        headers["Authorization"] = f"Bearer {config.api_key}"
        payload = _openai_payload(config, prompt)

    for attempt in range(config.max_retries + 1):
        response = client.post(config.endpoint, json=payload, **request_kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < config.max_retries:
                time.sleep(2**attempt)
                continue
        response.raise_for_status()
        return _extract_response_text(response.json(), config.provider)
    raise RuntimeError("LLM request failed after all retries")


def classify_text(
    text: str,
    prompt_builder: Any,
    config: LLMConfig,
    *,
    session: requests.Session | None = None,
) -> Label:
    """Build a prompt, request a completion, and parse its binary label."""
    prompt = prompt_builder(text)
    return parse_label(request_completion(prompt, config, session=session))


__all__ = [
    "AlignmentExample",
    "DEFAULT_ALIGNMENT_EXAMPLES",
    "DEFAULT_FEW_SHOT_EXAMPLES",
    "LLMConfig",
    "PromptExample",
    "classify_text",
    "cross_script_alignment_prompt",
    "few_shot_prompt",
    "parse_label",
    "request_completion",
    "zero_shot_prompt",
]