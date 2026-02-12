"""
Evaluation metrics for the Hiligaynon ↔ English chatbot.

Metrics assess:
1. Whether the response contains a valid translation
2. Vowel-variant awareness (o↔u, i↔e matching)
3. Response quality and helpfulness
"""

import json
import re
from pathlib import Path

from src.retriever import normalize_vowels


def _contains_word(text: str, word: str) -> bool:
    """Check if text contains the word (case-insensitive, word-boundary aware)."""
    return bool(re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))


def _vowel_match(predicted: str, expected: str) -> bool:
    """Check if two strings match when vowels are normalized (o↔u, i↔e)."""
    return normalize_vowels(predicted.strip()) == normalize_vowels(expected.strip())


def translation_relevance_metric(example, prediction, trace=None) -> float:
    """
    Check if the chatbot's response is relevant and contains correct information.

    Scoring:
    - 0.4: Response mentions the key Hiligaynon word (or vowel variant)
    - 0.3: Response mentions the English meaning (or part of it)
    - 0.2: Response is non-empty and seems like a real answer
    - 0.1: Response is conversational (not just the raw word)
    """
    score = 0.0
    response = getattr(prediction, "response", "")
    if not response.strip():
        return 0.0

    # Get expected values
    expected_hil = getattr(example, "hiligaynon", "")
    expected_eng = getattr(example, "english", "")

    # Score: response is non-empty and substantial
    if len(response.strip()) > 10:
        score += 0.2

    # Score: mentions the Hiligaynon word (with vowel flexibility)
    if expected_hil:
        resp_norm = normalize_vowels(response)
        hil_norm = normalize_vowels(expected_hil)
        if hil_norm in resp_norm:
            score += 0.4
        elif any(
            normalize_vowels(w) == hil_norm
            for w in re.findall(r"\b\w+\b", response.lower())
        ):
            score += 0.4
        elif _contains_word(response, expected_hil):
            score += 0.3

    # Score: mentions the English meaning
    if expected_eng:
        # Check if any significant word from the English meaning appears
        eng_words = [
            w for w in re.findall(r"\b\w+\b", expected_eng.lower())
            if len(w) > 3
        ]
        if eng_words:
            matches = sum(1 for w in eng_words if _contains_word(response, w))
            match_ratio = matches / len(eng_words)
            score += 0.3 * min(match_ratio * 2, 1.0)  # up to 0.3

    # Score: conversational (contains explanation, not just a word)
    if len(response.split()) > 5:
        score += 0.1

    return min(score, 1.0)


def load_examples(examples_path: str | Path) -> list:
    """Load evaluation examples from JSON. Returns list of dspy.Example."""
    import dspy

    with open(examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        ex = dspy.Example(
            hiligaynon=item.get("hiligaynon", ""),
            english=item.get("english", ""),
            definition=item.get("definition", ""),
        ).with_inputs("hiligaynon", "english")
        examples.append(ex)

    return examples
