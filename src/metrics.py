"""
Evaluation metrics for the translation chatbot.

These metrics are used by DSPy optimizers to assess quality of translations.
"""

import json
from pathlib import Path


def exact_match_metric(example, prediction, trace=None) -> float:
    """
    Check if the predicted translation exactly matches any known translation.
    Used primarily for single-word lookups where exact answers exist.

    Returns 1.0 for exact match, 0.5 for partial match, 0.0 for no match.
    """
    score = 0.0

    # Check Hiligaynon
    if hasattr(prediction, "hiligaynon") and hasattr(example, "hiligaynon"):
        pred_hil = prediction.hiligaynon.lower().strip()
        # Check against all known translations
        all_hil = example.get("all_hiligaynon", [example.hiligaynon])
        if isinstance(all_hil, str):
            all_hil = [all_hil]
        all_hil_lower = [h.lower().strip() for h in all_hil]

        if pred_hil in all_hil_lower:
            score += 0.5
        elif any(h in pred_hil or pred_hil in h for h in all_hil_lower):
            score += 0.25

    # Check Akeanon
    if hasattr(prediction, "akeanon") and hasattr(example, "akeanon"):
        pred_ake = prediction.akeanon.lower().strip()
        all_ake = example.get("all_akeanon", [example.akeanon])
        if isinstance(all_ake, str):
            all_ake = [all_ake]
        all_ake_lower = [a.lower().strip() for a in all_ake]

        if pred_ake in all_ake_lower:
            score += 0.5
        elif any(a in pred_ake or pred_ake in a for a in all_ake_lower):
            score += 0.25

    return score


def translation_quality_metric(example, prediction, trace=None) -> float:
    """
    A combined metric that checks:
    1. Whether the output contains valid translations (not empty)
    2. Format correctness
    3. Dictionary match (if available)

    Returns a score between 0.0 and 1.0.
    """
    score = 0.0
    max_score = 0.0

    # Check that translations are non-empty
    max_score += 0.2
    if hasattr(prediction, "hiligaynon") and prediction.hiligaynon.strip():
        score += 0.1
    if hasattr(prediction, "akeanon") and prediction.akeanon.strip():
        score += 0.1

    # Check format (should not contain English words from the query)
    max_score += 0.2
    if hasattr(example, "english") and hasattr(prediction, "hiligaynon"):
        english_lower = example.english.lower()
        hil_lower = prediction.hiligaynon.lower()
        ake_lower = getattr(prediction, "akeanon", "").lower()
        # Penalize if the translation is just the English word repeated
        if english_lower != hil_lower:
            score += 0.1
        if english_lower != ake_lower:
            score += 0.1

    # Dictionary exact match (highest weight)
    max_score += 0.6
    exact_score = exact_match_metric(example, prediction, trace)
    score += exact_score * 0.6

    return score / max_score if max_score > 0 else 0.0


def load_examples(examples_path: str | Path) -> list[dict]:
    """Load evaluation examples from a JSON file."""
    with open(examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to dspy.Example objects
    import dspy

    examples = []
    for item in data:
        ex = dspy.Example(
            english=item["english"],
            pos=item.get("pos", ""),
            hiligaynon=item["hiligaynon"],
            akeanon=item["akeanon"],
            all_hiligaynon=item.get("all_hiligaynon", [item["hiligaynon"]]),
            all_akeanon=item.get("all_akeanon", [item["akeanon"]]),
        ).with_inputs("english", "pos")
        examples.append(ex)

    return examples
