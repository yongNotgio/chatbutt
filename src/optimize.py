"""
DSPy optimization pipeline for the translation chatbot.

Supports:
1. BootstrapFewShot — good with 10-50 examples
2. MIPROv2 — better with 200+ examples, optimizes instructions + demos
3. BootstrapFinetune — fine-tunes a local model (requires GPU)
"""

import json
from pathlib import Path

import dspy

from src.modules import TranslationChatbot
from src.metrics import translation_quality_metric, load_examples


def setup_lm(
    provider: str = "groq",
    model: str = "llama-3.3-70b-versatile",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dspy.LM:
    """
    Configure the language model for DSPy.

    Args:
        provider: 'groq', 'openai', 'ollama', or 'anthropic'
        model: Model name (e.g., 'llama-3.3-70b-versatile', 'gpt-4o-mini')
        api_key: API key (not needed for Ollama)
        base_url: Base URL for the API (for Ollama: 'http://localhost:11434')
    """
    if provider == "groq":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        lm = dspy.LM(f"groq/{model}", **kwargs)
    elif provider == "ollama":
        lm = dspy.LM(
            f"ollama_chat/{model}",
            api_base=base_url or "http://localhost:11434",
        )
    elif provider == "openai":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        lm = dspy.LM(f"openai/{model}", **kwargs)
    elif provider == "anthropic":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        lm = dspy.LM(f"anthropic/{model}", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    dspy.configure(lm=lm)
    print(f"Configured DSPy with {provider}/{model}")
    return lm


def optimize_bootstrap_fewshot(
    chatbot: TranslationChatbot,
    trainset: list,
    metric=translation_quality_metric,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
) -> TranslationChatbot:
    """
    Optimize using BootstrapFewShot.
    Best for 10-50 training examples.
    """
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    # The optimizer needs a simple wrapper that accepts Example objects
    optimized = optimizer.compile(chatbot, trainset=trainset)
    print("BootstrapFewShot optimization complete!")
    return optimized


def optimize_mipro(
    chatbot: TranslationChatbot,
    trainset: list,
    metric=translation_quality_metric,
    num_trials: int = 20,
) -> TranslationChatbot:
    """
    Optimize using MIPROv2.
    Best for 200+ examples. Jointly optimizes instructions and demos.
    """
    optimizer = dspy.MIPROv2(
        metric=metric,
        num_candidates=7,
        init_temperature=1.0,
    )

    optimized = optimizer.compile(
        chatbot,
        trainset=trainset,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        num_trials=num_trials,
    )
    print("MIPROv2 optimization complete!")
    return optimized


def evaluate_program(
    program: TranslationChatbot,
    devset: list,
    metric=translation_quality_metric,
) -> float:
    """Evaluate a program on the dev set."""
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=1,
        display_progress=True,
    )

    result = evaluator(program)
    # Extract numeric score from EvaluationResult
    score = float(result) if not isinstance(result, (int, float)) else result
    print(f"Evaluation score: {score:.2f}%")
    return score


def save_program(program: TranslationChatbot, path: str | Path):
    """Save an optimized program to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))
    print(f"Saved optimized program to {path}")


def load_program(path: str | Path, chroma_dir: str = "chroma_db") -> TranslationChatbot:
    """Load an optimized program from disk."""
    chatbot = TranslationChatbot(chroma_dir=chroma_dir)
    chatbot.load(str(path))
    print(f"Loaded optimized program from {path}")
    return chatbot


# --- CLI ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    project_root = Path(__file__).parent.parent

    # Setup LM
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "gsk_your-key-here":
        setup_lm(provider="groq", model="llama-3.3-70b-versatile", api_key=api_key)
    else:
        print("No Groq API key found. Trying Ollama...")
        setup_lm(provider="ollama", model="llama3.2")

    # Load examples
    trainset = load_examples(project_root / "data" / "examples" / "trainset.json")
    devset = load_examples(project_root / "data" / "examples" / "devset.json")

    # Create chatbot
    chroma_dir = project_root / "chroma_db"
    chatbot = TranslationChatbot(chroma_dir=str(chroma_dir))

    # Optimize
    print("\n--- Running BootstrapFewShot optimization ---")
    optimized = optimize_bootstrap_fewshot(chatbot, trainset)

    # Evaluate
    print("\n--- Evaluating optimized program ---")
    score = evaluate_program(optimized, devset)

    # Save
    save_path = project_root / "optimized" / "translation_v1.json"
    save_program(optimized, save_path)

    print(f"\nDone! Score: {score:.2%}")
