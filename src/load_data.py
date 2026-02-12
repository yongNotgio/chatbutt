"""
Data loader for Hiligaynon dictionary datasets.

Merges:
  - hiligaynon_dictionary.json  (25k entries: word + definition from pinoydictionary)
  - dictionary.json             (1.6k entries: Form + Meaning, concise)

Produces a unified dictionary list and train/dev examples.
"""

import json
import re
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip extra whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ── loaders ──────────────────────────────────────────────────────────────────

def load_pinoydictionary(path: str | Path) -> list[dict]:
    """
    Load hiligaynon_dictionary.json → list[{word, definition}].
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        word = _clean(item.get("word", ""))
        defn = _clean(item.get("definition", ""))
        if word and defn:
            entries.append({"word": word, "definition": defn})
    return entries


def load_concise_dictionary(path: str | Path) -> list[dict]:
    """
    Load dictionary.json → list[{word, definition}].
    Maps Form→word, Meaning→definition.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        word = _clean(item.get("Form", ""))
        meaning = _clean(item.get("Meaning", ""))
        if word and meaning:
            entries.append({"word": word, "definition": meaning})
    return entries


def merge_datasets(
    pinoy_path: str | Path | None = None,
    concise_path: str | Path | None = None,
) -> list[dict]:
    """
    Merge both datasets, deduplicating by (word, definition).
    Each entry: {word, definition, source}.
    """
    entries = []
    seen = set()

    if pinoy_path and Path(pinoy_path).exists():
        for e in load_pinoydictionary(pinoy_path):
            key = (e["word"].lower(), e["definition"][:80].lower())
            if key not in seen:
                seen.add(key)
                entries.append({**e, "source": "pinoydictionary"})

    if concise_path and Path(concise_path).exists():
        for e in load_concise_dictionary(concise_path):
            key = (e["word"].lower(), e["definition"][:80].lower())
            if key not in seen:
                seen.add(key)
                entries.append({**e, "source": "concise"})

    return entries


# ── save helpers ─────────────────────────────────────────────────────────────

def save_merged(entries: list[dict], out_path: str | Path):
    """Save merged dictionary to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(entries)} merged entries → {out_path}")


# ── train / dev split ───────────────────────────────────────────────────────

def create_examples(
    entries: list[dict],
    n_train: int = 60,
    n_dev: int = 30,
    out_dir: str | Path = "data/examples",
) -> tuple[list[dict], list[dict]]:
    """
    Sample train and dev examples from the concise dictionary entries
    (they have cleaner 1-to-1 word→meaning mappings).

    Each example: {hiligaynon, english, definition}
    """
    import random
    random.seed(42)

    # Prefer concise entries because they're cleaner for evaluation
    concise = [e for e in entries if e.get("source") == "concise"]
    if len(concise) < n_train + n_dev:
        concise = entries  # fallback to all

    pool = random.sample(concise, min(len(concise), n_train + n_dev))

    examples = []
    for e in pool:
        examples.append({
            "hiligaynon": e["word"],
            "english": e["definition"],
            "definition": e["definition"],
        })

    train = examples[:n_train]
    dev = examples[n_train : n_train + n_dev]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trainset.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out_dir / "devset.json", "w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False, indent=2)

    print(f"Created {len(train)} train + {len(dev)} dev examples → {out_dir}")
    return train, dev


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    entries = merge_datasets(
        pinoy_path=root / "hiligaynon_dictionary.json",
        concise_path=root / "dictionary.json",
    )
    save_merged(entries, root / "data" / "processed" / "dictionary.json")
    create_examples(entries, out_dir=root / "data" / "examples")
