"""
Parse the English-Hiligaynon-Akeanon HTML dictionary into structured JSON.

Source: https://howtospeakaklanon.blogspot.com/2011/09/english-hiligaynon-akeanon.html
Author of dictionary: Melchor F. Cichon

The HTML is a Blogger page where translation entries are inside
<div class='post-body entry-content'> separated by <br/> tags.

Entry format (with variations):
  english_term (part_of_speech)--hiligaynon--akeanon
  english_term (part_of_speech)--hiligaynon
  english_term--hiligaynon--akeanon
  english_term hiligaynon--akeanon   (space-separated, no --)
"""

import json
import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup


# Known parts of speech to help with parsing
POS_TAGS = {
    "noun", "verb", "adjective", "adverb", "preposition", "pronoun",
    "conjunction", "interjection", "prefix", "suffix", "article",
    "indefinite article", "definite article", "plural", "singular",
    "past tense", "present tense", "future tense",
}


def load_html(html_path: str | Path) -> str:
    """Load the HTML file and return as string."""
    html_path = Path(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_raw_lines(html_content: str) -> list[str]:
    """
    Extract the raw dictionary lines from the HTML.
    The content is inside <div class='post-body entry-content'> and
    entries are separated by <br/> tags.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Find the post body div
    post_body = soup.find("div", class_="post-body entry-content")
    if not post_body:
        raise ValueError("Could not find post-body div in HTML")

    # Get the inner HTML and split by <br/> variants
    inner_html = str(post_body)

    # Replace <br/> variants with a unique delimiter
    inner_html = re.sub(r"<br\s*/?>", "\n", inner_html)

    # Parse the cleaned HTML to get text
    cleaned_soup = BeautifulSoup(inner_html, "lxml")
    text = cleaned_soup.get_text()

    # Split into lines and clean
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Skip empty lines, metadata, and header lines
        if not line:
            continue
        if line.startswith("English") and "Hiligaynon" in line:
            continue
        if line.startswith("by"):
            continue
        if line.startswith("Melchor"):
            continue
        if line.startswith("started:"):
            continue
        if line.startswith("As of") or line.startswith("as of"):
            continue
        if line.startswith("(This is"):
            continue
        if len(line) < 3:
            continue
        lines.append(line)

    return lines


def extract_pos(text: str) -> tuple[str, str]:
    """
    Extract part of speech from a string like 'abandon verb' or 'abandon (verb)'.
    Returns (cleaned_text, pos).
    """
    # Try parenthesized POS: "word (noun)" or "word ( noun)"
    match = re.search(r"\(\s*([^)]+)\s*\)", text)
    if match:
        candidate = match.group(1).strip().lower()
        # Check if it looks like a POS tag
        for pos in POS_TAGS:
            if candidate.startswith(pos) or pos in candidate:
                cleaned = text[: match.start()].strip()
                return cleaned, candidate
        # It might be a usage note like "(to be taken aback)" - not a POS
        # Keep it in the text
        return text, ""

    # Try unparenthesized POS at the end: "abandon verb"
    words = text.split()
    if len(words) >= 2:
        last_word = words[-1].lower().rstrip(",;:")
        if last_word in POS_TAGS:
            return " ".join(words[:-1]), last_word
        # Check two-word POS: "indefinite article"
        if len(words) >= 3:
            last_two = f"{words[-2].lower()} {last_word}"
            if last_two in POS_TAGS:
                return " ".join(words[:-2]), last_two

    return text, ""


def split_english_from_hiligaynon(text: str) -> tuple[str, str, str]:
    """
    Split a string that contains English word(s) followed by Hiligaynon translations.
    
    The format is:  english_word(s) [POS] hiligaynon_translations
    
    Returns (english, pos, hiligaynon_raw).
    
    Heuristic: 
    - If a POS tag is found (in parens or as a known word), it divides English from Hiligaynon
    - Otherwise, the first word is English and the rest is Hiligaynon
    - Multi-word English phrases (e.g., "action song") are detected via common patterns
    """
    text = text.strip()
    if not text:
        return "", "", ""

    # First try to find a parenthesized POS tag
    match = re.search(r"\(\s*([^)]+)\s*\)", text)
    if match:
        candidate = match.group(1).strip().lower()
        for pos in POS_TAGS:
            if candidate.startswith(pos) or pos in candidate:
                english = text[: match.start()].strip()
                hiligaynon = text[match.end():].strip().lstrip(",; ")
                return english, candidate, hiligaynon
    
    # Try to find an unparenthesized POS tag as a standalone word
    words = text.split()
    for i, word in enumerate(words):
        w_lower = word.lower().rstrip(",;:()")
        if w_lower in POS_TAGS and i > 0:
            english = " ".join(words[:i])
            hiligaynon = " ".join(words[i + 1:]).lstrip(",; ")
            # Check for two-word POS
            if i + 1 < len(words) and f"{w_lower} {words[i+1].lower().rstrip(',;:()')}" in POS_TAGS:
                hiligaynon = " ".join(words[i + 2:]).lstrip(",; ")
                return english, f"{w_lower} {words[i+1].lower().rstrip(',;:()')}", hiligaynon
            return english, w_lower, hiligaynon

    # No POS found — first word is English, rest is Hiligaynon
    if len(words) >= 2:
        return words[0], "", " ".join(words[1:])
    
    return text, "", ""


def parse_entry(line: str) -> dict | None:
    """
    Parse a single dictionary line into a structured entry.

    The dictionary has three main formats:
    1. english (POS)--hiligaynon--akeanon      (3 parts, two -- separators)
    2. english hiligaynon--akeanon              (2 parts, one -- separator)
    3. english hiligaynon                       (1 part, no -- separator)

    Returns a dict with keys:
        english, pos, hiligaynon, akeanon, raw_line
    or None if the line cannot be parsed.
    """
    raw_line = line

    # Normalize dashes: replace em-dash, en-dash with regular dash
    line = line.replace("\u2013", "-").replace("\u2014", "-")

    # The primary delimiter is "--" (double dash)
    parts = re.split(r"--", line)

    if len(parts) >= 3:
        # Standard 3-part format: english (POS)--hiligaynon--akeanon
        english_raw = parts[0].strip()
        hiligaynon_raw = parts[1].strip()
        akeanon_raw = "--".join(parts[2:]).strip()  # rejoin in case akeanon has --

        english, pos = extract_pos(english_raw)

        hiligaynon = parse_translations(hiligaynon_raw)
        akeanon = parse_translations(akeanon_raw)

        if english:
            return {
                "english": english.strip().lower(),
                "pos": pos,
                "hiligaynon": hiligaynon,
                "akeanon": akeanon,
                "raw_line": raw_line,
            }

    elif len(parts) == 2:
        # Two-part format: "english [POS] hiligaynon--akeanon"
        # parts[0] = English word(s) + possibly Hiligaynon translations
        # parts[1] = Akeanon translations
        combined = parts[0].strip()
        akeanon_raw = parts[1].strip()

        # Try to split English from Hiligaynon in parts[0]
        english, pos, hiligaynon_raw = split_english_from_hiligaynon(combined)

        hiligaynon = parse_translations(hiligaynon_raw) if hiligaynon_raw else []
        akeanon = parse_translations(akeanon_raw) if akeanon_raw else []

        if english:
            return {
                "english": english.strip().lower(),
                "pos": pos,
                "hiligaynon": hiligaynon,
                "akeanon": akeanon,
                "raw_line": raw_line,
            }

    elif len(parts) == 1:
        # No "--" delimiter - space-separated: "english hiligaynon"
        english, pos, hiligaynon_raw = split_english_from_hiligaynon(line)
        hiligaynon = parse_translations(hiligaynon_raw) if hiligaynon_raw else []

        if english:
            return {
                "english": english.strip().lower(),
                "pos": pos,
                "hiligaynon": hiligaynon,
                "akeanon": [],
                "raw_line": raw_line,
            }

    return None


def parse_translations(text: str) -> list[str]:
    """
    Parse a translation string into a list of individual translations.
    Translations can be separated by commas, semicolons, or slashes.
    Handles 'see also:' references.
    """
    if not text or text.strip() == "":
        return []

    # Remove 'see also:' prefixed notes but keep the referenced word
    text = re.sub(r"see\s+also\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"see\s+root\s+word\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"see\s+", "", text, flags=re.IGNORECASE)

    # Split on comma or semicolon
    parts = re.split(r"[;,]", text)

    translations = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 0:
            # Remove leading/trailing spaces and artifacts
            part = part.strip(" \t\n\r:")
            if part:
                translations.append(part)

    return translations


def parse_html_dictionary(html_path: str | Path) -> list[dict]:
    """
    Main function: Parse the HTML file and return a list of dictionary entries.
    """
    html_content = load_html(html_path)
    raw_lines = extract_raw_lines(html_content)

    entries = []
    skipped = []

    for line in raw_lines:
        entry = parse_entry(line)
        if entry:
            entries.append(entry)
        else:
            skipped.append(line)

    print(f"Parsed {len(entries)} entries, skipped {len(skipped)} lines")
    if skipped:
        print(f"Sample skipped lines:")
        for s in skipped[:5]:
            print(f"  - {s}")

    return entries


def save_json(entries: list[dict], output_path: str | Path):
    """Save entries to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")


def save_csv(entries: list[dict], output_path: str | Path):
    """Save entries to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["english", "pos", "hiligaynon", "akeanon", "raw_line"],
        )
        writer.writeheader()
        for entry in entries:
            row = entry.copy()
            row["hiligaynon"] = "; ".join(entry["hiligaynon"])
            row["akeanon"] = "; ".join(entry["akeanon"])
            writer.writerow(row)
    print(f"Saved {len(entries)} entries to {output_path}")


def create_trainset(
    entries: list[dict],
    output_path: str | Path,
    n_samples: int = 50,
    seed: int = 42,
):
    """
    Create a training set of high-quality examples for DSPy optimization.
    Selects entries that have translations in all three languages.
    """
    import random

    random.seed(seed)

    # Filter entries with all three languages present
    complete_entries = [
        e for e in entries if e["hiligaynon"] and e["akeanon"] and e["english"]
    ]

    # Sample
    n_samples = min(n_samples, len(complete_entries))
    sampled = random.sample(complete_entries, n_samples)

    # Convert to DSPy-compatible format
    trainset = []
    for entry in sampled:
        trainset.append(
            {
                "english": entry["english"],
                "pos": entry["pos"],
                "hiligaynon": entry["hiligaynon"][0],  # Primary translation
                "akeanon": entry["akeanon"][0],  # Primary translation
                "all_hiligaynon": entry["hiligaynon"],
                "all_akeanon": entry["akeanon"],
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trainset, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(trainset)} training examples to {output_path}")

    return trainset


def create_devset(
    entries: list[dict],
    output_path: str | Path,
    n_samples: int = 25,
    seed: int = 123,
):
    """Create a dev/eval set (different seed from trainset)."""
    return create_trainset(entries, output_path, n_samples, seed)


# --- CLI entrypoint ---
if __name__ == "__main__":
    import sys

    # Default paths
    project_root = Path(__file__).parent.parent
    html_path = project_root / "data" / "raw" / "english-hiligaynon-akeanon.html"

    # Fallback to root if not in data/raw yet
    if not html_path.exists():
        html_path = project_root / "english-hiligaynon-akeanon.html"

    if not html_path.exists():
        print(f"HTML file not found at {html_path}")
        sys.exit(1)

    print(f"Parsing {html_path}...")
    entries = parse_html_dictionary(html_path)

    # Save outputs
    processed_dir = project_root / "data" / "processed"
    save_json(entries, processed_dir / "dictionary.json")
    save_csv(entries, processed_dir / "dictionary.csv")

    # Create train/dev sets
    examples_dir = project_root / "data" / "examples"
    create_trainset(entries, examples_dir / "trainset.json", n_samples=50)
    create_devset(entries, examples_dir / "devset.json", n_samples=25)
