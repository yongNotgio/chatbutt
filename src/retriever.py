"""
Dictionary retriever using ChromaDB for the Hiligaynon ↔ English chatbot.

Features:
- Semantic search via sentence-transformers embeddings
- Vowel-aware search: generates o↔u, i↔e variants for broader matching
- Supports both Hiligaynon→English and English→Hiligaynon lookups
"""

import json
import re
from pathlib import Path
import chromadb


COLLECTION_NAME = "hiligaynon_dictionary"


# ── Vowel-awareness helpers ──────────────────────────────────────────────────

def generate_vowel_variants(word: str) -> list[str]:
    """
    Generate vowel-swapped variants of a word.
    In Hiligaynon, o↔u and i↔e are interchangeable.

    Example:
        'buot'  → ['buot', 'boot', 'buut', 'bout']
        'diin'  → ['diin', 'deen', 'dien', 'dein']
    """
    word_lower = word.lower().strip()
    if not word_lower:
        return [word_lower]

    # Map of interchangeable vowels
    swaps = {"o": "u", "u": "o", "i": "e", "e": "i"}

    variants = {word_lower}

    # Generate all single-swap and multi-swap variants
    def _gen(chars: list, idx: int, current: list):
        if idx == len(chars):
            variants.add("".join(current))
            return
        _gen(chars, idx + 1, current + [chars[idx]])
        if chars[idx] in swaps:
            _gen(chars, idx + 1, current + [swaps[chars[idx]]])

    _gen(list(word_lower), 0, [])
    return list(variants)


def normalize_vowels(text: str) -> str:
    """Normalize vowels for comparison: o→u, e→i (canonical form)."""
    return text.lower().replace("o", "u").replace("e", "i")


# ── Index builder ────────────────────────────────────────────────────────────

def build_index(
    dictionary_path: str | Path,
    persist_dir: str | Path = "chroma_db",
) -> chromadb.Collection:
    """
    Build the ChromaDB index from the merged dictionary.

    Expects entries with {word, definition, source?}.
    """
    dictionary_path = Path(dictionary_path)
    persist_dir = Path(persist_dir)

    with open(dictionary_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Rebuild from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []

    for i, entry in enumerate(entries):
        word = entry.get("word", "").strip()
        defn = entry.get("definition", "").strip()

        if not word or not defn:
            continue

        # Build document text for embedding — includes word + definition
        # Also include the canonical vowel-normalized form for better matching
        normalized = normalize_vowels(word)
        doc = f"Hiligaynon: {word} | Definition: {defn}"

        documents.append(doc)
        metadatas.append({
            "word": word,
            "definition": defn[:500],  # ChromaDB metadata limit
            "normalized": normalized,
            "source": entry.get("source", ""),
        })
        ids.append(f"entry_{i}")

    # Add in batches
    batch_size = 5000
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Indexed {len(documents)} entries into ChromaDB at {persist_dir}")
    return collection


# ── Collection access ────────────────────────────────────────────────────────

def get_collection(persist_dir: str | Path = "chroma_db") -> chromadb.Collection:
    """Get the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=COLLECTION_NAME)


# ── Retrieval functions ──────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
    top_k: int = 10,
) -> list[str]:
    """
    Semantic search for the top-k most relevant dictionary entries.
    """
    if collection is None:
        collection = get_collection(persist_dir)

    results = collection.query(query_texts=[query], n_results=top_k)

    context = []
    if results and results["documents"]:
        for doc in results["documents"][0]:
            context.append(doc)
    return context


def retrieve_vowel_aware(
    word: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
    top_k: int = 10,
) -> list[str]:
    """
    Retrieve entries using vowel variants of the query word.
    Generates o↔u, i↔e variants and searches for all of them.
    Returns deduplicated results.
    """
    if collection is None:
        collection = get_collection(persist_dir)

    variants = generate_vowel_variants(word)
    seen = set()
    context = []

    # Search with each variant
    for variant in variants[:6]:  # limit to avoid too many queries
        results = collection.query(query_texts=[variant], n_results=top_k)
        if results and results["documents"]:
            for doc in results["documents"][0]:
                if doc not in seen:
                    seen.add(doc)
                    context.append(doc)

    # Also try normalized vowel search via metadata
    normalized = normalize_vowels(word)
    try:
        meta_results = collection.get(
            where={"normalized": normalized},
            limit=top_k,
        )
        if meta_results and meta_results["documents"]:
            for doc in meta_results["documents"]:
                if doc not in seen:
                    seen.add(doc)
                    context.append(doc)
    except Exception:
        pass  # metadata filter may fail if field not indexed

    return context[:top_k * 2]  # cap results


def retrieve_for_sentence(
    sentence: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
    top_k_per_word: int = 3,
) -> list[str]:
    """
    Retrieve entries for each significant word in a sentence.
    Uses vowel-aware retrieval for each word.
    """
    if collection is None:
        collection = get_collection(persist_dir)

    words = re.findall(r"\b[a-zA-ZáàâäéèêëíìîïóòôöúùûüñÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜÑ\-]+\b", sentence.lower())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "our", "their",
        "what", "how", "translate", "say", "mean", "means", "word",
        # Hiligaynon function words
        "ang", "sang", "sing", "sa", "si", "ni", "kay", "nga", "kag",
        "na", "pa", "man", "lang", "gid",
    }

    significant = [w for w in words if w not in stop_words and len(w) > 1]

    seen = set()
    context = []

    # First: semantic search on the full sentence
    full_results = retrieve(sentence, collection, persist_dir, top_k=5)
    for doc in full_results:
        if doc not in seen:
            seen.add(doc)
            context.append(doc)

    # Then: vowel-aware search per word
    for word in significant:
        results = retrieve_vowel_aware(word, collection, persist_dir, top_k=top_k_per_word)
        for doc in results:
            if doc not in seen:
                seen.add(doc)
                context.append(doc)

    return context


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    root = Path(__file__).parent.parent
    dict_path = root / "data" / "processed" / "dictionary.json"
    db_path = root / "chroma_db"

    if not dict_path.exists():
        print(f"Dictionary not found at {dict_path}. Run load_data.py first.")
        sys.exit(1)

    print("Building ChromaDB index...")
    coll = build_index(dict_path, db_path)

    # Test retrieval
    tests = ["maayong aga", "good morning", "love", "tubig", "water", "buut", "boot"]
    for q in tests:
        results = retrieve_vowel_aware(q, coll, db_path, top_k=3)
        print(f"\nQuery: '{q}'")
        for r in results:
            print(f"  → {r[:120]}")
