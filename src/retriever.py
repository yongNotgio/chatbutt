"""
Dictionary retriever using ChromaDB for the translation chatbot.

Embeds all dictionary entries into a vector store and provides
semantic search for relevant context during translation.
"""

import json
from pathlib import Path
import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "dictionary_entries"


def build_index(
    dictionary_path: str | Path,
    persist_dir: str | Path = "chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> chromadb.Collection:
    """
    Build (or rebuild) the ChromaDB index from the parsed dictionary.

    Args:
        dictionary_path: Path to dictionary.json
        persist_dir: Directory to persist the ChromaDB data
        embedding_model: Sentence-transformers model name for embeddings
    """
    dictionary_path = Path(dictionary_path)
    persist_dir = Path(persist_dir)

    with open(dictionary_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Create ChromaDB client with persistence
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection if it exists (rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create collection with default embedding function
    # ChromaDB will use sentence-transformers automatically
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare documents and metadata
    documents = []
    metadatas = []
    ids = []

    for i, entry in enumerate(entries):
        # Create a rich document string for embedding
        doc_parts = [f"English: {entry['english']}"]
        if entry.get("pos"):
            doc_parts.append(f"({entry['pos']})")
        if entry.get("hiligaynon"):
            hil = (
                ", ".join(entry["hiligaynon"])
                if isinstance(entry["hiligaynon"], list)
                else entry["hiligaynon"]
            )
            doc_parts.append(f"Hiligaynon: {hil}")
        if entry.get("akeanon"):
            ake = (
                ", ".join(entry["akeanon"])
                if isinstance(entry["akeanon"], list)
                else entry["akeanon"]
            )
            doc_parts.append(f"Akeanon: {ake}")

        doc = " | ".join(doc_parts)
        documents.append(doc)

        metadatas.append(
            {
                "english": entry["english"],
                "pos": entry.get("pos", ""),
                "hiligaynon": (
                    "; ".join(entry["hiligaynon"])
                    if isinstance(entry["hiligaynon"], list)
                    else entry.get("hiligaynon", "")
                ),
                "akeanon": (
                    "; ".join(entry["akeanon"])
                    if isinstance(entry["akeanon"], list)
                    else entry.get("akeanon", "")
                ),
            }
        )
        ids.append(f"entry_{i}")

    # Add in batches (ChromaDB has a limit of ~41666 per batch)
    batch_size = 5000
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Indexed {len(documents)} dictionary entries into ChromaDB at {persist_dir}")
    return collection


def get_collection(persist_dir: str | Path = "chroma_db") -> chromadb.Collection:
    """Get the existing ChromaDB collection."""
    persist_dir = Path(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    query: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
    top_k: int = 10,
) -> list[str]:
    """
    Retrieve the top-k most relevant dictionary entries for a query.

    Args:
        query: The search query (English word, phrase, or sentence)
        collection: Optional pre-loaded ChromaDB collection
        persist_dir: ChromaDB persistence directory
        top_k: Number of results to return

    Returns:
        List of formatted dictionary entry strings for use as context
    """
    if collection is None:
        collection = get_collection(persist_dir)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    # Format results as context strings
    context_entries = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_entries.append(doc)

    return context_entries


def retrieve_exact(
    english_word: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
) -> list[dict]:
    """
    Try to find an exact match for an English word in the dictionary.

    Returns list of matching metadata dicts.
    """
    if collection is None:
        collection = get_collection(persist_dir)

    # Use metadata filtering for exact match
    results = collection.get(
        where={"english": english_word.lower()},
    )

    matches = []
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            matches.append(meta)

    return matches


def retrieve_for_sentence(
    sentence: str,
    collection: chromadb.Collection | None = None,
    persist_dir: str | Path = "chroma_db",
    top_k_per_word: int = 3,
) -> list[str]:
    """
    Retrieve dictionary entries for each significant word in a sentence.
    Deduplicates and returns a combined context list.
    """
    if collection is None:
        collection = get_collection(persist_dir)

    # Simple tokenization - split on spaces and remove punctuation
    import re

    words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())

    # Filter out very common English words that likely aren't in the dictionary
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "our", "their",
    }

    significant_words = [w for w in words if w not in stop_words and len(w) > 1]

    # Retrieve for each word
    seen_docs = set()
    context = []

    for word in significant_words:
        results = retrieve(word, collection, persist_dir, top_k=top_k_per_word)
        for doc in results:
            if doc not in seen_docs:
                seen_docs.add(doc)
                context.append(doc)

    return context


# --- CLI ---
if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    dict_path = project_root / "data" / "processed" / "dictionary.json"
    db_path = project_root / "chroma_db"

    if not dict_path.exists():
        print(f"Dictionary not found at {dict_path}. Run parse_html.py first.")
        sys.exit(1)

    print("Building ChromaDB index...")
    collection = build_index(dict_path, db_path)

    # Test retrieval
    test_queries = ["hello", "good morning", "love", "eat", "water"]
    for q in test_queries:
        results = retrieve(q, collection, db_path, top_k=3)
        print(f"\nQuery: '{q}'")
        for r in results:
            print(f"  -> {r}")
