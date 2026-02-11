"""
DSPy Modules for the English-Hiligaynon-Akeanon translation chatbot.

The main module is TranslationChatbot which:
1. Classifies the user query type
2. Retrieves relevant dictionary entries
3. Generates translation using the appropriate signature
"""

import dspy
from pathlib import Path

from src.signatures import (
    ClassifyQuery,
    TranslateWord,
    TranslatePhrase,
    AnswerGrammarQuestion,
)
from src.retriever import retrieve, retrieve_for_sentence, get_collection


class TranslationChatbot(dspy.Module):
    """
    Main translation chatbot module.

    Pipeline:
        User Query -> Classify -> Retrieve Context -> Translate -> Format Response
    """

    def __init__(self, chroma_dir: str | Path = "chroma_db"):
        super().__init__()

        # Sub-modules
        self.classify = dspy.Predict(ClassifyQuery)
        self.translate_word = dspy.ChainOfThought(TranslateWord)
        self.translate_phrase = dspy.ChainOfThought(TranslatePhrase)
        self.answer_grammar = dspy.ChainOfThought(AnswerGrammarQuestion)

        # Retriever
        self.chroma_dir = str(chroma_dir)
        self._collection = None

    @property
    def collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            self._collection = get_collection(self.chroma_dir)
        return self._collection

    def forward(self, user_query: str = None, english: str = None, pos: str = None, **kwargs) -> dspy.Prediction:
        """
        Process a user query and return a translation response.

        Accepts either:
        - user_query: a natural language chat message (e.g., "Translate 'love'")
        - english + pos: direct word lookup (used by DSPy evaluator with Example objects)
        """
        # Handle DSPy evaluator passing Example fields directly
        if user_query is None and english is not None:
            user_query = f"Translate '{english}'"
            # Skip classification — we know it's a word lookup
            query_type = "word"
            extracted_text = english
            context = retrieve(
                extracted_text,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k=10,
            )
            result = self.translate_word(
                english_word=extracted_text,
                part_of_speech=pos or "",
                dictionary_context=context,
            )
            return dspy.Prediction(
                query_type=query_type,
                extracted_text=extracted_text,
                hiligaynon=result.hiligaynon,
                akeanon=result.akeanon,
                notes=getattr(result, "notes", ""),
                context_used=context,
                reasoning=getattr(result, "reasoning", ""),
            )
        # Step 1: Classify the query
        classification = self.classify(user_query=user_query)
        query_type = classification.query_type
        extracted_text = classification.extracted_text

        # Step 2: Retrieve relevant dictionary context
        if query_type in ("word",):
            context = retrieve(
                extracted_text,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k=10,
            )
        else:
            context = retrieve_for_sentence(
                extracted_text,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k_per_word=3,
            )

        # Step 3: Generate translation based on query type
        if query_type == "word":
            result = self.translate_word(
                english_word=extracted_text,
                part_of_speech="",  # Let the model determine from context
                dictionary_context=context,
            )
            return dspy.Prediction(
                query_type=query_type,
                extracted_text=extracted_text,
                hiligaynon=result.hiligaynon,
                akeanon=result.akeanon,
                notes=result.notes,
                context_used=context,
                reasoning=getattr(result, "reasoning", ""),
            )

        elif query_type in ("phrase", "sentence"):
            result = self.translate_phrase(
                english_text=extracted_text,
                dictionary_context=context,
            )
            return dspy.Prediction(
                query_type=query_type,
                extracted_text=extracted_text,
                hiligaynon=result.hiligaynon,
                akeanon=result.akeanon,
                literal_breakdown=result.literal_breakdown,
                notes=result.notes,
                context_used=context,
                reasoning=getattr(result, "reasoning", ""),
            )

        elif query_type == "grammar":
            result = self.answer_grammar(
                question=extracted_text,
                dictionary_context=context,
            )
            return dspy.Prediction(
                query_type=query_type,
                extracted_text=extracted_text,
                answer=result.answer,
                examples=result.examples,
                context_used=context,
                reasoning=getattr(result, "reasoning", ""),
            )

        else:
            # Fallback: treat as phrase
            result = self.translate_phrase(
                english_text=extracted_text,
                dictionary_context=context,
            )
            return dspy.Prediction(
                query_type="phrase",
                extracted_text=extracted_text,
                hiligaynon=result.hiligaynon,
                akeanon=result.akeanon,
                literal_breakdown=result.literal_breakdown,
                notes=result.notes,
                context_used=context,
                reasoning=getattr(result, "reasoning", ""),
            )


def format_response(prediction: dspy.Prediction) -> str:
    """
    Format a DSPy prediction into a human-readable chat response.
    """
    lines = []
    query_type = prediction.query_type

    if query_type in ("word", "phrase", "sentence"):
        lines.append(f"**English:** {prediction.extracted_text}")
        lines.append(f"**Hiligaynon:** {prediction.hiligaynon}")
        lines.append(f"**Akeanon:** {prediction.akeanon}")

        if hasattr(prediction, "literal_breakdown") and prediction.literal_breakdown:
            lines.append(f"\n**Breakdown:** {prediction.literal_breakdown}")

        if hasattr(prediction, "notes") and prediction.notes:
            lines.append(f"\n**Notes:** {prediction.notes}")

    elif query_type == "grammar":
        lines.append(prediction.answer)
        if hasattr(prediction, "examples") and prediction.examples:
            lines.append(f"\n**Examples:** {prediction.examples}")

    return "\n".join(lines)
