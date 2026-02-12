"""
DSPy Module for the English ↔ Hiligaynon conversational chatbot.

Pipeline:
    User Message → Analyze Input → Retrieve Context (vowel-aware) → Generate Response

The chatbot:
1. Analyzes the input language, intent, and key text
2. Retrieves relevant dictionary entries with vowel-variant awareness
3. Generates a natural conversational response (not structured output)
"""

import dspy
from pathlib import Path

from src.signatures import (
    AnalyzeInput,
    TranslateToHiligaynon,
    TranslateToEnglish,
    ExplainGrammar,
    ConversationalResponse,
)
from src.retriever import (
    retrieve,
    retrieve_vowel_aware,
    retrieve_for_sentence,
    get_collection,
)


class HiligaynonChatbot(dspy.Module):
    """
    Conversational English ↔ Hiligaynon chatbot.

    Analyzes input before responding. Context-aware and vowel-structure-aware.
    """

    def __init__(self, chroma_dir: str | Path = "chroma_db"):
        super().__init__()

        # Sub-modules
        self.analyze = dspy.ChainOfThought(AnalyzeInput)
        self.translate_to_hil = dspy.ChainOfThought(TranslateToHiligaynon)
        self.translate_to_eng = dspy.ChainOfThought(TranslateToEnglish)
        self.explain_grammar = dspy.ChainOfThought(ExplainGrammar)
        self.general_chat = dspy.ChainOfThought(ConversationalResponse)

        self.chroma_dir = str(chroma_dir)
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_collection(self.chroma_dir)
        return self._collection

    def forward(
        self,
        user_message: str = None,
        # Aliases for DSPy evaluator compatibility
        hiligaynon: str = None,
        english: str = None,
        **kwargs,
    ) -> dspy.Prediction:
        """
        Process a user message and return a conversational response.

        Accepts:
        - user_message: natural language chat input
        - hiligaynon + english: direct lookup (for DSPy evaluator)
        """

        # Handle DSPy evaluator passing fields directly
        if user_message is None:
            if hiligaynon:
                user_message = f"What does '{hiligaynon}' mean in English?"
            elif english:
                user_message = f"How do you say '{english}' in Hiligaynon?"
            else:
                user_message = "Hello"

        # ── Step 1: Analyze the input ────────────────────────────────────
        analysis = self.analyze(user_message=user_message)
        input_lang = analysis.input_language
        intent = analysis.intent
        key_text = analysis.key_text
        analysis_text = analysis.analysis

        # ── Step 2: Retrieve context (vowel-aware) ──────────────────────
        if len(key_text.split()) <= 2:
            # Single word or short phrase — use vowel-aware retrieval
            context = retrieve_vowel_aware(
                key_text,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k=10,
            )
        else:
            # Longer text — per-word retrieval
            context = retrieve_for_sentence(
                key_text,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k_per_word=3,
            )

        # Also add context for the full message if different from key_text
        if key_text.lower() != user_message.lower():
            extra = retrieve(
                user_message,
                collection=self.collection,
                persist_dir=self.chroma_dir,
                top_k=5,
            )
            seen = set(context)
            for doc in extra:
                if doc not in seen:
                    context.append(doc)

        # ── Step 3: Generate response based on intent & language ────────

        if intent == "grammar":
            result = self.explain_grammar(
                question=key_text,
                dictionary_context=context,
            )
        elif intent == "chat":
            result = self.general_chat(
                user_message=user_message,
                dictionary_context=context,
            )
        elif input_lang == "hiligaynon":
            # Hiligaynon → English
            result = self.translate_to_eng(
                hiligaynon_text=key_text,
                dictionary_context=context,
                input_analysis=analysis_text,
            )
        elif input_lang == "english" or intent in ("translate", "define"):
            # English → Hiligaynon
            result = self.translate_to_hil(
                english_text=key_text,
                dictionary_context=context,
                input_analysis=analysis_text,
            )
        elif input_lang == "mixed":
            # Mixed — default to translating and explaining both directions
            result = self.translate_to_hil(
                english_text=key_text,
                dictionary_context=context,
                input_analysis=analysis_text,
            )
        else:
            # Unclear — try to be helpful
            result = self.general_chat(
                user_message=user_message,
                dictionary_context=context,
            )

        return dspy.Prediction(
            response=result.response,
            input_language=input_lang,
            intent=intent,
            key_text=key_text,
            analysis=analysis_text,
            context_used=context,
            reasoning=getattr(result, "reasoning", ""),
        )
