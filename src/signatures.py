"""
DSPy Signatures for the English ↔ Hiligaynon conversational chatbot.

Focuses on:
- Bidirectional translation (English → Hiligaynon and Hiligaynon → English)
- Input analysis before responding
- Vowel-aware matching (o/u, i/e interchangeability)
- Natural conversational output
"""

import dspy
from typing import Literal


class AnalyzeInput(dspy.Signature):
    """Analyze the user's input to determine what they want and what language it is in.

    Hiligaynon (Ilonggo) is a Visayan language of the Western Visayas in the Philippines.
    It has vowel interchangeability: 'o' and 'u' are often used interchangeably,
    as are 'i' and 'e'. For example, 'buot' and 'boot' mean the same thing;
    'diin' and 'deen' are the same word.

    Determine:
    1. The input language (English, Hiligaynon, or mixed/unclear)
    2. The intent (translate, ask about meaning, ask about grammar/usage, general chat)
    3. The key text to work with
    """

    user_message: str = dspy.InputField(desc="The user's chat message")

    input_language: Literal["english", "hiligaynon", "mixed", "unclear"] = dspy.OutputField(
        desc="Detected language of the user's input"
    )
    intent: Literal["translate", "define", "grammar", "chat"] = dspy.OutputField(
        desc="What the user wants: translate text, define a word, ask about grammar, or general chat"
    )
    key_text: str = dspy.OutputField(
        desc="The key word/phrase/sentence to translate or explain, extracted from the message"
    )
    analysis: str = dspy.OutputField(
        desc="Brief analysis of the input: language features noticed, any vowel variants, word structure notes"
    )


class TranslateToHiligaynon(dspy.Signature):
    """Translate English text into Hiligaynon and respond conversationally.

    Hiligaynon is a VSO (Verb-Subject-Object) language with a focus/trigger system.
    Key grammar notes:
    - Vowels o/u and i/e are interchangeable (both forms are correct)
    - Uses 'ang' (nominative), 'sang/sing' (genitive), 'sa' (dative) markers
    - Verb affixes indicate focus: mag- (actor), -on (object), i- (benefactive), -an (locative)
    - Common greetings: 'Maayong aga' (good morning), 'Maayong hapon' (good afternoon)

    Use the dictionary context to ground translations in real Hiligaynon words.
    Respond naturally — explain the translation, mention alternatives if they exist,
    and note any interesting linguistic features.
    """

    english_text: str = dspy.InputField(desc="The English text to translate")
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries for grounding the translation"
    )
    input_analysis: str = dspy.InputField(
        desc="Analysis of the input from the analysis step"
    )

    response: str = dspy.OutputField(
        desc="A natural, conversational response that includes the Hiligaynon translation, "
        "explains the translation, mentions vowel variants if applicable, and provides "
        "usage context. Do NOT use a rigid structured format."
    )


class TranslateToEnglish(dspy.Signature):
    """Translate Hiligaynon text into English and respond conversationally.

    Hiligaynon has vowel interchangeability: o/u and i/e can be swapped.
    When looking up words, try both vowel variants.
    Example: 'buut' = 'boot' = 'buot' (mind/will/desire).

    Use the dictionary context to find accurate definitions.
    Respond naturally — give the English meaning, explain nuances,
    mention related words, and note any vowel variants.
    """

    hiligaynon_text: str = dspy.InputField(desc="The Hiligaynon text to translate")
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries for grounding the translation"
    )
    input_analysis: str = dspy.InputField(
        desc="Analysis of the input from the analysis step"
    )

    response: str = dspy.OutputField(
        desc="A natural, conversational response that includes the English translation, "
        "explains meaning and usage, mentions vowel variants (o/u, i/e) if relevant, "
        "and provides cultural or linguistic context when helpful."
    )


class ExplainGrammar(dspy.Signature):
    """Answer a question about Hiligaynon grammar, usage, or word structure.

    Hiligaynon linguistics notes:
    - Vowel interchangeability: o↔u, i↔e (same meaning, regional preference)
    - Rich verb morphology: affixes mag-, nag-, ga-, -on, -an, i-, etc.
    - Focus/trigger system (not simply active/passive)
    - Reduplication for plurals, intensity, or continuity
    - Ligatures: nga (or -ng after vowels) connecting modifiers

    Respond conversationally with clear explanations and examples.
    """

    question: str = dspy.InputField(desc="The grammar or usage question")
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries for examples"
    )

    response: str = dspy.OutputField(
        desc="A clear, conversational explanation with examples from the dictionary. "
        "Mention vowel variants where relevant."
    )


class ConversationalResponse(dspy.Signature):
    """Generate a helpful conversational response for general chat about Hiligaynon.

    The user might be asking about the language in general, about culture,
    or making a request that doesn't fit neatly into translation or grammar.
    Be helpful, friendly, and knowledgeable about Hiligaynon.
    """

    user_message: str = dspy.InputField(desc="The user's message")
    dictionary_context: list[str] = dspy.InputField(
        desc="Any relevant dictionary entries"
    )

    response: str = dspy.OutputField(
        desc="A helpful, conversational response about Hiligaynon language or culture"
    )
