"""
DSPy Signatures for the English-Hiligaynon-Akeanon translation chatbot.

Signatures are declarative specs that tell DSPy what the input/output
behavior of each module should be, without specifying how to achieve it.
"""

import dspy
from typing import Literal


class ClassifyQuery(dspy.Signature):
    """Classify a user's translation query into a type and extract the text to translate.

    Query types:
    - 'word': single word translation lookup
    - 'phrase': multi-word phrase or expression translation
    - 'sentence': full sentence translation
    - 'grammar': grammar or usage question about Hiligaynon or Akeanon
    """

    user_query: str = dspy.InputField(desc="The user's input message")
    query_type: Literal["word", "phrase", "sentence", "grammar"] = dspy.OutputField(
        desc="The type of translation query"
    )
    extracted_text: str = dspy.OutputField(
        desc="The English text to be translated, extracted from the query"
    )


class TranslateWord(dspy.Signature):
    """Translate a single English word into Hiligaynon and Akeanon (Aklanon).

    Use the provided dictionary context to find accurate translations.
    Hiligaynon is a major Philippine language spoken in Western Visayas.
    Akeanon (Aklanon) is a related language spoken in Aklan province.
    If the word is NOT found in the dictionary context, use your linguistic
    knowledge to provide the best possible translation, and note the uncertainty.
    """

    english_word: str = dspy.InputField(desc="The English word to translate")
    part_of_speech: str = dspy.InputField(
        desc="Part of speech (noun, verb, adjective, etc.), if known"
    )
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries retrieved for context"
    )
    hiligaynon: str = dspy.OutputField(
        desc="Translation(s) in Hiligaynon, separated by commas if multiple"
    )
    akeanon: str = dspy.OutputField(
        desc="Translation(s) in Akeanon, separated by commas if multiple"
    )
    notes: str = dspy.OutputField(
        desc="Brief usage notes or confidence level"
    )


class TranslatePhrase(dspy.Signature):
    """Translate an English phrase or sentence into Hiligaynon and Akeanon (Aklanon).

    Use the dictionary context as a reference for individual words, then compose
    the phrase/sentence translation considering the grammar and word order of each
    target language. Hiligaynon and Akeanon are VSO (Verb-Subject-Object) languages
    with focus/trigger systems.
    """

    english_text: str = dspy.InputField(
        desc="The English phrase or sentence to translate"
    )
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries for words in the phrase"
    )
    hiligaynon: str = dspy.OutputField(desc="Translation in Hiligaynon")
    akeanon: str = dspy.OutputField(desc="Translation in Akeanon")
    literal_breakdown: str = dspy.OutputField(
        desc="Word-by-word breakdown showing how the translation was constructed"
    )
    notes: str = dspy.OutputField(
        desc="Grammar notes or other relevant information"
    )


class AnswerGrammarQuestion(dspy.Signature):
    """Answer a question about Hiligaynon or Akeanon grammar, usage, or culture.

    Use the dictionary context and your knowledge of Philippine linguistics
    to provide an informative answer.
    """

    question: str = dspy.InputField(desc="The user's grammar or usage question")
    dictionary_context: list[str] = dspy.InputField(
        desc="Relevant dictionary entries for context"
    )
    answer: str = dspy.OutputField(
        desc="A clear, informative answer about the grammar or usage"
    )
    examples: str = dspy.OutputField(
        desc="Example words or phrases illustrating the grammar point"
    )
