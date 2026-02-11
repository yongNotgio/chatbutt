"""
Gradio chatbot interface for the English-Hiligaynon-Akeanon translator.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import dspy
from dotenv import load_dotenv

from src.modules import TranslationChatbot, format_response
from src.optimize import setup_lm, load_program


# --- Setup ---
load_dotenv(project_root / ".env")

# Global chatbot instance
chatbot_instance: TranslationChatbot | None = None


def initialize():
    """Initialize the LM and chatbot."""
    global chatbot_instance

    # Configure LM
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "gsk_your-key-here":
        setup_lm(provider="groq", model="llama-3.3-70b-versatile", api_key=api_key)
    else:
        print("No Groq API key. Attempting Ollama (local)...")
        setup_lm(provider="ollama", model="llama3.2")

    # Load optimized program if available, otherwise use base
    optimized_path = project_root / "optimized" / "translation_v1.json"
    chroma_dir = str(project_root / "chroma_db")

    if optimized_path.exists():
        print(f"Loading optimized program from {optimized_path}")
        chatbot_instance = load_program(optimized_path, chroma_dir=chroma_dir)
    else:
        print("No optimized program found. Using base chatbot.")
        chatbot_instance = TranslationChatbot(chroma_dir=chroma_dir)


def translate(message: str, history: list) -> str:
    """
    Handle a chat message and return the translation.
    """
    global chatbot_instance

    if chatbot_instance is None:
        return "Error: Chatbot not initialized. Please restart the app."

    if not message.strip():
        return "Please enter an English word, phrase, or sentence to translate."

    try:
        prediction = chatbot_instance(user_query=message)
        response = format_response(prediction)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your query."


# --- Gradio UI ---
TITLE = "🌏 English ↔ Hiligaynon ↔ Akeanon Translator"

DESCRIPTION = """
**Translate English words, phrases, and sentences into Hiligaynon and Akeanon (Aklanon).**

Hiligaynon (Ilonggo) is a major language of the Western Visayas region in the Philippines.
Akeanon (Aklanon) is spoken in Aklan province, Philippines.

Dictionary data by **Melchor F. Cichon** from [How To Speak Aklanon The Easy Way](https://howtospeakaklanon.blogspot.com/).

**Try asking:**
- "Translate 'good morning'"
- "How do you say 'I love you' in Akeanon?"
- "What is the Hiligaynon word for water?"
- "Translate: The house is beautiful"
"""

EXAMPLES = [
    "Translate 'hello'",
    "How do you say 'thank you' in Akeanon?",
    "What is 'water' in Hiligaynon?",
    "Translate: I love you",
    "How do you say 'good morning'?",
    "Translate: The food is delicious",
    "What is the word for 'friend'?",
    "Translate: Where are you going?",
]


def create_app():
    """Create and return the Gradio app."""
    demo = gr.ChatInterface(
        fn=translate,
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
    )
    return demo


if __name__ == "__main__":
    print("Initializing translation chatbot...")
    initialize()
    print("Starting Gradio app...")
    app = create_app()
    app.launch(share=False)
