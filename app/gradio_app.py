"""
Gradio chatbot interface for the English ↔ Hiligaynon translator.

Conversational, context-aware, and vowel-structure-aware.
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

from src.modules import HiligaynonChatbot
from src.optimize import setup_lm, load_program


# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv(project_root / ".env")

chatbot_instance: HiligaynonChatbot | None = None


def initialize():
    """Initialize the LM and chatbot."""
    global chatbot_instance

    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "gsk_your-key-here":
        setup_lm(provider="groq", model="llama-3.1-8b-instant", api_key=api_key)
    else:
        print("No Groq API key. Attempting Ollama (local)...")
        setup_lm(provider="ollama", model="llama3.2")

    # Load optimized program if available
    optimized_path = project_root / "optimized" / "hiligaynon_v1.json"
    chroma_dir = str(project_root / "chroma_db")

    if optimized_path.exists():
        print(f"Loading optimized program from {optimized_path}")
        chatbot_instance = load_program(optimized_path, chroma_dir=chroma_dir)
    else:
        print("No optimized program found. Using base chatbot.")
        chatbot_instance = HiligaynonChatbot(chroma_dir=chroma_dir)


def chat(message: str, history: list) -> str:
    """Handle a chat message."""
    global chatbot_instance

    if chatbot_instance is None:
        return "Error: Chatbot not initialized. Please restart the app."

    if not message.strip():
        return "Please type a word, phrase, or sentence in English or Hiligaynon."

    try:
        prediction = chatbot_instance(user_message=message)
        return prediction.response
    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}\n\nPlease try again."


# ── Gradio UI ────────────────────────────────────────────────────────────────

TITLE = "🇵🇭 English ↔ Hiligaynon Chatbot"

DESCRIPTION = """
**Translate between English and Hiligaynon (Ilonggo) — conversationally.**

This chatbot understands Hiligaynon word structure, including vowel interchangeability
(**o ↔ u**, **i ↔ e**). Type in either language and it will translate, explain, and
help you learn.

**Try:**
- "What does *maayo* mean?"
- "How do you say *beautiful* in Hiligaynon?"
- "Translate: I love you"
- "diin ka gakadto?" (Where are you going?)
- "Explain how Hiligaynon verb affixes work"
"""

EXAMPLES = [

    "What is 'water' in Hiligaynon?",
    "Translate: The food is delicious",
]


def create_app():
    demo = gr.ChatInterface(
        fn=chat,
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
    )
    return demo


if __name__ == "__main__":
    print("Initializing chatbot...")
    initialize()
    print("Starting Gradio app...")
    app = create_app()
    app.launch(share=False)
