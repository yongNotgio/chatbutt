# English ↔ Hiligaynon ↔ Akeanon Translation Chatbot

A DSPy-powered translation chatbot for English, Hiligaynon (Ilonggo), and Akeanon (Aklanon) — Philippine languages from the Western Visayas region.

## Data Source

Dictionary data by **Melchor F. Cichon** from [How To Speak Aklanon The Easy Way](https://howtospeakaklanon.blogspot.com/2011/09/english-hiligaynon-akeanon.html).

## Features

- **Word translation**: Look up individual words across all three languages
- **Phrase/sentence translation**: Translate multi-word expressions and sentences
- **Grammar questions**: Ask about Hiligaynon/Akeanon grammar and usage
- **RAG-powered**: Uses a vector database of dictionary entries for accurate lookups
- **DSPy-optimized**: Prompt optimization for better translation quality

## Project Structure

```
chatbutt/
├── data/
│   ├── raw/                          # Source HTML file
│   ├── processed/                    # Parsed JSON/CSV dictionary
│   └── examples/                     # Train/dev sets for DSPy
├── src/
│   ├── parse_html.py                 # HTML → structured data parser
│   ├── signatures.py                 # DSPy Signature definitions
│   ├── modules.py                    # DSPy Module (pipeline)
│   ├── metrics.py                    # Evaluation metrics
│   ├── retriever.py                  # ChromaDB retriever
│   └── optimize.py                   # DSPy optimizer configs
├── app/
│   └── gradio_app.py                 # Gradio chatbot UI
├── notebooks/
│   ├── 01_parse_data.ipynb           # Data parsing & exploration
│   ├── 02_prototype.ipynb            # Pipeline prototyping
│   └── 03_optimize.ipynb             # DSPy optimization
├── optimized/                        # Saved optimized programs
├── .env                              # API keys (not committed)
├── .env.example                      # Template for .env
├── requirements.txt                  # Python dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```powershell
cd chatbutt
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Key

Copy `.env.example` to `.env` and add your API key:

```bash
# Option A: OpenAI (recommended)
OPENAI_API_KEY=sk-your-key-here

# Option B: Ollama (free, local)
# Install from https://ollama.com, then: ollama pull llama3.2
```

### 3. Parse the Dictionary Data

Run the notebooks in order, or use the CLI:

```powershell
python -m src.parse_html
```

Or run the notebook: `notebooks/01_parse_data.ipynb`

### 4. Prototype & Test

Run `notebooks/02_prototype.ipynb` to test the pipeline interactively.

### 5. Optimize

Run `notebooks/03_optimize.ipynb` to optimize with DSPy.

### 6. Launch the Chatbot

```powershell
python app/gradio_app.py
```

Open http://localhost:7860 in your browser.

## How It Works

```
User Query
    │
    ▼
┌──────────────┐
│ ClassifyQuery │  ← Determine: word / phrase / sentence / grammar
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Retriever   │  ← Find relevant dictionary entries (ChromaDB)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ TranslateWord /  │  ← Generate translation using LLM + context
│ TranslatePhrase  │
└──────┬───────────┘
       │
       ▼
  Formatted Response
  (English / Hiligaynon / Akeanon)
```

## DSPy Optimization Options

| Optimizer | When to Use | Requirements |
|-----------|------------|--------------|
| `BootstrapFewShot` | 10-50 examples (start here) | API key |
| `MIPROv2` | 200+ examples | API key, patience |
| `BootstrapFinetune` | Full fine-tuning | GPU, torch, transformers |

## Languages

- **Hiligaynon** (Ilonggo): ~9 million speakers, major language of Western Visayas
- **Akeanon** (Aklanon): ~500k speakers, spoken in Aklan province, Philippines
- Both are Austronesian languages in the Visayan family

## License

Dictionary data is attributed to Melchor F. Cichon. Code is for educational purposes.
