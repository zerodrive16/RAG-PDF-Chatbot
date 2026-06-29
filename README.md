# PDF Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and ask questions about it. Built with Streamlit, Qdrant, and the OpenAI API.

## How it works

1. Upload a PDF → the text is extracted and split into overlapping chunks
2. Each chunk is embedded via OpenAI's `text-embedding-3-large` model and stored in a Qdrant vector database
3. When you ask a question, the most relevant chunks are retrieved and passed to `gpt-4o-mini` as context
4. The model answers based only on what's in the document

## Requirements

- Python 3.12+
- Docker (for Qdrant)
- An OpenAI API key

## Setup

**1. Clone the repo and install dependencies**

```bash
# using uv (recommended)
uv sync

# or pip
pip install -e .
```

**2. Create a `.env` file**

```env
OPENAI_API_KEY=your-openai-api-key
```

**3. Start Qdrant**

```bash
docker compose up -d
```

**4. Run the app**

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project structure

```
.
├── app.py              # Streamlit UI
├── main.py             # PDF ingestion, embedding, vector search, chat logic
├── docker-compose.yml  # Qdrant vector database
├── pyproject.toml      # Dependencies
└── .env                # API keys (not committed)
```

## Configuration

Key constants in `main.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `EMBED_DIM` | `3072` | Vector dimension |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `COLLECTION_NAME` | `documents` | Qdrant collection name |

## Stack

- [Streamlit](https://streamlit.io) — UI
- [Qdrant](https://qdrant.tech) — vector database
- [OpenAI API](https://platform.openai.com) — embeddings + chat
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF text extraction
