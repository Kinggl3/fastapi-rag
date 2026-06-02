# FastAPI RAG Assistant

> Production-grade RAG assistant for FastAPI developers — hybrid retrieval over official docs, GitHub issues, and Stack Overflow with cross-encoder reranking and LLM generation with citations.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What it does

Ask any FastAPI question and get a grounded answer with cited sources — pulled from official documentation and real GitHub issues.

```
$ python -m fastapi_rag.cli query "how to handle CORS in production"

[Answer with inline citations and source URLs]
```

---

## Architecture

```
Query
  │
  ├── Dense retrieval (FAISS + bge-small-en-v1.5)  ──┐
  └── Sparse retrieval (BM25s + stemming)            ├── top-50 each
                                                     │
                        Reciprocal Rank Fusion  ◄────┘  → top-50 merged
                                   │
                    Cross-encoder Reranker (bge-reranker-v2-m3)  → top-10
                                   │
                     Gemini 2.5 Flash (RAG prompt + citations)
                                   │
                              Answer + Sources
```

---

## Stack

| Component | Choice |
|---|---|
| Python | 3.11+ · uv |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse retrieval | bm25s + PyStemmer |
| Vector store | FAISS (IndexFlatIP) + SQLite metadata |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Google Gemini 2.5 Flash |
| Web demo | Streamlit |

---

## Corpus

| Source | Chunks | Content |
|---|---|---|
| FastAPI official docs | 1 373 | All tutorial + advanced pages |
| GitHub issues (`tiangolo/fastapi`) | 614 | Closed question/bug issues |
| **Total** | **1 987** | |

---

## Retrieval Metrics

Evaluated on 25 hand-crafted queries covering the full FastAPI topic range.

| Metric | Hybrid (no rerank) |
|---|---|
| MRR@10 | 0.59 |
| Recall@10 | 0.92 |
| Hit Rate@10 | 0.92 |
| Avg latency | 39 ms |

---

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Kinggl3/fastapi-rag
cd fastapi-rag
uv sync
cp .env.example .env   # fill in API keys
```

**`.env` keys:**

| Key | Where to get |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) — scope: `public_repo` |
| `HTTPS_PROXY` | Optional — if Gemini API is geo-blocked in your region |

---

## Usage

### 1. Build the corpus

```bash
# Fetch FastAPI docs from GitHub (~150 pages)
python -m fastapi_rag.cli ingest

# Fetch closed GitHub issues (question + bug labels)
python -m fastapi_rag.cli ingest-github
```

### 2. Build indexes

```bash
# Encodes 1987 chunks with bge-small-en-v1.5 + builds BM25 index
python -m fastapi_rag.cli build-index
```

### 3. Ask questions

```bash
# Full pipeline: hybrid retrieval → reranker → Gemini answer
python -m fastapi_rag.cli query "how to add JWT authentication"

# Retrieval only (no LLM call)
python -m fastapi_rag.cli query "how to use background tasks" --no-generate
```

### 4. Streamlit demo

```bash
python -m streamlit run app/app.py
```

### 5. Evaluate retrieval

```bash
python -m fastapi_rag.cli eval --no-rerank
```

---

## Project structure

```
src/fastapi_rag/
├── config.py           # pydantic-settings config
├── cli.py              # Typer CLI
├── corpus/             # Loaders, chunker, SQLite store
│   ├── loader.py       # FastAPI docs via GitHub API
│   ├── github.py       # GitHub issues loader
│   ├── chunker.py      # Header-aware markdown chunking
│   └── store.py        # SQLite CRUD
├── retrieval/          # Retrieval pipeline
│   ├── dense.py        # FAISS + sentence-transformers
│   ├── bm25.py         # bm25s sparse retrieval
│   ├── hybrid.py       # Reciprocal Rank Fusion
│   └── reranker.py     # CrossEncoder reranker
├── generation/         # LLM generation
│   └── generator.py    # Gemini with citation prompt
└── eval/               # Evaluation harness
    ├── harness.py      # Runs queries, collects results
    └── metrics.py      # MRR, Recall@k, Hit Rate
app/
└── app.py              # Streamlit demo
eval_data/
└── ground_truth.jsonl  # 25 hand-crafted test queries
```

---

## License

MIT
