# FastAPI RAG Assistant

> Production-grade RAG assistant for FastAPI developers. Hybrid retrieval over official docs, GitHub issues, and Stack Overflow with reranking and LLM-based generation with citations.

## Status

🚧 **Work in progress.** This README will be updated as the project evolves.

## Motivation

FastAPI has extensive documentation, but answering specific developer questions often requires consulting multiple sources: official docs (for canonical patterns), GitHub issues (for edge cases and bugs), and Stack Overflow (for community wisdom and troubleshooting). This project unifies all three into a single retrieval-augmented assistant.

## Architecture
Query
↓
[Retrieval, parallel]
├─ BM25     → top-50
└─ Dense    → top-50
↓
[Reciprocal Rank Fusion]  → top-50 merged
↓
[Cross-encoder Reranker]  → top-10
↓
[LLM Generation] → answer with cited sources

## Stack

| Component | Choice |
|---|---|
| Python | 3.11+ |
| Package manager | uv |
| Embeddings | BAAI/bge-small-en-v1.5 (baseline), bge-large-en-v1.5 (final) |
| Sparse retrieval | bm25s |
| Vector store | FAISS + SQLite for metadata |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Google Gemini 2.5 (Flash for dev, Pro for demo) |
| Web demo | Streamlit |
| Evaluation | Custom harness + RAGAS |

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd fastapi-rag
uv sync
```

Configuration (LLM keys etc.) goes in `.env` — see `.env.example` for the template.

## Usage

_Coming soon — CLI under development._

## Project structure
src/fastapi_rag/
├── corpus/        # Source loaders and chunking
├── retrieval/     # Dense, BM25, hybrid, reranking
├── generation/    # LLM-based RAG generation
└── eval/          # Benchmark harness and metrics
configs/           # YAML configurations per pipeline variant
data/              # Raw corpora, processed chunks, indexes (gitignored)
eval_data/         # Ground truth query→relevant pairs
tests/             # Unit tests
notebooks/         # Exploration and ad-hoc experiments
app/               # Streamlit demo

## Roadmap

- [ ] Document corpus loader + header-aware chunking
- [ ] Dense retriever (FAISS + sentence-transformers)
- [ ] CLI for querying
- [ ] Manual ground truth (30-50 queries)
- [ ] Evaluation harness (MRR, NDCG, Recall@k, latency)
- [ ] BM25 retriever
- [ ] Hybrid retriever (RRF)
- [ ] GitHub issues loader
- [ ] Stack Overflow loader
- [ ] Cross-encoder reranker
- [ ] LLM generation with citations
- [ ] Synthetic ground truth + RAGAS evaluation
- [ ] Streamlit demo
- [ ] HuggingFace Spaces deployment

## License

MIT