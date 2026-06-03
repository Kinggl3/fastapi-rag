"""Streamlit demo for the FastAPI RAG Assistant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FastAPI RAG Assistant",
    page_icon="⚡",
    layout="centered",
)

# ── Paths & constants ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DB = _ROOT / "data" / "corpus.db"
_FAISS = _ROOT / "data" / "indexes" / "docs.faiss"
_BM25 = _ROOT / "data" / "indexes" / "docs.bm25"
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_LLM_MODEL = "gemini-2.5-flash"


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dense index...")
def load_dense():
    from fastapi_rag.retrieval.dense import DenseRetriever
    return DenseRetriever.load(_FAISS, _EMBED_MODEL)


@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    from fastapi_rag.retrieval.bm25 import BM25Retriever
    return BM25Retriever.load(_BM25)


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    from fastapi_rag.retrieval.reranker import Reranker
    return Reranker(_RERANKER_MODEL)


@st.cache_resource(show_spinner="Loading corpus...")
def load_corpus():
    from fastapi_rag.corpus.store import load_chunks
    return {c.chunk_id: c for c in load_chunks(_DB)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ FastAPI RAG")
    st.caption("Hybrid retrieval over FastAPI docs")
    st.divider()

    reranker_k = st.slider("Results sent to LLM", min_value=3, max_value=15, value=8)
    use_reranker = st.toggle("Use reranker", value=True)
    st.divider()
    st.markdown(
        "**Sources:** [FastAPI docs](https://fastapi.tiangolo.com)  \n"
        "**Models:** bge-small-en-v1.5 · bge-reranker-v2-m3  \n"
        "**LLM:** Gemini 2.5 Flash"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("FastAPI RAG Assistant")
st.caption("Ask anything about FastAPI — answers grounded in the official documentation.")

indexes_ready = _FAISS.exists() and _BM25.exists() and _DB.exists()

if not indexes_ready:
    st.error(
        "Indexes not found. Run these commands first:\n\n"
        "```\npython -m fastapi_rag.cli ingest\n"
        "python -m fastapi_rag.cli build-index\n```"
    )
    st.stop()

question = st.chat_input("Ask a question about FastAPI...")

if question:
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            from fastapi_rag.retrieval.hybrid import reciprocal_rank_fusion

            dense = load_dense()
            bm25 = load_bm25()
            corpus = load_corpus()

            dense_results = dense.search(question, k=50)
            bm25_results = bm25.search(question, k=50)
            hybrid = reciprocal_rank_fusion([dense_results, bm25_results], top_n=50)

        if use_reranker:
            with st.spinner("Reranking..."):
                reranker = load_reranker()
                chunk_texts = {cid: c.content for cid, c in corpus.items()}
                final = reranker.rerank(question, hybrid, chunk_texts, top_k=reranker_k)
        else:
            final = hybrid[:reranker_k]

        top_chunks = [corpus[r.chunk_id] for r in final if r.chunk_id in corpus]

        with st.spinner("Generating answer..."):
            from google.genai.errors import ServerError
            from fastapi_rag.config import settings
            from fastapi_rag.generation.generator import generate_answer

            if not settings.google_api_key:
                st.error("GOOGLE_API_KEY not set in .env")
                st.stop()

            try:
                result = generate_answer(
                    question, top_chunks,
                    model=_LLM_MODEL,
                    api_key=settings.google_api_key,
                )
            except ServerError as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.warning(
                        "Gemini is overloaded right now. Try again in a moment.",
                        icon="⏳",
                    )
                else:
                    st.error(f"Gemini error: {e}")
                st.stop()

        st.markdown(result.answer)

        with st.expander(f"📚 Sources ({len(top_chunks)} chunks retrieved)", expanded=False):
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**[{i}] [{chunk.breadcrumb}]({chunk.url})**")
                st.caption(chunk.content[:300] + "...")
                if i < len(top_chunks):
                    st.divider()
