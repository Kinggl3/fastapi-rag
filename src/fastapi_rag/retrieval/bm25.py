from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import bm25s
import Stemmer

from .dense import SearchResult


def _stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer("english")


def build_bm25(
    chunk_ids: list[str],
    texts: list[str],
    index_path: str | Path,
) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    stemmer = _stemmer()
    corpus_tokens = bm25s.tokenize(texts, stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    retriever.save(str(index_path))
    with open(index_path.with_suffix(".ids.pkl"), "wb") as f:
        pickle.dump(chunk_ids, f)


class BM25Retriever:
    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str]):
        self._retriever = retriever
        self._ids = chunk_ids
        self._stemmer = _stemmer()

    @classmethod
    def load(cls, index_path: str | Path) -> BM25Retriever:
        index_path = Path(index_path)
        retriever = bm25s.BM25.load(str(index_path), load_corpus=False)
        with open(index_path.with_suffix(".ids.pkl"), "rb") as f:
            chunk_ids = pickle.load(f)
        return cls(retriever, chunk_ids)

    def search(self, query: str, k: int = 50) -> list[SearchResult]:
        tokens = bm25s.tokenize([query], stemmer=self._stemmer)
        results, scores = self._retriever.retrieve(tokens, k=min(k, len(self._ids)))
        return [
            SearchResult(chunk_id=self._ids[results[0][i]], score=float(scores[0][i]))
            for i in range(results.shape[1])
        ]
