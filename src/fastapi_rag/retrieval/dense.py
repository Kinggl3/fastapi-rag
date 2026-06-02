from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    chunk_id: str
    score: float


class DenseRetriever:
    def __init__(self, index: faiss.Index, chunk_ids: list[str], model: SentenceTransformer):
        self._index = index
        self._ids = chunk_ids
        self._model = model

    @classmethod
    def load(cls, index_path: str | Path, model_name: str) -> DenseRetriever:
        index_path = Path(index_path)
        index = faiss.read_index(str(index_path))
        ids = np.load(str(index_path.with_suffix(".ids.npy")), allow_pickle=True).tolist()
        model = SentenceTransformer(model_name)
        return cls(index, ids, model)

    def search(self, query: str, k: int = 50) -> list[SearchResult]:
        vec = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(vec, k)
        return [
            SearchResult(chunk_id=self._ids[idx], score=float(scores[0][j]))
            for j, idx in enumerate(indices[0])
            if idx != -1
        ]


def build_index(
    chunk_ids: list[str],
    texts: list[str],
    model_name: str,
    index_path: str | Path,
    batch_size: int = 64,
) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    np.save(str(index_path.with_suffix(".ids.npy")), np.array(chunk_ids))
