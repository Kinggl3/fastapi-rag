from __future__ import annotations

from sentence_transformers import CrossEncoder

from .dense import SearchResult


class Reranker:
    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        chunk_texts: dict[str, str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        pairs = [(query, chunk_texts[r.chunk_id]) for r in results if r.chunk_id in chunk_texts]
        if not pairs:
            return results[:top_k]

        scores = self._model.predict(pairs)
        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [SearchResult(chunk_id=r.chunk_id, score=float(s)) for r, s in reranked[:top_k]]
