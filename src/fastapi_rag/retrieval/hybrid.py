from __future__ import annotations

from .dense import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_n: int = 50,
) -> list[SearchResult]:
    """Merge multiple ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(chunk_id=cid, score=score) for cid, score in merged[:top_n]]
