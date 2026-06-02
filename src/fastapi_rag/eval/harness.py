from __future__ import annotations

import json
import time
from pathlib import Path

from ..corpus.store import load_chunks
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever
from ..retrieval.hybrid import reciprocal_rank_fusion
from .metrics import EvalReport, QueryResult

_GT_PATH = Path("eval_data/ground_truth.jsonl")


def load_ground_truth(path: str | Path = _GT_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(
    db_path: str | Path,
    faiss_path: str | Path,
    bm25_path: str | Path,
    embed_model: str,
    gt_path: str | Path = _GT_PATH,
    retrieval_k: int = 50,
    eval_k: int = 10,
    reranker_model: str | None = None,
) -> EvalReport:
    ground_truth = load_ground_truth(gt_path)
    all_chunks = {c.chunk_id: c for c in load_chunks(db_path)}

    dense = DenseRetriever.load(faiss_path, embed_model)
    bm25 = BM25Retriever.load(bm25_path)

    reranker = None
    if reranker_model:
        from ..retrieval.reranker import Reranker
        reranker = Reranker(reranker_model)

    from tqdm import tqdm
    report = EvalReport()
    for item in tqdm(ground_truth, desc="Evaluating", unit="query"):
        query = item["query"]
        relevant_urls = item["relevant_urls"]

        t0 = time.perf_counter()

        dense_res = dense.search(query, k=retrieval_k)
        bm25_res = bm25.search(query, k=retrieval_k)
        hybrid = reciprocal_rank_fusion([dense_res, bm25_res], top_n=retrieval_k)

        if reranker:
            chunk_texts = {cid: c.content for cid, c in all_chunks.items()}
            final = reranker.rerank(query, hybrid, chunk_texts, top_k=eval_k)
        else:
            final = hybrid[:eval_k]

        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_urls = [
            all_chunks[r.chunk_id].url
            for r in final
            if r.chunk_id in all_chunks
        ]

        report.results.append(QueryResult(
            query=query,
            relevant_urls=relevant_urls,
            retrieved_urls=retrieved_urls,
            latency_ms=latency_ms,
        ))

    return report
