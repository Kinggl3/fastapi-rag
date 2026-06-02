from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    query: str
    relevant_urls: list[str]
    retrieved_urls: list[str]   # ordered, top-k
    latency_ms: float

    @property
    def reciprocal_rank(self) -> float:
        relevant = set(self.relevant_urls)
        for rank, url in enumerate(self.retrieved_urls, 1):
            if url in relevant:
                return 1.0 / rank
        return 0.0

    def recall_at(self, k: int) -> float:
        relevant = set(self.relevant_urls)
        found = {url for url in self.retrieved_urls[:k] if url in relevant}
        return len(found) / len(relevant) if relevant else 0.0

    def hit_at(self, k: int) -> float:
        relevant = set(self.relevant_urls)
        return float(any(url in relevant for url in self.retrieved_urls[:k]))


@dataclass
class EvalReport:
    results: list[QueryResult] = field(default_factory=list)

    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def mean_recall(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at(k) for r in self.results) / len(self.results)

    def hit_rate(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit_at(k) for r in self.results) / len(self.results)

    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
