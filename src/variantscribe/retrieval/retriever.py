"""Two-stage retriever: dense vector search, then optional cross-encoder rerank."""

from __future__ import annotations

import numpy as np

from variantscribe.models import EvidenceItem
from variantscribe.retrieval.embedder import Embedder
from variantscribe.retrieval.index import LanceCorpusIndex
from variantscribe.retrieval.reranker import Reranker


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index: LanceCorpusIndex,
        reranker: Reranker | None = None,
        *,
        k_dense: int = 20,
        k_final: int = 5,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.k_dense = k_dense
        self.k_final = k_final

    def retrieve(self, query: str, *, gene: str | None = None) -> list[EvidenceItem]:
        qvec = self.embedder.embed_queries([query])[0]
        hits = self.index.search(qvec, self.k_dense, gene=gene)
        if not hits:
            return []

        if self.reranker is not None:
            scores = self.reranker.rerank(query, [h["text"] for h in hits])
            order = np.argsort(scores)[::-1]
            hits = [{**hits[i], "_rerank": float(scores[i])} for i in order]

        top = hits[:self.k_final]
        items: list[EvidenceItem] = []
        for h in top:
            # Prefer rerank score; else cosine sim (1 - lance distance on normalised vecs).
            if "_rerank" in h:
                score = h["_rerank"]
            elif "_distance" in h:
                score = 1.0 - float(h["_distance"])
            else:
                score = None
            items.append(
                EvidenceItem(
                    source="pubmed",
                    kind="literature",
                    text=h["text"],
                    citation=h.get("citation") or None,
                    score=score,
                )
            )
        return items
