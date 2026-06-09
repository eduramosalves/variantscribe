"""Cross-encoder reranking of retrieved passages.

Dense retrieval is recall-oriented; a cross-encoder that reads (query, passage) jointly is
far more precise about which passages actually answer the query. This is the Text Ranking
task from the project's HuggingFace task list.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage (higher = more relevant)."""
        ...


class NoOpReranker:
    """Identity reranker — preserves the dense-retrieval order. The ablation baseline
    that the cross-encoder is measured against."""

    name = "none"

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        # Descending scores so a stable sort keeps the incoming (dense) order.
        return [float(len(passages) - i) for i in range(len(passages))]


class MedCPTReranker:
    """NCBI MedCPT cross-encoder (query, article) relevance scorer via transformers."""

    name = "medcpt-cross-encoder"

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            pairs = [[query, p] for p in batch]
            enc = self._tok(
                pairs,
                truncation=True,
                padding=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                logits = self._model(**enc).logits.squeeze(dim=1)
            scores.extend(logits.cpu().tolist())
        return scores
