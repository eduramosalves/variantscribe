"""Embedders for the retrieval index.

Two implementations behind one Protocol:

* HashingEmbedder — dependency-free, deterministic char-n-gram hashing. Weak but real;
  lets the full retrieval pipeline + tests run without torch or model downloads.
* MedCPTEmbedder — the production path: NCBI MedCPT asymmetric article/query encoders
  (PubMedBERT-based, CLS pooling), loaded via transformers.

Both expose embed_documents / embed_queries returning L2-normalised float32 arrays, so
cosine similarity reduces to a dot product and the index is comparable across embedders.
"""

from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic char-n-gram hashing embedder. Uses crc32 (not Python's salted
    hash()) so vectors are stable across processes — essential for build-then-query."""

    name = "hashing"

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (3, 4, 5)) -> None:
        self.dim = dim
        self.ngrams = ngrams

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        t = text.lower()
        for n in self.ngrams:
            for i in range(len(t) - n + 1):
                bucket = zlib.crc32(t[i : i + n].encode("utf-8")) % self.dim
                vec[bucket] += 1.0
        return vec

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._embed_one(t) for t in texts]))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


class MedCPTEmbedder:
    """NCBI MedCPT article/query encoders via transformers (CLS-pooled, normalised)."""

    name = "medcpt"

    def __init__(
        self,
        article_model: str,
        query_model: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_doc_tokens: int = 512,
        max_query_tokens: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_doc_tokens = max_doc_tokens
        self.max_query_tokens = max_query_tokens

        self._art_tok = AutoTokenizer.from_pretrained(article_model)
        self._art_model = AutoModel.from_pretrained(article_model).to(self.device).eval()
        self._qry_tok = AutoTokenizer.from_pretrained(query_model)
        self._qry_model = AutoModel.from_pretrained(query_model).to(self.device).eval()
        self.dim = int(self._art_model.config.hidden_size)

    def _encode(self, texts, tokenizer, model, max_len) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                cls = model(**enc).last_hidden_state[:, 0, :]  # CLS token
            out.append(cls.cpu().numpy())
        return _l2_normalize(np.vstack(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._art_tok, self._art_model, self.max_doc_tokens)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._qry_tok, self._qry_model, self.max_query_tokens)
