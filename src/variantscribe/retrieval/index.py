"""LanceDB-backed vector index for the literature corpus.

LanceDB is file-based (no server) — the right Phase-1 choice on a box without Docker.
The VectorStore interface is deliberately small so swapping in pgvector later is mechanical;
that swap is the planned "scaled the data layer" milestone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel


class Passage(BaseModel):
    """One indexed unit of literature evidence (here: one PubMed article)."""

    id: str
    gene: str
    pmid: str | None = None
    title: str = ""
    text: str = ""
    citation: str | None = None


class LanceCorpusIndex:
    def __init__(self, path: Path, table: str = "corpus") -> None:
        import lancedb

        self._db = lancedb.connect(str(path))
        self._table_name = table

    def build(self, passages: list[Passage], vectors: np.ndarray) -> int:
        """(Re)build the index from passages and their document embeddings."""
        if len(passages) != len(vectors):
            raise ValueError("passages and vectors length mismatch")
        rows = [
            {
                "vector": vec.astype("float32").tolist(),
                "id": p.id,
                "gene": p.gene,
                "pmid": p.pmid or "",
                "title": p.title,
                "text": p.text,
                "citation": p.citation or "",
            }
            for p, vec in zip(passages, vectors, strict=True)
        ]
        self._db.create_table(self._table_name, data=rows, mode="overwrite")
        return len(rows)

    def search(
        self, query_vector: np.ndarray, k: int, *, gene: str | None = None
    ) -> list[dict]:
        """Vector search, optionally filtered to a gene (the hybrid-search hook)."""
        tbl = self._db.open_table(self._table_name)
        q = tbl.search(query_vector.astype("float32").tolist()).limit(k)
        if gene:
            q = q.where(f"gene = '{gene}'")
        return q.to_list()

    def count(self) -> int:
        return self._db.open_table(self._table_name).count_rows()
