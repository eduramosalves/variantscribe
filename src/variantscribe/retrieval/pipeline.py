"""Build the corpus index for a gene and reconstruct retrievers from saved metadata.

A small sidecar JSON records which embedder built each index, so `evaluate` rebuilds a
matching query encoder (an index embedded by MedCPT must be queried by MedCPT).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from variantscribe.config import settings
from variantscribe.ingest.eutils import EutilsClient
from variantscribe.ingest.pubmed import gene_literature_query, search_articles
from variantscribe.retrieval.embedder import Embedder, HashingEmbedder, MedCPTEmbedder
from variantscribe.retrieval.index import LanceCorpusIndex, Passage
from variantscribe.retrieval.reranker import MedCPTReranker, Reranker
from variantscribe.retrieval.retriever import Retriever

_DB_DIR = "lancedb"


def _table_name(gene: str) -> str:
    return f"corpus_{gene.upper()}"


def _meta_path(gene: str) -> Path:
    return settings.index_dir / f"meta_{gene.upper()}.json"


def _db_path() -> Path:
    return settings.index_dir / _DB_DIR


def make_embedder(name: str, *, dim: int = 256) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim=dim)
    if name == "medcpt":
        return MedCPTEmbedder(settings.embedding_model, settings.query_embedding_model)
    raise ValueError(f"unknown embedder {name!r} (use 'hashing' or 'medcpt')")


def make_reranker(enabled: bool) -> Reranker | None:
    if not enabled:
        return None
    return MedCPTReranker(settings.reranker_model)


def corpus_passages(
    gene: str, *, max_articles: int = 400, client: EutilsClient | None = None
) -> list[Passage]:
    articles = search_articles(
        gene_literature_query(gene), retmax=max_articles, client=client
    )
    return [
        Passage(
            id=a.pmid,
            gene=gene.upper(),
            pmid=a.pmid,
            title=a.title,
            text=f"{a.title}\n{a.abstract}".strip(),
            citation=a.citation,
        )
        for a in articles
    ]


def build_index(gene: str, *, embedder_name: str = "medcpt", max_articles: int = 400) -> dict:
    """Fetch literature, embed it, and (re)build the LanceDB index. Returns metadata."""
    settings.ensure_dirs()
    passages = corpus_passages(gene, max_articles=max_articles)
    if not passages:
        raise RuntimeError(f"No PubMed articles found for {gene}.")

    embedder = make_embedder(embedder_name)
    t0 = time.monotonic()
    vectors = embedder.embed_documents([p.text for p in passages])
    embed_s = time.monotonic() - t0

    index = LanceCorpusIndex(_db_path(), table=_table_name(gene))
    n = index.build(passages, vectors)

    meta = {
        "gene": gene.upper(),
        "embedder": embedder.name,
        "dim": int(embedder.dim),
        "n_passages": n,
        "article_model": settings.embedding_model if embedder.name == "medcpt" else None,
        "query_model": settings.query_embedding_model if embedder.name == "medcpt" else None,
        "embed_seconds": round(embed_s, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _meta_path(gene).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_meta(gene: str) -> dict:
    path = _meta_path(gene)
    if not path.exists():
        raise FileNotFoundError(
            f"No index for {gene} at {path}. Run `build-index --gene {gene}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_retriever(
    gene: str, *, rerank: bool = True, k_dense: int = 20, k_final: int = 5
) -> Retriever:
    meta = load_meta(gene)
    embedder = make_embedder(meta["embedder"], dim=meta["dim"])
    index = LanceCorpusIndex(_db_path(), table=_table_name(gene))
    reranker = make_reranker(rerank)  # None when rerank is False
    return Retriever(embedder, index, reranker, k_dense=k_dense, k_final=k_final)
