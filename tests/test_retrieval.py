"""Retrieval pipeline tests using the dependency-free HashingEmbedder + real LanceDB."""

import numpy as np

from variantscribe.agent.evidence import _variant_query, retrieval_evidence_fn
from variantscribe.models import Variant
from variantscribe.retrieval.embedder import HashingEmbedder
from variantscribe.retrieval.index import LanceCorpusIndex, Passage
from variantscribe.retrieval.reranker import NoOpReranker
from variantscribe.retrieval.retriever import Retriever


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    a = emb.embed_documents(["BRCA1 pathogenic frameshift variant"])
    b = emb.embed_documents(["BRCA1 pathogenic frameshift variant"])
    assert np.allclose(a, b)  # stable across calls (crc32, not salted hash)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def _build_index(tmp_path):
    passages = [
        Passage(id="1", gene="BRCA1", pmid="1",
                text="BRCA1 c.68_69del frameshift pathogenic loss of function"),
        Passage(id="2", gene="BRCA1", pmid="2",
                text="BRCA1 common benign polymorphism high population frequency"),
        Passage(id="3", gene="TP53", pmid="3",
                text="TP53 hotspot mutation in Li-Fraumeni syndrome"),
    ]
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed_documents([p.text for p in passages])
    index = LanceCorpusIndex(tmp_path / "db", table="corpus_BRCA1")
    index.build(passages, vecs)
    return emb, index


def test_index_build_and_gene_filtered_search(tmp_path):
    emb, index = _build_index(tmp_path)
    assert index.count() == 3
    qv = emb.embed_queries(["frameshift loss of function pathogenic"])[0]
    hits = index.search(qv, k=5, gene="BRCA1")
    assert len(hits) == 2  # TP53 passage filtered out
    assert all(h["gene"] == "BRCA1" for h in hits)


def test_retriever_returns_evidence_items_ranked(tmp_path):
    emb, index = _build_index(tmp_path)
    retriever = Retriever(emb, index, NoOpReranker(), k_dense=10, k_final=2)
    items = retriever.retrieve("frameshift pathogenic loss of function", gene="BRCA1")
    assert len(items) == 2
    assert items[0].source == "pubmed"
    # the pathogenic-frameshift passage should rank above the benign one
    assert "frameshift" in items[0].text
    assert items[0].score is not None


def test_retrieval_evidence_fn_builds_query_and_filters_gene(tmp_path):
    emb, index = _build_index(tmp_path)
    retriever = Retriever(emb, index, NoOpReranker(), k_final=2)
    fn = retrieval_evidence_fn(retriever)
    items = fn(Variant(gene="BRCA1", variation_id="x", name="NM_007294.4(BRCA1):c.68_69del"))
    assert items and all(it.kind == "literature" for it in items)


def test_variant_query_mentions_gene_and_change():
    q = _variant_query(Variant(gene="BRCA1", variation_id="x", hgvs_p="p.Gln1756fs"))
    assert "BRCA1" in q and "p.Gln1756fs" in q
