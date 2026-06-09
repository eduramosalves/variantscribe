"""Guideline-PDF retrieval: index ACMG/VCEP guideline pages and retrieve relevant ones.

Two interchangeable backends behind one retriever interface:

* text    — page text-layer → HashingEmbedder → LanceDB (reuses the Phase-1 index infra).
            Dependency-free; works on any machine.
* colpali — page image → ColPali multi-vector → MaxSim late interaction. Reads tables and
            figures the text layer misses. Needs the `colpali` (torch) extra.

No copyrighted guideline PDFs are committed; point `--pdf-dir` at PDFs you hold a licence to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from variantscribe.config import settings
from variantscribe.models import EvidenceItem, Variant
from variantscribe.retrieval.embedder import HashingEmbedder
from variantscribe.retrieval.index import LanceCorpusIndex, Passage
from variantscribe.retrieval.pdf import load_pdf_dir
from variantscribe.retrieval.retriever import Retriever

_GL_TABLE = "guidelines"
_DB_DIR = "lancedb"
_GUIDELINE_GENE = "_GUIDELINE"  # sentinel; guideline pages are not gene-filtered


def _db_path() -> Path:
    return settings.index_dir / _DB_DIR


def _meta_path() -> Path:
    return settings.index_dir / "meta_guidelines.json"


def _colpali_path() -> Path:
    return settings.index_dir / "guidelines_colpali.npz"


def build_guideline_index(
    pdf_dir: str | Path, *, embedder: str = "text", render_dpi: int = 120
) -> dict:
    """Index every page of every PDF in `pdf_dir`. Returns metadata."""
    settings.ensure_dirs()
    use_colpali = embedder == "colpali"
    pages = load_pdf_dir(pdf_dir, with_images=use_colpali, render_dpi=render_dpi)
    pages = [p for p in pages if (p.image_png if use_colpali else p.text)]
    if not pages:
        raise RuntimeError(f"No usable pages found in {pdf_dir}.")

    if use_colpali:
        from variantscribe.retrieval.colpali import ColPaliPageEmbedder, MaxSimIndex

        enc = ColPaliPageEmbedder(settings.colpali_model)
        vecs = enc.embed_images([p.image_png for p in pages])
        idx = MaxSimIndex()
        for p, v in zip(pages, vecs, strict=True):
            idx.add(
                {"doc": p.doc, "page_no": p.page_no, "text": p.text[:1500],
                 "citation": p.citation},
                v,
            )
        idx.save(_colpali_path())
        dim = int(vecs[0].shape[-1]) if vecs else 0
    else:
        he = HashingEmbedder(dim=256)
        passages = [
            Passage(
                id=f"{p.doc}-{p.page_no}",
                gene=_GUIDELINE_GENE,
                title=p.doc,
                text=p.text,
                citation=p.citation,
            )
            for p in pages
        ]
        vectors = he.embed_documents([p.text for p in passages])
        LanceCorpusIndex(_db_path(), table=_GL_TABLE).build(passages, vectors)
        dim = he.dim

    meta = {
        "embedder": embedder,
        "n_pages": len(pages),
        "dim": dim,
        "docs": sorted({p.doc for p in pages}),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class ColPaliGuidelineRetriever:
    def __init__(self, encoder, index, *, k_final: int = 3) -> None:
        self.encoder = encoder
        self.index = index
        self.k_final = k_final

    def retrieve(self, query: str, *, gene: str | None = None) -> list[EvidenceItem]:
        qv = self.encoder.embed_queries([query])[0]
        return [
            EvidenceItem(
                source="guideline",
                kind="guideline",
                text=m.get("text", ""),
                citation=m.get("citation"),
                score=score,
            )
            for m, score in self.index.search(qv, self.k_final)
        ]


def load_guideline_retriever(*, k_final: int = 3):
    path = _meta_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No guideline index at {path}. Run `build-guidelines --pdf-dir ...` first."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["embedder"] == "colpali":
        from variantscribe.retrieval.colpali import ColPaliPageEmbedder, MaxSimIndex

        enc = ColPaliPageEmbedder(settings.colpali_model)
        return ColPaliGuidelineRetriever(enc, MaxSimIndex.load(_colpali_path()), k_final=k_final)

    he = HashingEmbedder(dim=meta["dim"])
    index = LanceCorpusIndex(_db_path(), table=_GL_TABLE)
    return Retriever(
        he, index, None, k_final=k_final, source="guideline", kind="guideline"
    )


def guideline_query(variant: Variant) -> str:
    change = variant.hgvs_p or variant.hgvs_c or variant.name or ""
    return (
        f"ACMG/AMP criteria and classification rules relevant to a {variant.gene} variant "
        f"{change} — applicable PVS1/PS/PM/PP and benign criteria."
    )


def guideline_evidence_fn(retriever):
    def fn(variant: Variant) -> list[EvidenceItem]:
        # Guidelines are not gene-partitioned; search the whole corpus.
        return retriever.retrieve(guideline_query(variant), gene=None)

    return fn
