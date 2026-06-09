"""Phase-1 evidence providers fed to the classifier.

Deliberately simple: a naive per-variant PubMed search. Phase 2 replaces this with the
retrieval index (embed + rerank over a pre-built corpus); the classifier interface is the
same, so swapping providers is the experiment that measures retrieval's contribution.
"""

from __future__ import annotations

from collections.abc import Callable

from variantscribe.ingest.eutils import EutilsClient
from variantscribe.ingest.pubmed import search_articles
from variantscribe.models import EvidenceItem, Variant

EvidenceFn = Callable[[Variant], list[EvidenceItem]]


def _variant_search_terms(variant: Variant) -> str:
    # Try to include the protein/cDNA change so the search is specific to the variant.
    bits = [f"{variant.gene}[Title/Abstract]"]
    change = variant.hgvs_p or variant.hgvs_c
    if not change and variant.name and ":" in variant.name:
        change = variant.name.split(":", 1)[1].strip()
    if change:
        bits.append(f'"{change}"[Title/Abstract]')
    return " AND ".join(bits)


def pubmed_evidence(
    variant: Variant,
    *,
    max_items: int = 5,
    client: EutilsClient | None = None,
) -> list[EvidenceItem]:
    """Best-effort literature evidence for a variant from PubMed abstracts."""
    articles = search_articles(
        _variant_search_terms(variant), retmax=max_items, client=client
    )
    items: list[EvidenceItem] = []
    for art in articles:
        snippet = art.abstract or art.title
        items.append(
            EvidenceItem(
                source="pubmed",
                kind="literature",
                text=f"{art.title} — {snippet}"[:1200],
                citation=art.citation,
            )
        )
    return items


def _variant_query(variant: Variant) -> str:
    """A natural-language retrieval query describing the variant and the question."""
    change = variant.hgvs_p or variant.hgvs_c or variant.name or ""
    return (
        f"Clinical significance and pathogenicity of {variant.gene} variant {change}: "
        "functional studies, segregation, and population frequency evidence."
    )


def retrieval_evidence_fn(retriever) -> EvidenceFn:
    """Adapt a Retriever into an evidence provider for the classifier (gene-filtered)."""

    def fn(variant: Variant) -> list[EvidenceItem]:
        return retriever.retrieve(_variant_query(variant), gene=variant.gene.upper())

    return fn


def combined_evidence_fn(*fns: EvidenceFn | None) -> EvidenceFn:
    """Concatenate evidence from several providers (e.g. literature + guideline pages)."""
    active = [f for f in fns if f is not None]

    def fn(variant: Variant) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for f in active:
            items.extend(f(variant))
        return items

    return fn
