"""ClinVar ingestion: fetch variants for a gene with their germline classification.

ClinVar is doubly useful here: it supplies the variants to interpret *and* the gold
labels (plus a review-status "star" rating that tells us how much to trust each label).
"""

from __future__ import annotations

import logging

import httpx

from variantscribe.ingest.eutils import EutilsClient
from variantscribe.models import ClinSig, GoldRecord, Variant

logger = logging.getLogger(__name__)

# ClinVar review-status -> gold-star rating (the stars shown in the ClinVar UI).
REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classifications from unflagged records": 0,
}


def review_status_to_stars(status: str) -> int:
    return REVIEW_STATUS_STARS.get(status.strip().lower(), 0)


# The ClinVar [Review Status] filter phrases, by star rating. We query per tier and
# merge, because a single OR of two quoted phrases trips ClinVar's query parser (500).
STARS_TO_REVIEW_PHRASE: dict[int, str] = {
    4: "practice guideline",
    3: "reviewed by expert panel",
    2: "criteria provided, multiple submitters, no conflicts",
}


def _extract_classification(doc: dict) -> tuple[str | None, str | None]:
    """Pull (significance_description, review_status) from a ClinVar esummary doc,
    tolerating the schema variants NCBI has shipped over the years."""
    # Current schema: germline_classification: {description, review_status}
    gc = doc.get("germline_classification") or {}
    if gc.get("description"):
        return gc.get("description"), gc.get("review_status")
    # Legacy schema: clinical_significance: {description, review_status}
    cs = doc.get("clinical_significance") or {}
    if cs.get("description"):
        return cs.get("description"), cs.get("review_status")
    return None, None


def _variant_from_doc(gene: str, uid: str, doc: dict) -> Variant:
    # variation_set carries HGVS / canonical name; fall back to top-level title.
    name = doc.get("title")
    hgvs_c = None
    vset = doc.get("variation_set") or []
    if vset:
        v0 = vset[0]
        name = v0.get("variation_name") or name
        hgvs_c = v0.get("cdna_change") or None
    return Variant(gene=gene, variation_id=str(uid), name=name, hgvs_c=hgvs_c)


def _esearch_all(client: EutilsClient, term: str, max_results: int) -> list[str]:
    """Page through esearch up to max_results UIDs for a term."""
    uids: list[str] = []
    retstart = 0
    while len(uids) < max_results:
        batch = client.esearch(
            "clinvar",
            term,
            retmax=min(200, max_results - len(uids)),
            retstart=retstart,
        )
        if not batch:
            break
        uids.extend(batch)
        retstart += len(batch)
        if len(batch) < 200:
            break
    return uids


def fetch_variants_for_term(
    gene: str,
    term: str,
    *,
    max_variants: int = 500,
    client: EutilsClient | None = None,
) -> list[tuple[Variant, str | None, str | None]]:
    """Return (variant, germline_significance, review_status) for variants matching a
    ClinVar Entrez `term` (already including the gene + any filters)."""
    own = client is None
    client = client or EutilsClient()
    try:
        uids = _esearch_all(client, term, max_variants)
        results: list[tuple[Variant, str | None, str | None]] = []
        for i in range(0, len(uids), 200):
            chunk = uids[i : i + 200]
            summ = client.esummary("clinvar", chunk)
            for uid in chunk:
                doc = summ.get(uid)
                if not isinstance(doc, dict):
                    continue
                sig, status = _extract_classification(doc)
                results.append((_variant_from_doc(gene, uid, doc), sig, status))
        return results
    finally:
        if own:
            client.close()


def build_gold_records(
    gene: str,
    *,
    min_stars: int = 2,
    max_variants: int = 500,
    client: EutilsClient | None = None,
) -> list[GoldRecord]:
    """Trustworthy gold set: variants whose ClinVar germline label is (a) unambiguous
    and (b) backed by at least `min_stars` review status (default 2★ = multiple
    submitters, no conflicts). 3★ ('reviewed by expert panel') is the gold standard for
    BRCA1/2, curated by the ENIGMA expert panel.

    We query each qualifying review-status tier separately (ClinVar 500s on an OR of two
    quoted [Review Status] phrases) and merge, deduplicating by variation_id.
    """
    own = client is None
    client = client or EutilsClient()
    try:
        tiers = sorted(
            (s for s in STARS_TO_REVIEW_PHRASE if s >= min_stars), reverse=True
        )
        seen: set[str] = set()
        gold: list[GoldRecord] = []
        for stars in tiers:
            phrase = STARS_TO_REVIEW_PHRASE[stars]
            term = f'{gene}[gene] AND "{phrase}"[Review Status]'
            try:
                tier_variants = fetch_variants_for_term(
                    gene, term, max_variants=max_variants, client=client
                )
            except httpx.HTTPStatusError as exc:
                # NCBI E-utilities 500s intermittently; don't let one flaky tier sink
                # the whole gold set. Skip it and keep the tiers that succeeded.
                logger.warning(
                    "Skipping %d★ tier for %s after repeated NCBI errors: %s",
                    stars,
                    gene,
                    exc,
                )
                continue
            for variant, sig, status in tier_variants:
                if not sig or not status:
                    continue
                if variant.variation_id in seen:
                    continue
                tier = ClinSig.from_clinvar(sig)
                if tier is None:  # conflicting / non-standard label — exclude from gold
                    continue
                seen.add(variant.variation_id)
                gold.append(
                    GoldRecord(
                        variant=variant,
                        gold=tier,
                        review_status=status,
                        gold_stars=review_status_to_stars(status),
                    )
                )
        return gold
    finally:
        if own:
            client.close()
