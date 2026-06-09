"""Domain models for VariantScribe.

These are the contract between ingestion, retrieval, the classifier agent, and the
eval harness. They intentionally mirror the ACMG/AMP 2015 framework (Richards et al.,
Genet Med 2015) so the system's output is reviewable by a molecular geneticist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ClinSig(StrEnum):
    """The 5-tier ACMG/AMP classification. Used both as the model's output and,
    via ClinVar, as the gold label for evaluation."""

    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely pathogenic"
    VUS = "Uncertain significance"
    LIKELY_BENIGN = "Likely benign"
    BENIGN = "Benign"

    @classmethod
    def from_clinvar(cls, raw: str) -> ClinSig | None:
        """Normalize ClinVar's free-text significance strings to a tier.

        ClinVar uses combined labels ("Pathogenic/Likely pathogenic"), aggregate
        descriptors, and conflicting calls. We map the unambiguous ones and return
        None for everything else so the gold-set builder can filter them out.
        """
        s = raw.strip().lower()
        exact = {
            "pathogenic": cls.PATHOGENIC,
            "pathogenic/likely pathogenic": cls.PATHOGENIC,
            "likely pathogenic": cls.LIKELY_PATHOGENIC,
            "uncertain significance": cls.VUS,
            "likely benign": cls.LIKELY_BENIGN,
            "benign": cls.BENIGN,
            "benign/likely benign": cls.BENIGN,
        }
        return exact.get(s)


# Ordinal scale used by metrics to measure how *far off* a prediction is and to flag
# the clinically dangerous direction (calling a pathogenic variant benign).
CLINSIG_ORDINAL: dict[ClinSig, int] = {
    ClinSig.BENIGN: 0,
    ClinSig.LIKELY_BENIGN: 1,
    ClinSig.VUS: 2,
    ClinSig.LIKELY_PATHOGENIC: 3,
    ClinSig.PATHOGENIC: 4,
}


class Variant(BaseModel):
    """A genetic variant, keyed by the identifiers we use to join across sources."""

    gene: str
    variation_id: str  # ClinVar VariationID (stable primary key)
    name: str | None = None  # e.g. "NM_007294.4(BRCA1):c.5266dupC (p.Gln1756fs)"
    hgvs_c: str | None = None
    hgvs_p: str | None = None
    rsid: str | None = None  # dbSNP, when available


class GnomadFrequency(BaseModel):
    """Population allele frequency, the basis for ACMG BA1/BS1/PM2 criteria."""

    allele_frequency: float | None = None
    popmax_af: float | None = None
    popmax_population: str | None = None
    allele_count: int | None = None
    allele_number: int | None = None
    found: bool = False


class EvidenceItem(BaseModel):
    """A single retrieved piece of evidence with provenance for citation checking."""

    source: str  # "pubmed" | "gnomad" | "clinvar-submitter" | ...
    kind: str  # "literature" | "frequency" | "computational" | "functional"
    text: str
    citation: str | None = None  # PMID, URL, or accession — must be verifiable
    score: float | None = None  # retrieval / rerank score


class ACMGCriterion(BaseModel):
    """One ACMG/AMP evidence criterion the agent decided to apply (or not)."""

    code: str  # e.g. "PVS1", "PM2", "BS1"
    applied: bool
    strength: str | None = None  # "very_strong" | "strong" | "moderate" | "supporting"
    rationale: str = ""


class Classification(BaseModel):
    """The agent's draft assessment for a variant — the unit the eval harness scores."""

    variation_id: str
    call: ClinSig | None  # None == abstained
    abstained: bool = False
    confidence: float | None = None  # 0..1 self-reported; checked against accuracy
    criteria: list[ACMGCriterion] = Field(default_factory=list)
    rationale: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # LLMOps telemetry — populated by the LLM classifier, summed into a run's cost report.
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class GoldRecord(BaseModel):
    """A variant with a trustworthy ClinVar classification, used as ground truth."""

    variant: Variant
    gold: ClinSig
    review_status: str
    gold_stars: int  # ClinVar review-status star rating (0-4)
    conditions: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One scored prediction: gold vs. the model's call."""

    variation_id: str
    gold: ClinSig
    predicted: ClinSig | None
    abstained: bool
    confidence: float | None = None
