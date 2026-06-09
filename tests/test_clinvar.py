from variantscribe.ingest.clinvar import (
    _extract_classification,
    review_status_to_stars,
)
from variantscribe.models import ClinSig


def test_review_status_stars():
    assert review_status_to_stars("reviewed by expert panel") == 3
    assert review_status_to_stars("criteria provided, multiple submitters, no conflicts") == 2
    assert review_status_to_stars("no assertion criteria provided") == 0
    assert review_status_to_stars("something unknown") == 0  # safe default


def test_clinsig_normalization():
    assert ClinSig.from_clinvar("Pathogenic") is ClinSig.PATHOGENIC
    assert ClinSig.from_clinvar("Pathogenic/Likely pathogenic") is ClinSig.PATHOGENIC
    assert ClinSig.from_clinvar("Likely benign") is ClinSig.LIKELY_BENIGN
    # conflicting / non-standard labels are excluded from the gold set
    assert ClinSig.from_clinvar("Conflicting classifications of pathogenicity") is None
    assert ClinSig.from_clinvar("drug response") is None


def test_extract_classification_prefers_germline():
    doc = {
        "germline_classification": {
            "description": "Pathogenic",
            "review_status": "reviewed by expert panel",
        },
        "clinical_impact_classification": {"description": "Tier III - Unknown"},
    }
    sig, status = _extract_classification(doc)
    assert sig == "Pathogenic"
    assert status == "reviewed by expert panel"


def test_extract_classification_empty_germline_returns_none():
    # somatic-only submissions have an empty germline block
    doc = {"germline_classification": {"description": "", "review_status": ""}}
    assert _extract_classification(doc) == (None, None)
