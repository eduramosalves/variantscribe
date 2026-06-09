from variantscribe.agent.acmg import (
    bucket_of,
    combine,
    count_buckets,
    tier_confidence,
)
from variantscribe.models import ACMGCriterion, ClinSig


def crit(code, applied=True):
    return ACMGCriterion(code=code, applied=applied, rationale="t")


def tier(criteria):
    return combine(criteria)[0]


def test_bucket_of_prefixes():
    assert bucket_of("PVS1") == "PVS"
    assert bucket_of("PS3") == "PS"
    assert bucket_of("PM2") == "PM"
    assert bucket_of("PP3") == "PP"
    assert bucket_of("BA1") == "BA"
    assert bucket_of("BS1") == "BS"
    assert bucket_of("BP4") == "BP"
    assert bucket_of("ZZ9") is None


def test_count_ignores_unapplied():
    counts = count_buckets([crit("PVS1"), crit("PM2", applied=False)])
    assert counts["PVS"] == 1 and counts["PM"] == 0


# --- pathogenic combinations -------------------------------------------------------

def test_pvs1_plus_one_strong_is_pathogenic():
    assert tier([crit("PVS1"), crit("PS3")]) is ClinSig.PATHOGENIC


def test_pvs1_plus_two_moderate_is_pathogenic():
    assert tier([crit("PVS1"), crit("PM1"), crit("PM2")]) is ClinSig.PATHOGENIC


def test_two_strong_is_pathogenic():
    assert tier([crit("PS1"), crit("PS3")]) is ClinSig.PATHOGENIC


def test_pvs1_plus_one_moderate_is_likely_pathogenic():
    assert tier([crit("PVS1"), crit("PM2")]) is ClinSig.LIKELY_PATHOGENIC


def test_three_moderate_is_likely_pathogenic():
    assert tier([crit("PM1"), crit("PM2"), crit("PM5")]) is ClinSig.LIKELY_PATHOGENIC


# --- benign combinations -----------------------------------------------------------

def test_ba1_is_benign():
    assert tier([crit("BA1")]) is ClinSig.BENIGN


def test_two_strong_benign_is_benign():
    assert tier([crit("BS1"), crit("BS2")]) is ClinSig.BENIGN


def test_one_strong_one_supporting_benign_is_likely_benign():
    assert tier([crit("BS1"), crit("BP4")]) is ClinSig.LIKELY_BENIGN


def test_two_supporting_benign_is_likely_benign():
    assert tier([crit("BP1"), crit("BP4")]) is ClinSig.LIKELY_BENIGN


# --- VUS / abstain -----------------------------------------------------------------

def test_conflicting_pathogenic_and_benign_is_vus():
    assert tier([crit("PVS1"), crit("BA1")]) is ClinSig.VUS


def test_single_supporting_pathogenic_is_vus():
    assert tier([crit("PP3")]) is ClinSig.VUS


def test_no_criteria_is_abstain():
    t, reason = combine([])
    assert t is None and "insufficient" in reason


def test_only_unapplied_is_abstain():
    assert tier([crit("PVS1", applied=False)]) is None


def test_tier_confidence():
    assert tier_confidence(ClinSig.PATHOGENIC) == 0.9
    assert tier_confidence(ClinSig.LIKELY_BENIGN) == 0.7
    assert tier_confidence(None) is None
