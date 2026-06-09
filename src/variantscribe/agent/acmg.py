"""Deterministic ACMG/AMP classification rules (Richards et al., Genet Med 2015, Table 5).

The multi-agent graph splits the *evidence* work across specialist LLM nodes, but the final
tier is decided here by the published combining rules — a transparent, reviewable function,
not a black-box model output. Strength is inferred from each criterion's code prefix
(PVS/PS/PM/PP for pathogenic; BA/BS/BP for benign).
"""

from __future__ import annotations

from collections import Counter

from variantscribe.models import ACMGCriterion, ClinSig

# Code prefix -> strength bucket. Order matters: check PVS before PS, etc.
_PREFIXES = ("PVS", "PS", "PM", "PP", "BA", "BS", "BP")


def bucket_of(code: str) -> str | None:
    code = code.strip().upper()
    for prefix in _PREFIXES:
        if code.startswith(prefix):
            return prefix
    return None


def count_buckets(criteria: list[ACMGCriterion]) -> Counter:
    counts: Counter = Counter()
    for cr in criteria:
        if cr.applied:
            b = bucket_of(cr.code)
            if b:
                counts[b] += 1
    return counts


def _pathogenic(pvs: int, ps: int, pm: int, pp: int) -> bool:
    return (
        (pvs >= 1 and (ps >= 1 or pm >= 2 or (pm >= 1 and pp >= 1) or pp >= 2))
        or ps >= 2
        or (ps >= 1 and (pm >= 3 or (pm >= 2 and pp >= 2) or (pm >= 1 and pp >= 4)))
    )


def _likely_pathogenic(pvs: int, ps: int, pm: int, pp: int) -> bool:
    # Evaluated only after _pathogenic fails, so overlapping stronger combos are safe.
    return (
        (pvs >= 1 and pm >= 1)
        or (ps >= 1 and pm >= 1)
        or (ps >= 1 and pp >= 2)
        or pm >= 3
        or (pm >= 2 and pp >= 2)
        or (pm >= 1 and pp >= 4)
    )


def _benign(ba: int, bs: int) -> bool:
    return ba >= 1 or bs >= 2


def _likely_benign(bs: int, bp: int) -> bool:
    return (bs >= 1 and bp >= 1) or bp >= 2


def combine(criteria: list[ACMGCriterion]) -> tuple[ClinSig | None, str]:
    """Combine applied criteria into (tier, reason). Returns (None, ...) only when no
    criterion applies at all — the caller treats that as an abstention."""
    counts = count_buckets(criteria)
    pvs, ps, pm, pp = counts["PVS"], counts["PS"], counts["PM"], counts["PP"]
    ba, bs, bp = counts["BA"], counts["BS"], counts["BP"]

    if sum(counts.values()) == 0:
        return None, "no ACMG criteria met — insufficient evidence"

    summary = f"PVS={pvs} PS={ps} PM={pm} PP={pp} | BA={ba} BS={bs} BP={bp}"

    # Contradictory evidence (any pathogenic AND any benign criterion) is the ACMG
    # "criteria are contradictory" case → Uncertain. Conservative by design: this avoids
    # ever letting benign evidence silently override pathogenic evidence (the dangerous
    # direction) — e.g. a PVS1 + BA1 variant is flagged for human review, not called benign.
    has_pathogenic = (pvs + ps + pm + pp) > 0
    has_benign = (ba + bs + bp) > 0
    if has_pathogenic and has_benign:
        return ClinSig.VUS, f"contradictory pathogenic and benign criteria ({summary})"

    if _pathogenic(pvs, ps, pm, pp):
        return ClinSig.PATHOGENIC, f"Pathogenic by ACMG rules ({summary})"
    if _likely_pathogenic(pvs, ps, pm, pp):
        return ClinSig.LIKELY_PATHOGENIC, f"Likely pathogenic by ACMG rules ({summary})"
    if _benign(ba, bs):
        return ClinSig.BENIGN, f"Benign by ACMG rules ({summary})"
    if _likely_benign(bs, bp):
        return ClinSig.LIKELY_BENIGN, f"Likely benign by ACMG rules ({summary})"
    return ClinSig.VUS, f"criteria insufficient for a classification ({summary})"


# Decisiveness → a heuristic confidence (self-reported, checked by the calibration metric).
_TIER_CONFIDENCE: dict[ClinSig, float] = {
    ClinSig.PATHOGENIC: 0.9,
    ClinSig.BENIGN: 0.9,
    ClinSig.LIKELY_PATHOGENIC: 0.7,
    ClinSig.LIKELY_BENIGN: 0.7,
    ClinSig.VUS: 0.5,
}


def tier_confidence(tier: ClinSig | None) -> float | None:
    return _TIER_CONFIDENCE.get(tier) if tier is not None else None
