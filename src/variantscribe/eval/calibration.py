"""Calibration analysis: does self-reported confidence track real accuracy?

A classifier that says "0.9 confident" should be right ~90% of the time. Expected
Calibration Error (ECE) summarises the gap; the reliability table shows it per bin. This
matters clinically: a well-calibrated abstain/confidence signal is what lets a reviewer
triage which drafts to trust.
"""

from __future__ import annotations

from dataclasses import dataclass

from variantscribe.models import EvalCase


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float

    @property
    def label(self) -> str:
        return f"{self.lo:.1f}-{self.hi:.1f}"


def _scored(cases: list[EvalCase]) -> list[tuple[float, bool]]:
    """(confidence, correct) for answered cases that carry a confidence."""
    out = []
    for c in cases:
        if c.abstained or c.predicted is None or c.confidence is None:
            continue
        out.append((c.confidence, c.predicted == c.gold))
    return out


def reliability_table(cases: list[EvalCase], bins: int = 5) -> list[ReliabilityBin]:
    scored = _scored(cases)
    table: list[ReliabilityBin] = []
    if not scored:
        return table
    width = 1.0 / bins
    for b in range(bins):
        lo, hi = b * width, (b + 1) * width
        in_last = b == bins - 1
        members = [
            (conf, ok)
            for conf, ok in scored
            if (lo <= conf < hi) or (in_last and conf == 1.0)
        ]
        if not members:
            continue
        n = len(members)
        table.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                n=n,
                mean_confidence=sum(c for c, _ in members) / n,
                accuracy=sum(ok for _, ok in members) / n,
            )
        )
    return table


def expected_calibration_error(cases: list[EvalCase], bins: int = 5) -> float | None:
    """ECE = sum_b (n_b/N) * |accuracy_b - mean_confidence_b|. None if no scored cases."""
    table = reliability_table(cases, bins=bins)
    if not table:
        return None
    total = sum(b.n for b in table)
    return sum(b.n / total * abs(b.accuracy - b.mean_confidence) for b in table)
