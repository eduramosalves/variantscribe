"""Metrics for variant-classification quality.

Beyond plain accuracy, this reports the things that actually matter in a clinical
setting and that interviewers probe for:

* coverage / abstention   — does the system know when to stay silent?
* macro-F1                — performance across all five tiers, not just the common one
* dangerous-error rate    — calling a (likely) pathogenic variant (likely) benign
* ordinal MAE             — how far off, on the 5-tier scale, when wrong
* 3-class accuracy        — collapsing to the clinically actionable grouping
* calibration             — does self-reported confidence track real accuracy?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import f1_score

from variantscribe.eval.calibration import expected_calibration_error
from variantscribe.models import CLINSIG_ORDINAL, ClinSig, EvalCase

# Clinically actionable 3-way collapse of the 5-tier scale.
_PATHOGENIC = {ClinSig.PATHOGENIC, ClinSig.LIKELY_PATHOGENIC}
_BENIGN = {ClinSig.BENIGN, ClinSig.LIKELY_BENIGN}


def _three_class(c: ClinSig) -> str:
    if c in _PATHOGENIC:
        return "pathogenic"
    if c in _BENIGN:
        return "benign"
    return "vus"


@dataclass
class EvalReport:
    n_total: int
    n_answered: int
    coverage: float
    accuracy: float  # exact 5-tier accuracy on answered
    macro_f1: float
    three_class_accuracy: float
    ordinal_mae: float
    dangerous_errors: int  # gold (L)Pathogenic, predicted (L)Benign
    dangerous_error_rate: float
    ece: float | None = None  # expected calibration error
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    calibration: list[tuple[str, int, float]] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"cases:            {self.n_total} ({self.n_answered} answered, "
            f"{self.n_total - self.n_answered} abstained)",
            f"coverage:         {self.coverage:.1%}",
            f"accuracy (5-tier):{self.accuracy:.3f}",
            f"macro-F1:         {self.macro_f1:.3f}",
            f"3-class accuracy: {self.three_class_accuracy:.3f}",
            f"ordinal MAE:      {self.ordinal_mae:.3f}",
            f"dangerous errors: {self.dangerous_errors} "
            f"({self.dangerous_error_rate:.1%} of answered)",
        ] + (
            [f"ECE (calibration):{self.ece:.3f}"] if self.ece is not None else []
        )


def evaluate(cases: list[EvalCase]) -> EvalReport:
    n_total = len(cases)
    answered = [c for c in cases if not c.abstained and c.predicted is not None]
    n_answered = len(answered)

    if n_answered == 0:
        return EvalReport(
            n_total=n_total,
            n_answered=0,
            coverage=0.0,
            accuracy=0.0,
            macro_f1=0.0,
            three_class_accuracy=0.0,
            ordinal_mae=0.0,
            dangerous_errors=0,
            dangerous_error_rate=0.0,
        )

    gold = [c.gold for c in answered]
    pred = [c.predicted for c in answered]

    correct = sum(g == p for g, p in zip(gold, pred, strict=False))
    accuracy = correct / n_answered

    labels = list(ClinSig)
    macro_f1 = float(
        f1_score(
            [g.value for g in gold],
            [p.value for p in pred],
            labels=[c.value for c in labels],
            average="macro",
            zero_division=0,
        )
    )

    three_correct = sum(
        _three_class(g) == _three_class(p) for g, p in zip(gold, pred, strict=False)
    )
    three_class_accuracy = three_correct / n_answered

    ordinal_mae = sum(
        abs(CLINSIG_ORDINAL[g] - CLINSIG_ORDINAL[p]) for g, p in zip(gold, pred, strict=False)
    ) / n_answered

    dangerous = sum(g in _PATHOGENIC and p in _BENIGN for g, p in zip(gold, pred, strict=False))

    confusion: dict[str, dict[str, int]] = {
        g.value: {p.value: 0 for p in labels} for g in labels
    }
    for g, p in zip(gold, pred, strict=False):
        confusion[g.value][p.value] += 1

    return EvalReport(
        n_total=n_total,
        n_answered=n_answered,
        coverage=n_answered / n_total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        three_class_accuracy=three_class_accuracy,
        ordinal_mae=ordinal_mae,
        dangerous_errors=dangerous,
        dangerous_error_rate=dangerous / n_answered,
        ece=expected_calibration_error(answered),
        confusion=confusion,
        calibration=_calibration(answered),
    )


def _calibration(answered: list[EvalCase], bins: int = 5) -> list[tuple[str, int, float]]:
    """Bucket predictions by self-reported confidence and report accuracy per bucket.
    Returns (range_label, count, accuracy). Cases without confidence are skipped."""
    scored = [c for c in answered if c.confidence is not None]
    if not scored:
        return []
    out: list[tuple[str, int, float]] = []
    width = 1.0 / bins
    for b in range(bins):
        lo, hi = b * width, (b + 1) * width
        in_last = b == bins - 1  # fold confidence == 1.0 into the final bucket
        bucket = [
            c
            for c in scored
            if (lo <= c.confidence < hi) or (in_last and c.confidence == 1.0)
        ]
        if not bucket:
            continue
        acc = sum(c.gold == c.predicted for c in bucket) / len(bucket)
        out.append((f"{lo:.1f}-{hi:.1f}", len(bucket), acc))
    return out
