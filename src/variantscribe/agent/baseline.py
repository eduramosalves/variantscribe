"""Trivial baselines that set the floor the real classifier must beat.

A portfolio project without baselines can't tell whether its fancy LLM actually adds
value. Two honest baselines:

* majority  — always predict the most frequent label in the gold set (standard ML floor)
* always-vus — always predict "Uncertain significance", the clinically safe non-call

Both ignore everything about the variant except (for majority) the label prior, so any
real classifier that can't beat them isn't earning its compute.
"""

from __future__ import annotations

from collections import Counter

from variantscribe.models import Classification, ClinSig, GoldRecord


def majority_label(gold: list[GoldRecord]) -> ClinSig:
    return Counter(g.gold for g in gold).most_common(1)[0][0]


def baseline_predictions(gold: list[GoldRecord], strategy: str) -> list[Classification]:
    if strategy == "majority":
        label = majority_label(gold)
    elif strategy == "always-vus":
        label = ClinSig.VUS
    else:
        raise ValueError(f"unknown baseline strategy: {strategy!r}")

    return [
        Classification(
            variation_id=g.variant.variation_id,
            call=label,
            confidence=None,
            rationale=f"baseline:{strategy}",
            model=f"baseline:{strategy}",
        )
        for g in gold
    ]
