"""Stratified sampling so a capped eval still represents every ACMG tier."""

from __future__ import annotations

import random

from variantscribe.models import ClinSig, GoldRecord


def stratified_sample(gold: list[GoldRecord], n: int, *, seed: int = 0) -> list[GoldRecord]:
    """Sample ~n records while preserving the per-tier proportions of the gold set, with
    at least one of every tier that's present. Deterministic given `seed`."""
    if n >= len(gold):
        return list(gold)

    rng = random.Random(seed)
    by_tier: dict[ClinSig, list[GoldRecord]] = {}
    for g in gold:
        by_tier.setdefault(g.gold, []).append(g)

    total = len(gold)
    picked: list[GoldRecord] = []
    for members in by_tier.values():
        quota = max(1, round(n * len(members) / total))
        quota = min(quota, len(members))
        picked.extend(rng.sample(members, quota))

    # Rounding can over/undershoot n; trim or top up deterministically.
    rng.shuffle(picked)
    if len(picked) > n:
        return picked[:n]
    if len(picked) < n:
        remaining = [g for g in gold if g not in picked]
        picked.extend(rng.sample(remaining, min(n - len(picked), len(remaining))))
    return picked
