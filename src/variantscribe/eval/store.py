"""Persistence for gold sets and prediction runs (newline-delimited JSON).

JSONL keeps runs diff-able and append-friendly, and lets the CI eval gate load a
fixed gold set and a fresh prediction run to compare against a baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from variantscribe.models import Classification, ClinSig, EvalCase, GoldRecord


def write_gold(records: list[GoldRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


def read_gold(path: Path) -> list[GoldRecord]:
    with path.open(encoding="utf-8") as fh:
        return [GoldRecord.model_validate_json(line) for line in fh if line.strip()]


def write_predictions(preds: list[Classification], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(p.model_dump_json() + "\n")


def read_predictions(path: Path) -> list[Classification]:
    with path.open(encoding="utf-8") as fh:
        return [Classification.model_validate_json(line) for line in fh if line.strip()]


def build_eval_cases(
    gold: list[GoldRecord], preds: list[Classification]
) -> list[EvalCase]:
    """Join gold labels and predictions on variation_id. Variants the model never
    predicted on are treated as abstentions (they count against coverage, not accuracy)."""
    pred_by_id: dict[str, Classification] = {p.variation_id: p for p in preds}
    cases: list[EvalCase] = []
    for g in gold:
        vid = g.variant.variation_id
        p = pred_by_id.get(vid)
        if p is None:
            cases.append(
                EvalCase(variation_id=vid, gold=g.gold, predicted=None, abstained=True)
            )
            continue
        predicted: ClinSig | None = None if p.abstained else p.call
        cases.append(
            EvalCase(
                variation_id=vid,
                gold=g.gold,
                predicted=predicted,
                abstained=p.abstained or predicted is None,
                confidence=p.confidence,
            )
        )
    return cases


def write_report_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
