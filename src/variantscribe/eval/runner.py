"""Compose a single classification run into (predictions, report, cases, telemetry)."""

from __future__ import annotations

from dataclasses import dataclass

from variantscribe.agent.classifier import classify_batch
from variantscribe.agent.telemetry import RunTelemetry, summarize_run
from variantscribe.eval.metrics import EvalReport, evaluate
from variantscribe.eval.store import build_eval_cases
from variantscribe.models import Classification, EvalCase, GoldRecord


@dataclass
class RunOutput:
    predictions: list[Classification]
    cases: list[EvalCase]
    report: EvalReport
    telemetry: RunTelemetry


def run_classification(
    gold: list[GoldRecord],
    classifier,
    *,
    evidence_fn=None,
    max_workers: int = 4,
) -> RunOutput:
    preds = classify_batch(classifier, gold, evidence_fn=evidence_fn, max_workers=max_workers)
    cases = build_eval_cases(gold, preds)
    report = evaluate(cases)
    telem = summarize_run(preds, model=getattr(classifier, "model", None))
    return RunOutput(predictions=preds, cases=cases, report=report, telemetry=telem)
