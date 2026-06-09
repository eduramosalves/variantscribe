#!/usr/bin/env python
"""Run the evaluation sweep and publish the results into the README.

Computes the trivial baselines live (no API key needed) and attempts each LLM
configuration; configs that can't run yet (no API key, or a missing index) are recorded
as 'pending' so the table is always publishable. Splices a Markdown results table and a
calibration reliability plot into the README between markers, and writes RESULTS.md.

    uv run python scripts/run_results.py --gene BRCA1 --limit 60

Add VARIANTSCRIBE_ANTHROPIC_API_KEY (+ build the retrieval/guideline indexes) to fill the
LLM rows; re-run to refresh the published numbers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from variantscribe.agent.baseline import baseline_predictions
from variantscribe.config import settings
from variantscribe.eval.calibration import reliability_table
from variantscribe.eval.metrics import evaluate
from variantscribe.eval.report import inject_between_markers, reliability_plot, results_table
from variantscribe.eval.runner import run_classification
from variantscribe.eval.sampling import stratified_sample
from variantscribe.eval.store import build_eval_cases, read_gold

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# (classifier, evidence) configurations attempted for the LLM rows.
LLM_CONFIGS = [
    ("agent", "none"),
    ("agent", "retrieval"),
    ("graph", "none"),
    ("graph", "retrieval"),
    ("graph", "retrieval+guidelines"),
]


def _baseline_row(gold, strategy: str) -> dict:
    report = evaluate(build_eval_cases(gold, baseline_predictions(gold, strategy)))
    return {
        "classifier": "baseline",
        "evidence": strategy,
        "n": report.n_total,
        "macro_f1": report.macro_f1,
        "three_class_accuracy": report.three_class_accuracy,
        "dangerous_error_rate": report.dangerous_error_rate,
        "ece": report.ece,
        "cost_per_1k": 0.0,
    }


def _build_evidence_fn(evidence: str, gene: str, k: int):
    from variantscribe.agent.evidence import (
        combined_evidence_fn,
        retrieval_evidence_fn,
    )

    fn = None
    if "retrieval" in evidence:
        from variantscribe.retrieval.pipeline import load_retriever

        fn = retrieval_evidence_fn(load_retriever(gene, rerank=True, k_final=k))
    if "guidelines" in evidence:
        from variantscribe.retrieval.guidelines import (
            guideline_evidence_fn,
            load_guideline_retriever,
        )

        gl = guideline_evidence_fn(load_guideline_retriever(k_final=3))
        fn = combined_evidence_fn(fn, gl)
    return fn


def _llm_row(gold, classifier_kind: str, evidence: str, gene: str, k: int) -> tuple[dict, list]:
    """Returns (row, reliability_bins). On any setup failure, a 'pending' row."""
    base = {"classifier": classifier_kind, "evidence": evidence, "n": len(gold)}
    try:
        evidence_fn = _build_evidence_fn(evidence, gene, k)
        if classifier_kind == "graph":
            from variantscribe.agent.graph import GraphClassifier

            clf = GraphClassifier(evidence_fn=evidence_fn)
            out = run_classification(gold, clf, evidence_fn=None)
        else:
            from variantscribe.agent.classifier import LLMClassifier

            out = run_classification(gold, LLMClassifier(), evidence_fn=evidence_fn)
    except Exception as exc:  # no key, missing index, etc. — keep the table publishable
        base.update({"n": None, "note": type(exc).__name__})
        return base, []

    r = out.report
    base.update(
        {
            "macro_f1": r.macro_f1,
            "three_class_accuracy": r.three_class_accuracy,
            "dangerous_error_rate": r.dangerous_error_rate,
            "ece": r.ece,
            "cost_per_1k": out.telemetry.cost_per_1k_variants_usd,
        }
    )
    bins = [
        {"label": b.label, "n": b.n, "mean_confidence": b.mean_confidence, "accuracy": b.accuracy}
        for b in reliability_table(out.cases)
    ]
    return base, bins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", default="BRCA1")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    gold_path = settings.raw_dir / f"gold_{args.gene.upper()}.jsonl"
    if not gold_path.exists():
        print(f"No gold set at {gold_path}. Run: variantscribe build-gold --gene {args.gene}")
        return 1
    gold = stratified_sample(read_gold(gold_path), args.limit, seed=0)
    print(f"Sweep on {len(gold)} {args.gene} variants (stratified)…")

    rows = [_baseline_row(gold, "majority"), _baseline_row(gold, "always-vus")]
    best_bins: list[dict] = []
    best_f1 = -1.0
    for kind, evidence in LLM_CONFIGS:
        row, bins = _llm_row(gold, kind, evidence, args.gene, args.k)
        status = "pending" if row.get("macro_f1") is None else f"F1={row['macro_f1']:.3f}"
        print(f"  {kind:6s} {evidence:24s} {status}")
        rows.append(row)
        if row.get("macro_f1", -1) is not None and row.get("macro_f1", -1) > best_f1 and bins:
            best_f1, best_bins = row["macro_f1"], bins

    table = results_table(rows)

    # Reliability plot (only when at least one LLM config ran and reported confidence).
    if best_bins:
        settings.ensure_dirs()
        plot_path = ROOT / "docs" / "calibration.png"
        reliability_plot(best_bins, plot_path, title=f"{args.gene} — best config calibration")
        plot_md = "![calibration reliability](docs/calibration.png)"
    else:
        plot_md = "_Calibration plot pending — run with an API key to populate._"

    readme = README.read_text(encoding="utf-8")
    readme = inject_between_markers(readme, "results-table", table)
    readme = inject_between_markers(readme, "results-plot", plot_md)
    README.write_text(readme, encoding="utf-8")

    stamp = time.strftime("%Y-%m-%d %H:%M")
    (ROOT / "RESULTS.md").write_text(
        f"# Results — {args.gene} (n={len(gold)})\n\n_Generated {stamp}._\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"\nUpdated README results + RESULTS.md ({stamp}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
