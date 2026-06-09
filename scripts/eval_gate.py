#!/usr/bin/env python
"""CI eval-regression gate.

Runs the real pipeline (gold load -> baseline predictions -> scoring) against a committed,
deterministic fixture and fails if the metrics drift from pinned expectations. This guards
the eval harness, scoring, baselines, and gold I/O against silent regressions on every PR —
without needing network, an API key, or torch.

The full LLM eval (real model, real lift numbers) runs separately and on demand via
.github/workflows/eval-llm.yml, so PRs never incur model cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

from variantscribe.agent.baseline import baseline_predictions
from variantscribe.eval.metrics import evaluate
from variantscribe.eval.store import build_eval_cases, read_gold

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gold_fixture.jsonl"
TOL = 0.01

# Pinned expectations for the fixture. A change here must be deliberate (and reviewed).
EXPECTED = {
    "majority": {"macro_f1": 0.100, "accuracy": 0.333, "dangerous_errors": 0},
    "always-vus": {"macro_f1": 0.067, "accuracy": 0.200, "dangerous_errors": 0},
}


def main() -> int:
    if not FIXTURE.exists():
        print(f"FAIL: fixture missing at {FIXTURE}", file=sys.stderr)
        return 1

    gold = read_gold(FIXTURE)
    print(f"eval-gate: {len(gold)} fixture variants\n")

    ok = True
    for strategy, exp in EXPECTED.items():
        report = evaluate(build_eval_cases(gold, baseline_predictions(gold, strategy)))
        got = {
            "macro_f1": report.macro_f1,
            "accuracy": report.accuracy,
            "dangerous_errors": report.dangerous_errors,
        }
        for metric, expected in exp.items():
            actual = got[metric]
            if metric == "dangerous_errors":
                passed = actual == expected
            else:
                passed = abs(actual - expected) <= TOL
            mark = "ok " if passed else "FAIL"
            print(f"  [{mark}] {strategy:11s} {metric:16s} expected {expected} got {actual:.4f}")
            ok = ok and passed

    print("\n" + ("PASS — metrics within tolerance" if ok else "FAIL — metric regression"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
