"""Token/cost/latency rollups for a classification run (the LLMOps view).

PRICING is an editable estimate (USD per 1M tokens). Token counts are always exact —
only the dollar figure depends on these rates, and it is labelled an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from variantscribe.models import Classification

# USD per 1,000,000 tokens (input, output). Adjust to your contract.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
}
_DEFAULT_RATE = (3.0, 15.0)


@dataclass
class RunTelemetry:
    n: int
    input_tokens: int
    output_tokens: int
    est_cost_usd: float
    cost_per_1k_variants_usd: float
    latency_ms_mean: float
    latency_ms_p50: float

    def summary_lines(self) -> list[str]:
        return [
            f"tokens:           {self.input_tokens:,} in / {self.output_tokens:,} out",
            f"est. cost:        ${self.est_cost_usd:.4f} "
            f"(~${self.cost_per_1k_variants_usd:.2f} / 1k variants)",
            f"latency:          {self.latency_ms_mean:.0f} ms mean / "
            f"{self.latency_ms_p50:.0f} ms p50",
        ]


def summarize_run(preds: list[Classification], model: str | None = None) -> RunTelemetry:
    in_tok = sum(p.input_tokens or 0 for p in preds)
    out_tok = sum(p.output_tokens or 0 for p in preds)
    rate_in, rate_out = PRICING.get(model or "", _DEFAULT_RATE)
    cost = in_tok / 1_000_000 * rate_in + out_tok / 1_000_000 * rate_out
    latencies = [p.latency_ms for p in preds if p.latency_ms is not None] or [0.0]
    n = len(preds)
    return RunTelemetry(
        n=n,
        input_tokens=in_tok,
        output_tokens=out_tok,
        est_cost_usd=cost,
        cost_per_1k_variants_usd=(cost / n * 1000) if n else 0.0,
        latency_ms_mean=mean(latencies),
        latency_ms_p50=median(latencies),
    )
