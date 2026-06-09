"""Single-agent LLM classifier (Phase 1).

Wraps an Anthropic model with forced tool-use so every call returns a parseable ACMG
assessment. Takes a variant plus *optional* evidence — with no evidence it is the honest
"LLM-only from nomenclature" ablation; once the retrieval index lands, the same classifier
is fed evidence and the lift is measurable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from variantscribe.agent.prompts import (
    CLASSIFICATION_TOOL,
    SYSTEM_PROMPT,
    render_evidence,
    render_variant,
)
from variantscribe.config import settings
from variantscribe.models import (
    ACMGCriterion,
    Classification,
    ClinSig,
    EvidenceItem,
    GoldRecord,
    Variant,
)

EvidenceFn = Callable[[Variant], list[EvidenceItem]]


def _make_client():
    """Construct an Anthropic client, with a clear error if no key is configured."""
    import os

    import anthropic

    key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key. Set VARIANTSCRIBE_ANTHROPIC_API_KEY in .env "
            "(or ANTHROPIC_API_KEY in the environment)."
        )
    return anthropic.Anthropic(api_key=key)


def _tier_from_output(value: str) -> tuple[ClinSig | None, bool]:
    """Map the tool's classification string to (ClinSig | None, abstained)."""
    if value == "Abstain":
        return None, True
    try:
        return ClinSig(value), False
    except ValueError:
        # Defensive: an unexpected string is treated as an abstention, not a crash.
        return None, True


class LLMClassifier:
    def __init__(
        self,
        model: str | None = None,
        client=None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or settings.agent_model
        self._client = client if client is not None else _make_client()
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1.5, min=2, max=40),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call(self, user_prompt: str):
        return self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            tools=[CLASSIFICATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_classification"},
            messages=[{"role": "user", "content": user_prompt}],
        )

    def classify(
        self, variant: Variant, evidence: list[EvidenceItem] | None = None
    ) -> Classification:
        evidence = evidence or []
        user_prompt = (
            render_variant(variant.gene, variant.name, variant.hgvs_c)
            + "\n\n"
            + render_evidence(evidence)
        )

        t0 = time.monotonic()
        resp = self._call(user_prompt)
        latency_ms = (time.monotonic() - t0) * 1000.0

        payload = _extract_tool_input(resp)
        call, abstained = _tier_from_output(payload.get("classification", "Abstain"))
        criteria = [
            ACMGCriterion(
                code=c.get("code", "?"),
                applied=bool(c.get("applied", False)),
                strength=c.get("strength"),
                rationale=c.get("rationale", ""),
            )
            for c in payload.get("criteria", [])
        ]
        usage = getattr(resp, "usage", None)
        return Classification(
            variation_id=variant.variation_id,
            call=call,
            abstained=abstained,
            confidence=payload.get("confidence"),
            criteria=criteria,
            rationale=payload.get("rationale", ""),
            evidence=evidence,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=latency_ms,
        )


def _extract_tool_input(resp) -> dict:
    """Pull the forced tool call's input dict out of an Anthropic response."""
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError("model response contained no tool_use block")


def classify_batch(
    classifier: LLMClassifier,
    records: list[GoldRecord],
    *,
    evidence_fn: EvidenceFn | None = None,
    max_workers: int = 4,
    on_done: Callable[[int, int], None] | None = None,
) -> list[Classification]:
    """Classify a batch of variants concurrently. `evidence_fn` (optional) gathers
    evidence per variant; `on_done(completed, total)` is a progress callback."""
    results: dict[str, Classification] = {}
    total = len(records)

    def _work(rec: GoldRecord) -> Classification:
        evidence = evidence_fn(rec.variant) if evidence_fn else None
        return classifier.classify(rec.variant, evidence)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, rec): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            results[rec.variant.variation_id] = fut.result()
            if on_done:
                on_done(i, total)

    # Preserve input order.
    return [results[r.variant.variation_id] for r in records]
