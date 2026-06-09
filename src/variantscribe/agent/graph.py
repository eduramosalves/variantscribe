"""Multi-agent ACMG classifier built on LangGraph.

Topology:

    START ─┬─▶ null_variant ──────────┐
           ├─▶ population_frequency ───┤
           ├─▶ computational ──────────┼─▶ combine ─▶ END
           └─▶ functional_literature ──┘

The four specialists run in parallel, each emitting ACMG criteria for its slice; their
criteria are merged (reducer) and the deterministic `combine` node applies the published
ACMG rules to produce the final tier. Output is a `Classification`, so this is a drop-in
alternative to the single-agent `LLMClassifier` for the eval harness.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from variantscribe.agent.acmg import bucket_of, combine, tier_confidence
from variantscribe.agent.classifier import _extract_tool_input, _make_client
from variantscribe.agent.prompts import render_evidence, render_variant
from variantscribe.agent.specialists import SPECIALISTS, SUBMIT_CRITERIA_TOOL, Specialist
from variantscribe.agent.tracing import observe
from variantscribe.config import settings
from variantscribe.models import ACMGCriterion, Classification, ClinSig, EvidenceItem, Variant


class GraphState(TypedDict, total=False):
    variant: Variant
    evidence: list[EvidenceItem]
    criteria: Annotated[list[ACMGCriterion], operator.add]
    usage: Annotated[list[tuple[int, int]], operator.add]
    tier: ClinSig | None
    reason: str


class GraphClassifier:
    """LangGraph multi-agent classifier. `evidence_fn` (optional) gathers shared evidence
    once per variant; specialists then reason over it in parallel."""

    def __init__(
        self,
        model: str | None = None,
        client=None,
        *,
        evidence_fn=None,
        max_tokens: int = 900,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or settings.agent_model
        self._client = client if client is not None else _make_client()
        self.evidence_fn = evidence_fn
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._graph = self._build_graph()

    # --- graph construction ---------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("combine", self._combine_node)
        for spec in SPECIALISTS:
            g.add_node(spec.name, self._make_specialist_node(spec))
            g.add_edge(START, spec.name)
            g.add_edge(spec.name, "combine")
        g.add_edge("combine", END)
        return g.compile()

    def _make_specialist_node(self, spec: Specialist):
        allowed = set(spec.codes)

        @observe
        def node(state: GraphState) -> dict:
            payload, usage = self._tool_call(
                spec.system_prompt,
                self._user_prompt(state["variant"], state.get("evidence", [])),
            )
            criteria = []
            for c in payload.get("criteria", []):
                code = str(c.get("code", "")).upper()
                # Keep only criteria this specialist owns (guards against scope drift).
                if code in allowed and bucket_of(code):
                    criteria.append(
                        ACMGCriterion(
                            code=code,
                            applied=bool(c.get("applied", False)),
                            strength=c.get("strength"),
                            rationale=c.get("rationale", ""),
                        )
                    )
            return {"criteria": criteria, "usage": [usage]}

        node.__name__ = f"specialist_{spec.name}"
        return node

    @observe
    def _combine_node(self, state: GraphState) -> dict:
        tier, reason = combine(state.get("criteria", []))
        return {"tier": tier, "reason": reason}

    # --- LLM plumbing ---------------------------------------------------------------

    def _user_prompt(self, variant: Variant, evidence: list[EvidenceItem]) -> str:
        return (
            render_variant(variant.gene, variant.name, variant.hgvs_c)
            + "\n\n"
            + render_evidence(evidence)
        )

    def _tool_call(self, system: str, user: str) -> tuple[dict, tuple[int, int]]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            tools=[SUBMIT_CRITERIA_TOOL],
            tool_choice={"type": "tool", "name": "submit_criteria"},
            messages=[{"role": "user", "content": user}],
        )
        payload = _extract_tool_input(resp)
        usage = getattr(resp, "usage", None)
        return payload, (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    # --- public API (drop-in for LLMClassifier) -------------------------------------

    @observe
    def classify(
        self, variant: Variant, evidence: list[EvidenceItem] | None = None
    ) -> Classification:
        if evidence is None:
            evidence = self.evidence_fn(variant) if self.evidence_fn else []

        t0 = time.monotonic()
        final: GraphState = self._graph.invoke(
            {"variant": variant, "evidence": evidence, "criteria": [], "usage": []}
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        tier = final.get("tier")
        usage = final.get("usage", [])
        applied = [c for c in final.get("criteria", []) if c.applied]
        return Classification(
            variation_id=variant.variation_id,
            call=tier,
            abstained=tier is None,
            confidence=tier_confidence(tier),
            criteria=applied,
            rationale=final.get("reason", ""),
            evidence=evidence,
            model=self.model,
            input_tokens=sum(u[0] for u in usage),
            output_tokens=sum(u[1] for u in usage),
            latency_ms=latency_ms,
        )
