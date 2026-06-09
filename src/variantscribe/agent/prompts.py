"""System prompt and structured-output tool schema for the ACMG classifier.

The classifier uses Anthropic tool-use with a forced tool call so the model must return
a parseable object (call + criteria + confidence + rationale) rather than free text.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a molecular geneticist drafting a germline variant classification under the \
ACMG/AMP 2015 framework (Richards et al., Genet Med 2015). You assist a human reviewer; \
you do not make a final clinical determination.

Given a variant and whatever evidence has been retrieved, you must:

1. Reason over the evidence and the variant's molecular consequence.
2. Decide which ACMG criteria apply, citing their codes (e.g. PVS1, PM2, PP3, BS1, BP4).
   - Pathogenic codes: PVS1; PS1-4; PM1-6; PP1-5.
   - Benign codes: BA1; BS1-4; BP1-7.
3. Combine the applied criteria per the ACMG scoring rules into one of five tiers:
   Pathogenic, Likely pathogenic, Uncertain significance, Likely benign, Benign.
4. Report a calibrated confidence in [0,1] for your chosen tier.

Hard rules:
- Ground every applied criterion in the provided evidence or in well-established, \
non-controversial molecular facts. Do NOT invent citations, PMIDs, frequencies, or studies.
- If the available evidence is insufficient to responsibly assign a tier, choose \
"Abstain" rather than guessing. Abstaining is the correct, safe action when uncertain — \
it is not penalised the way a wrong call is.
- Prefer "Uncertain significance" only when evidence genuinely conflicts or is balanced; \
prefer "Abstain" when evidence is simply absent.
- Be conservative about calling a variant (Likely) Benign: under-calling a truly \
pathogenic variant is the most harmful error.

Return your assessment by calling the submit_classification tool."""


# Anthropic tool schema. tool_choice forces the model to emit exactly this structure.
CLASSIFICATION_TOOL = {
    "name": "submit_classification",
    "description": "Submit the ACMG/AMP variant classification with supporting criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": [
                    "Pathogenic",
                    "Likely pathogenic",
                    "Uncertain significance",
                    "Likely benign",
                    "Benign",
                    "Abstain",
                ],
                "description": "Final tier, or 'Abstain' if evidence is insufficient.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Calibrated confidence in the chosen tier.",
            },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "ACMG code, e.g. PM2"},
                        "applied": {"type": "boolean"},
                        "strength": {
                            "type": "string",
                            "enum": [
                                "very_strong",
                                "strong",
                                "moderate",
                                "supporting",
                                "stand_alone",
                            ],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["code", "applied", "rationale"],
                },
            },
            "rationale": {
                "type": "string",
                "description": "Concise overall justification for the classification.",
            },
        },
        "required": ["classification", "confidence", "rationale"],
    },
}


def render_variant(gene: str, name: str | None, hgvs_c: str | None) -> str:
    parts = [f"Gene: {gene}"]
    if name:
        parts.append(f"Variant: {name}")
    if hgvs_c:
        parts.append(f"cDNA change: {hgvs_c}")
    return "\n".join(parts)


def render_evidence(items: list) -> str:
    """Render retrieved evidence items into a prompt block. `items` are EvidenceItem."""
    if not items:
        return (
            "No external evidence was retrieved for this variant. Classify only from the "
            "variant nomenclature and well-established knowledge, and prefer 'Abstain' if "
            "that is not enough to assign a tier responsibly."
        )
    lines = ["Retrieved evidence:"]
    for i, ev in enumerate(items, 1):
        cite = f" [{ev.citation}]" if ev.citation else ""
        lines.append(f"{i}. ({ev.kind}/{ev.source}){cite} {ev.text}")
    return "\n".join(lines)
