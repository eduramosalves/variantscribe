"""ACMG specialist definitions for the multi-agent graph.

Each specialist owns a slice of the ACMG criteria and reasons only about that slice over
the same shared evidence. Decomposing the work this way (vs. one prompt that must juggle
all 28 criteria) gives focused prompts, parallelism, and per-criterion traceability.
"""

from __future__ import annotations

from dataclasses import dataclass

_BASE = (
    "You are an ACMG/AMP variant-curation specialist. Consider ONLY your assigned criteria. "
    "For each, decide whether it applies to this variant given the evidence, and justify it. "
    "Ground every applied criterion in the provided evidence or well-established molecular "
    "facts — never invent data. If your criteria do not apply or you cannot tell, return them "
    "as not applied. Report via the submit_criteria tool."
)


@dataclass(frozen=True)
class Specialist:
    name: str
    codes: tuple[str, ...]
    focus: str

    @property
    def system_prompt(self) -> str:
        codes = ", ".join(self.codes)
        return f"{_BASE}\n\nYour focus: {self.focus}\nYour assigned criteria: {codes}."


SPECIALISTS: list[Specialist] = [
    Specialist(
        name="null_variant",
        codes=("PVS1",),
        focus=(
            "loss-of-function consequence — nonsense, frameshift, canonical ±1/2 splice, "
            "initiation-codon, and single/multi-exon deletions in a gene where LOF is a "
            "known disease mechanism"
        ),
    ),
    Specialist(
        name="population_frequency",
        codes=("BA1", "BS1", "BS2", "PM2"),
        focus=(
            "population allele frequency — stand-alone benign if too common (BA1), benign "
            "strong if above the disease threshold (BS1), and supporting-pathogenic if "
            "absent/rare in population databases (PM2)"
        ),
    ),
    Specialist(
        name="computational",
        codes=("PP3", "BP4", "BP1", "BP7"),
        focus=(
            "in-silico evidence — concordant computational predictors of a deleterious "
            "(PP3) or benign (BP4) effect, missense-in-truncating-gene (BP1), and silent "
            "variants with no predicted splice impact (BP7)"
        ),
    ),
    Specialist(
        name="functional_literature",
        codes=("PS3", "BS3", "PS1", "PM5", "PS4", "PP5", "BP6"),
        focus=(
            "functional and clinical literature — well-established functional studies "
            "(PS3/BS3), same/different amino-acid change as a known pathogenic variant "
            "(PS1/PM5), case-control enrichment (PS4), and reputable-source assertions "
            "(PP5/BP6)"
        ),
    ),
]


# Tool the specialists call to return their criteria assessment.
SUBMIT_CRITERIA_TOOL = {
    "name": "submit_criteria",
    "description": "Report which of your assigned ACMG criteria apply to this variant.",
    "input_schema": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
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
            }
        },
        "required": ["criteria"],
    },
}
