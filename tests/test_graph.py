"""LangGraph multi-agent classifier tests with a fake Anthropic client (no key/network)."""

from variantscribe.agent.graph import GraphClassifier
from variantscribe.models import ClinSig, EvidenceItem, Variant


class _Block:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Usage:
    input_tokens = 90
    output_tokens = 40


class _Resp:
    def __init__(self, payload):
        self.content = [_Block(payload)]
        self.usage = _Usage()


class _Messages:
    """Routes each specialist (by its system prompt) to a canned criteria response."""

    def __init__(self):
        self.calls = 0

    def create(self, *, system, **kwargs):
        self.calls += 1
        if "PVS1" in system:  # null_variant specialist
            crits = [{"code": "PVS1", "applied": True, "rationale": "frameshift"}]
        elif "PS3" in system:  # functional_literature specialist
            crits = [{"code": "PS3", "applied": True, "rationale": "functional KO"}]
        elif "PM2" in system:  # population_frequency specialist
            crits = [{"code": "PM2", "applied": False, "rationale": "present in gnomAD"}]
        else:  # computational specialist
            crits = [{"code": "PP3", "applied": False, "rationale": "tools discordant"}]
        return _Resp({"criteria": crits})


class FakeClient:
    def __init__(self):
        self.messages = _Messages()


def _variant():
    return Variant(gene="BRCA1", variation_id="42", name="NM_007294.4(BRCA1):c.68_69del")


def test_graph_runs_all_specialists_and_combines_to_pathogenic():
    fake = FakeClient()
    clf = GraphClassifier(model="test-model", client=fake)
    out = clf.classify(_variant(), evidence=[])

    # 4 specialists each made one LLM call.
    assert fake.messages.calls == 4
    # PVS1 (null) + PS3 (functional) -> Pathogenic by ACMG rules.
    assert out.call is ClinSig.PATHOGENIC
    assert out.abstained is False
    # only applied criteria are kept in the evidence trail
    codes = {c.code for c in out.criteria}
    assert codes == {"PVS1", "PS3"}
    # token usage summed across the parallel specialists (4 * 90/40)
    assert out.input_tokens == 360 and out.output_tokens == 160
    assert out.model == "test-model"


def test_graph_abstains_when_no_criteria_apply():
    class _AllUnapplied(_Messages):
        def create(self, *, system, **kwargs):
            self.calls += 1
            return _Resp({"criteria": [{"code": "PP3", "applied": False, "rationale": "x"}]})

    fake = FakeClient()
    fake.messages = _AllUnapplied()
    clf = GraphClassifier(client=fake)
    out = clf.classify(_variant(), evidence=[])
    assert out.call is None and out.abstained is True


def test_graph_uses_evidence_fn_when_no_evidence_passed():
    seen = {}

    def ev_fn(v):
        seen["called"] = v.variation_id
        return [EvidenceItem(source="pubmed", kind="literature", text="study")]

    clf = GraphClassifier(client=FakeClient(), evidence_fn=ev_fn)
    clf.classify(_variant())
    assert seen.get("called") == "42"
