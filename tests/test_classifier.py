"""Classifier logic tests with a fake Anthropic client — no API key required."""

from variantscribe.agent.classifier import LLMClassifier, _extract_tool_input, classify_batch
from variantscribe.models import ClinSig, GoldRecord, Variant


class _Block:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Resp:
    def __init__(self, payload, i=120, o=60):
        self.content = [_Block(payload)]
        self.usage = _Usage(i, o)


class _Messages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Resp(self.payload)


class FakeClient:
    def __init__(self, payload):
        self.messages = _Messages(payload)


def _variant(vid="1"):
    return Variant(gene="BRCA1", variation_id=vid, name="NM_007294.4(BRCA1):c.68_69del")


def test_classify_parses_pathogenic():
    payload = {
        "classification": "Pathogenic",
        "confidence": 0.95,
        "criteria": [
            {"code": "PVS1", "applied": True, "strength": "very_strong", "rationale": "frameshift"},
        ],
        "rationale": "Null variant in a LOF gene.",
    }
    clf = LLMClassifier(model="test-model", client=FakeClient(payload))
    out = clf.classify(_variant())
    assert out.call is ClinSig.PATHOGENIC
    assert out.abstained is False
    assert out.confidence == 0.95
    assert out.criteria[0].code == "PVS1"
    assert out.input_tokens == 120 and out.output_tokens == 60
    assert out.latency_ms is not None and out.model == "test-model"


def test_abstain_maps_to_none():
    payload = {"classification": "Abstain", "confidence": 0.2, "rationale": "insufficient"}
    out = LLMClassifier(client=FakeClient(payload)).classify(_variant())
    assert out.call is None
    assert out.abstained is True


def test_unknown_label_is_defensively_an_abstention():
    payload = {"classification": "Probably fine", "confidence": 0.5, "rationale": "x"}
    out = LLMClassifier(client=FakeClient(payload)).classify(_variant())
    assert out.abstained is True and out.call is None


def test_extract_tool_input_requires_tool_block():
    class _NoTool:
        content = [type("B", (), {"type": "text", "text": "hi"})()]

    try:
        _extract_tool_input(_NoTool())
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_classify_batch_preserves_order_and_uses_evidence_fn():
    payload = {"classification": "Benign", "confidence": 0.8, "rationale": "common"}
    clf = LLMClassifier(client=FakeClient(payload))
    records = [GoldRecord(variant=_variant(str(i)), gold=ClinSig.BENIGN,
                          review_status="reviewed by expert panel", gold_stars=3)
               for i in range(5)]

    seen = []
    def ev(v):
        seen.append(v.variation_id)
        return []

    preds = classify_batch(clf, records, evidence_fn=ev, max_workers=2)
    assert [p.variation_id for p in preds] == ["0", "1", "2", "3", "4"]
    assert sorted(seen) == ["0", "1", "2", "3", "4"]
