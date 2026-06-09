"""End-to-end test of the `evaluate` CLI path with a fake Anthropic client.

Exercises sampling → batch classify → eval → confusion/calibration → cost → JSON report,
without an API key or network, against a small synthetic gold file.
"""

import json
from itertools import cycle

from typer.testing import CliRunner

import variantscribe.agent.classifier as clf_mod
from variantscribe.cli import app
from variantscribe.eval.store import write_gold
from variantscribe.models import ClinSig, GoldRecord, Variant

runner = CliRunner()

_TIERS = ["Pathogenic", "Likely pathogenic", "Uncertain significance",
          "Likely benign", "Benign", "Abstain"]


class _Block:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Usage:
    input_tokens = 100
    output_tokens = 50


class _Resp:
    def __init__(self, payload):
        self.content = [_Block(payload)]
        self.usage = _Usage()


class _Messages:
    def __init__(self):
        self._labels = cycle(_TIERS)

    def create(self, **kwargs):
        label = next(self._labels)
        return _Resp({"classification": label, "confidence": 0.7, "rationale": "synthetic"})


class _FakeClient:
    def __init__(self):
        self.messages = _Messages()


def _make_gold(n=18):
    tiers = cycle([ClinSig.PATHOGENIC, ClinSig.BENIGN, ClinSig.VUS,
                   ClinSig.LIKELY_BENIGN, ClinSig.LIKELY_PATHOGENIC])
    return [
        GoldRecord(
            variant=Variant(gene="TEST", variation_id=str(i), name=f"v{i}"),
            gold=next(tiers),
            review_status="reviewed by expert panel",
            gold_stars=3,
        )
        for i in range(n)
    ]


def test_evaluate_cli_end_to_end(tmp_path, monkeypatch):
    # Point data dir at tmp and seed a gold file the CLI will read.
    monkeypatch.setenv("VARIANTSCRIBE_DATA_DIR", str(tmp_path))
    from variantscribe.config import Settings

    settings = Settings()
    monkeypatch.setattr("variantscribe.cli.settings", settings)
    gold_path = settings.raw_dir / "gold_TEST.jsonl"
    write_gold(_make_gold(), gold_path)

    # No API key needed: inject a fake client.
    monkeypatch.setattr(clf_mod, "_make_client", lambda: _FakeClient())

    result = runner.invoke(
        app, ["evaluate", "--gene", "TEST", "--limit", "10", "--model", "claude-sonnet-4-6"]
    )
    assert result.exit_code == 0, result.output
    assert "macro-F1" in result.output
    assert "est. cost" in result.output
    assert "confusion" in result.output

    report_path = settings.runs_dir / "report_agent_claude-sonnet-4-6_TEST.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["model"] == "claude-sonnet-4-6"
    assert report["n"] == 10
    assert "macro_f1" in report and "est_cost_usd" in report
    assert report["input_tokens"] > 0
