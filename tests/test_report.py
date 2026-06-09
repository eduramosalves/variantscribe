import pytest

from variantscribe.eval.report import (
    inject_between_markers,
    reliability_plot,
    results_table,
)


def test_results_table_formats_and_handles_missing():
    rows = [
        {"classifier": "baseline", "evidence": "majority", "n": 60, "macro_f1": 0.121,
         "three_class_accuracy": 0.45, "dangerous_error_rate": 0.0, "ece": None,
         "cost_per_1k": 0.0},
        {"classifier": "graph", "evidence": "retrieval", "n": None},  # pending row
    ]
    table = results_table(rows)
    assert "| Classifier | Evidence |" in table
    assert "0.121" in table
    assert "0.0%" in table
    assert "$0.00" in table
    # pending row renders missing values as em dashes
    pending = table.splitlines()[-1]
    assert pending.count("—") >= 4


def test_inject_between_markers_replaces_content():
    text = "a\n<!-- x:start -->\nOLD\n<!-- x:end -->\nb"
    out = inject_between_markers(text, "x", "NEW")
    assert "NEW" in out and "OLD" not in out
    assert "<!-- x:start -->" in out and "<!-- x:end -->" in out
    assert out.startswith("a\n") and out.endswith("\nb")


def test_inject_between_markers_missing_raises():
    with pytest.raises(ValueError):
        inject_between_markers("no markers here", "x", "NEW")


def test_reliability_plot_writes_png(tmp_path):
    bins = [
        {"label": "0.4-0.6", "n": 3, "mean_confidence": 0.5, "accuracy": 0.33},
        {"label": "0.8-1.0", "n": 10, "mean_confidence": 0.9, "accuracy": 0.7},
    ]
    out = reliability_plot(bins, tmp_path / "cal.png", title="test")
    assert out.exists() and out.stat().st_size > 0


def test_reliability_plot_handles_empty_bins(tmp_path):
    out = reliability_plot([], tmp_path / "empty.png")
    assert out.exists()
