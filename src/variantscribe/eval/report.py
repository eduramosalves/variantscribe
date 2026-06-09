"""Render eval runs into a Markdown results table, a reliability plot, and splice them
into the README between markers — so `scripts/run_results.py` keeps the published numbers
in sync with the code.
"""

from __future__ import annotations

from pathlib import Path

_COLUMNS = [
    ("classifier", "Classifier"),
    ("evidence", "Evidence"),
    ("n", "n"),
    ("macro_f1", "macro-F1"),
    ("three_class_accuracy", "3-class acc"),
    ("dangerous_error_rate", "dangerous-err"),
    ("ece", "ECE"),
    ("cost_per_1k", "$/1k"),
]


def _fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if key in {"macro_f1", "three_class_accuracy", "ece"}:
        return f"{value:.3f}"
    if key == "dangerous_error_rate":
        return f"{value:.1%}"
    if key == "cost_per_1k":
        return f"${value:.2f}"
    return str(value)


def results_table(rows: list[dict]) -> str:
    """Render rows (dicts keyed by the column ids above) as a GitHub Markdown table.
    Missing values render as '—', so pending (un-run) rows are fine."""
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [_fmt(key, row.get(key)) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inject_between_markers(text: str, name: str, payload: str) -> str:
    """Replace the content between `<!-- {name}:start -->` and `<!-- {name}:end -->`."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.find(start), text.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"markers for {name!r} not found (or out of order) in target text")
    return text[: i + len(start)] + "\n" + payload + "\n" + text[j:]


def reliability_plot(bins: list[dict], path: str | Path, *, title: str | None = None) -> Path:
    """Scatter mean-confidence vs accuracy per bin against the perfect-calibration diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    if bins:
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        sizes = [20 + 6 * b["n"] for b in bins]
        ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.8, label="observed")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Calibration reliability")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
