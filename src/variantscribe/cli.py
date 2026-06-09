"""VariantScribe command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from variantscribe.agent.baseline import baseline_predictions, majority_label
from variantscribe.config import settings
from variantscribe.eval.metrics import EvalReport, evaluate
from variantscribe.eval.sampling import stratified_sample
from variantscribe.eval.store import (
    build_eval_cases,
    read_gold,
    write_gold,
    write_predictions,
    write_report_json,
)
from variantscribe.ingest.clinvar import build_gold_records
from variantscribe.models import ClinSig

app = typer.Typer(help="Agentic clinical variant interpretation copilot.", no_args_is_help=True)
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _gold_path(gene: str) -> Path:
    return settings.raw_dir / f"gold_{gene.upper()}.jsonl"


@app.command("build-gold")
def build_gold(
    gene: str = typer.Option(..., "--gene", "-g", help="Gene symbol, e.g. BRCA1"),
    min_stars: int = typer.Option(2, "--min-stars", help="Minimum ClinVar review stars (2-4)"),
    max_variants: int = typer.Option(400, "--max-variants", help="Cap per review-status tier"),
) -> None:
    """Fetch a trustworthy gold set for GENE from ClinVar and save it as JSONL."""
    settings.ensure_dirs()
    console.print(f"[bold]Building gold set[/] for [cyan]{gene}[/] (≥{min_stars}★)…")
    gold = build_gold_records(gene, min_stars=min_stars, max_variants=max_variants)
    if not gold:
        console.print("[red]No gold records found.[/] Try lowering --min-stars.")
        raise typer.Exit(1)

    path = _gold_path(gene)
    write_gold(gold, path)

    counts = {c: 0 for c in ClinSig}
    stars: dict[int, int] = {}
    for g in gold:
        counts[g.gold] += 1
        stars[g.gold_stars] = stars.get(g.gold_stars, 0) + 1

    table = Table(title=f"{gene} gold set — {len(gold)} variants")
    table.add_column("ACMG tier")
    table.add_column("n", justify="right")
    for tier in ClinSig:
        table.add_row(tier.value, str(counts[tier]))
    console.print(table)
    console.print(f"stars: {stars}")
    console.print(f"[green]saved[/] → {path}")


@app.command("eval-baseline")
def eval_baseline(
    gene: str = typer.Option(..., "--gene", "-g"),
    strategy: str = typer.Option("majority", "--strategy", help="majority | always-vus"),
) -> None:
    """Score a trivial baseline against GENE's gold set — the floor to beat."""
    path = _gold_path(gene)
    if not path.exists():
        console.print(f"[red]No gold set at {path}.[/] Run `build-gold --gene {gene}` first.")
        raise typer.Exit(1)

    gold = read_gold(path)
    preds = baseline_predictions(gold, strategy)
    settings.ensure_dirs()
    run_path = settings.runs_dir / f"baseline_{strategy}_{gene.upper()}.jsonl"
    write_predictions(preds, run_path)

    report = evaluate(build_eval_cases(gold, preds))
    console.print(f"\n[bold]Baseline[/] [cyan]{strategy}[/] on [cyan]{gene}[/]:")
    for line in report.summary_lines():
        console.print("  " + line)
    console.print(f"[green]predictions saved[/] → {run_path}")


def _print_confusion(report: EvalReport) -> None:
    if not report.confusion:
        return
    short = {"Pathogenic": "P", "Likely pathogenic": "LP", "Uncertain significance": "VUS",
             "Likely benign": "LB", "Benign": "B"}
    table = Table(title="confusion (rows=gold, cols=predicted)")
    table.add_column("gold↓ / pred→")
    for tier in ClinSig:
        table.add_column(short[tier.value], justify="right")
    for g in ClinSig:
        row = [short[g.value]]
        for p in ClinSig:
            n = report.confusion[g.value][p.value]
            row.append(f"[bold]{n}[/]" if g == p and n else str(n))
        table.add_row(*row)
    console.print(table)


@app.command("build-index")
def build_index_cmd(
    gene: str = typer.Option(..., "--gene", "-g"),
    embedder: str = typer.Option("medcpt", "--embedder", help="medcpt | hashing"),
    max_articles: int = typer.Option(400, "--max-articles"),
) -> None:
    """Build the literature retrieval index for GENE (PubMed → embeddings → LanceDB)."""
    from variantscribe.retrieval.pipeline import build_index

    console.print(
        f"Building [cyan]{embedder}[/] index for [cyan]{gene}[/] "
        f"(up to {max_articles} PubMed articles)…"
    )
    try:
        meta = build_index(gene, embedder_name=embedder, max_articles=max_articles)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]indexed[/] {meta['n_passages']} passages · dim={meta['dim']} · "
        f"embed {meta['embed_seconds']}s → {settings.index_dir}"
    )


@app.command("evaluate")
def evaluate_agent(
    gene: str = typer.Option(..., "--gene", "-g"),
    limit: int | None = typer.Option(None, "--limit", help="Stratified sample size"),
    evidence: str = typer.Option("none", "--evidence", help="none | pubmed | retrieval"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Cross-encoder rerank"),
    k: int = typer.Option(5, "--k", help="Evidence passages passed to the classifier"),
    classifier: str = typer.Option("agent", "--classifier", help="agent | graph"),
    model: str | None = typer.Option(None, "--model", help="Override the agent model"),
    max_workers: int = typer.Option(4, "--max-workers"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run the LLM classifier over GENE's gold set, score it, and report cost."""
    from variantscribe.agent.classifier import LLMClassifier, classify_batch
    from variantscribe.agent.evidence import pubmed_evidence, retrieval_evidence_fn
    from variantscribe.agent.telemetry import summarize_run
    from variantscribe.agent.tracing import flush as flush_tracing
    from variantscribe.agent.tracing import tracing_enabled

    path = _gold_path(gene)
    if not path.exists():
        console.print(f"[red]No gold set at {path}.[/] Run `build-gold --gene {gene}` first.")
        raise typer.Exit(1)

    gold = read_gold(path)
    if limit:
        gold = stratified_sample(gold, limit, seed=seed)

    evidence_fn = None
    evidence_label = evidence
    if evidence == "pubmed":
        evidence_fn = lambda v: pubmed_evidence(v, max_items=k)  # noqa: E731
    elif evidence == "retrieval":
        from variantscribe.retrieval.pipeline import load_retriever

        try:
            retriever = load_retriever(gene, rerank=rerank, k_final=k)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        evidence_fn = retrieval_evidence_fn(retriever)
        evidence_label = f"retrieval(rerank={rerank}, k={k})"
    elif evidence != "none":
        console.print(f"[red]Unknown --evidence {evidence!r}[/] (none | pubmed | retrieval)")
        raise typer.Exit(1)

    # `graph` is the LangGraph multi-agent classifier; it gathers evidence internally,
    # so we hand the evidence_fn to it and pass None to classify_batch.
    batch_evidence_fn = evidence_fn
    try:
        if classifier == "graph":
            from variantscribe.agent.graph import GraphClassifier

            clf = GraphClassifier(model=model, evidence_fn=evidence_fn)
            batch_evidence_fn = None
        elif classifier == "agent":
            clf = LLMClassifier(model=model)
        else:
            console.print(f"[red]Unknown --classifier {classifier!r}[/] (agent | graph)")
            raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    trace_note = " [dim](langfuse tracing on)[/]" if tracing_enabled() else ""
    console.print(
        f"Classifying [cyan]{len(gold)}[/] {gene} variants with "
        f"[cyan]{classifier}:{clf.model}[/] (evidence: {evidence_label}){trace_note}…"
    )
    with Progress(transient=True) as progress:
        task = progress.add_task("classifying", total=len(gold))

        def _tick(done: int, total: int) -> None:
            progress.update(task, completed=done)

        preds = classify_batch(
            clf, gold, evidence_fn=batch_evidence_fn, max_workers=max_workers, on_done=_tick
        )

    settings.ensure_dirs()
    tag = f"{classifier}_{(model or clf.model)}".replace("/", "-")
    run_path = settings.runs_dir / f"{tag}_{gene.upper()}.jsonl"
    write_predictions(preds, run_path)

    report = evaluate(build_eval_cases(gold, preds))
    telem = summarize_run(preds, model=clf.model)

    console.print(f"\n[bold]{classifier}[/] [cyan]{clf.model}[/] on [cyan]{gene}[/]:")
    for line in report.summary_lines():
        console.print("  " + line)
    console.print("  [dim]— cost/latency —[/]")
    for line in telem.summary_lines():
        console.print("  " + line)

    # Context: did it beat the trivial floor?
    floor = evaluate(build_eval_cases(gold, baseline_predictions(gold, "majority")))
    delta = report.macro_f1 - floor.macro_f1
    verdict = "[green]beats[/]" if delta > 0 else "[red]below[/]"
    console.print(
        f"  [dim]majority-baseline macro-F1 = {floor.macro_f1:.3f}; "
        f"agent {verdict} floor by {delta:+.3f}[/]"
    )
    _print_confusion(report)
    if report.calibration:
        console.print("calibration (confidence → accuracy):")
        for rng, n, acc in report.calibration:
            console.print(f"  {rng}: n={n}, acc={acc:.2f}")

    report_path = settings.runs_dir / f"report_{tag}_{gene.upper()}.json"
    write_report_json(
        {
            "gene": gene,
            "classifier": classifier,
            "model": clf.model,
            "evidence": evidence_label,
            "n": report.n_total,
            **{k: getattr(report, k) for k in (
                "coverage", "accuracy", "macro_f1", "three_class_accuracy",
                "ordinal_mae", "dangerous_errors", "dangerous_error_rate",
            )},
            "majority_macro_f1": floor.macro_f1,
            "est_cost_usd": telem.est_cost_usd,
            "input_tokens": telem.input_tokens,
            "output_tokens": telem.output_tokens,
        },
        report_path,
    )
    console.print(f"[green]saved[/] predictions → {run_path}")
    console.print(f"[green]saved[/] report → {report_path}")
    flush_tracing()  # push any buffered Langfuse traces


# `majority_label` is imported for parity with baseline reporting/tests.
_ = majority_label


if __name__ == "__main__":
    app()
