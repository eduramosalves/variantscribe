# 🧬 VariantScribe

[![CI](https://github.com/eduramosalves/variantscribe/actions/workflows/ci.yml/badge.svg)](https://github.com/eduramosalves/variantscribe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**An agentic clinical variant interpretation copilot.** Given a genetic variant,
VariantScribe drafts an [ACMG/AMP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4544753/)
pathogenicity classification — *Pathogenic → Benign* — backed by a cited evidence trail,
and is evaluated against **ClinVar expert-reviewed gold labels**.

> ⚕️ **Decision support, not diagnosis.** VariantScribe drafts a reviewable assessment for
> a qualified molecular geneticist. It is a research/engineering portfolio project, not a
> clinical device, and includes an explicit *abstain* path when evidence is insufficient.

---

## Why this exists

Variant interpretation is the bottleneck in clinical genomics: labs carry backlogs of
thousands of variants, each taking 30–60 min of manual literature and database review, and
classifications drift as new evidence appears. VariantScribe drafts the criterion-by-criterion
assessment a geneticist would assemble by hand — and, crucially, **measures its own accuracy
against ground truth** so the quality is provable rather than asserted.

## What makes it an *engineering* project (not an API wrapper)

- **Real data, joined across sources** — ClinVar (variants + gold labels), gnomAD
  (population frequency), PubMed (literature). No synthetic toy data.
- **Ground-truth evaluation** — ClinVar's review-status *star rating* gives a trustworthy
  gold set; the harness reports macro-F1, a **dangerous-error rate** (calling a pathogenic
  variant benign), ordinal distance, abstention/coverage, and calibration.
- **Retrieval that's measured, not assumed** — biomedical embeddings + cross-encoder
  reranking, benchmarked against a dense-only baseline.
- **Production patterns** — typed domain models, pluggable vector store, retrying
  rate-limited API clients, JSONL run artifacts, and a CI eval gate that blocks metric
  regressions.

## HuggingFace tasks used

Token Classification (biomedical NER) · Text Ranking (cross-encoder reranking) ·
Feature Extraction (domain embeddings) · Summarization · Document QA + Visual Document
Retrieval (guideline PDFs, *Phase 3*) · Zero-Shot Classification (query triage).

## Architecture (Phase 1 spine)

```
            ┌────────── ingest/ ──────────┐
ClinVar ───▶│ gold labels + variants      │
gnomAD  ───▶│ population frequency         │──▶ retrieval/ (embed + rerank)
PubMed  ───▶│ literature corpus            │            │
            └─────────────────────────────┘            ▼
                                              agent/ classifier ──▶ Classification
                                                                         │
                              ClinVar gold ──▶ eval/ (metrics) ◀─────────┘
```

Phases: **(1)** single-gene spine + baseline eval → **(2)** LangGraph multi-agent
(one node per ACMG criterion) + Langfuse tracing + CI gate → **(3)** ColPali visual
retrieval over guideline PDFs + calibration analysis + UI.

## Tech stack

`Python 3.11+` · `httpx` · `pydantic` · `LanceDB` (file-based vector store; pgvector swap-in
planned) · `sentence-transformers` (MedCPT embeddings + cross-encoder) · `Anthropic` ·
`scikit-learn` · `typer`. Managed with [`uv`](https://docs.astral.sh/uv/).

## Quickstart

```bash
uv sync                                   # base deps (ingestion + eval)
uv sync --extra retrieval --extra agent   # add ML + LLM deps
cp .env.example .env                       # set NCBI email (required) + keys

# Build a trustworthy gold set from ClinVar (≥2★ reviewed variants)
uv run variantscribe build-gold --gene BRCA1 --min-stars 2

# Build the literature retrieval index (hashing = no torch; medcpt = production)
uv run variantscribe build-index --gene BRCA1 --embedder hashing --max-articles 400
uv sync --extra medcpt        # then: --embedder medcpt for MedCPT encoders

# Classify + evaluate, measuring the lift from evidence (needs an Anthropic key)
uv run variantscribe evaluate --gene BRCA1 --limit 50 --evidence none
uv run variantscribe evaluate --gene BRCA1 --limit 50 --evidence retrieval --no-rerank
uv run variantscribe evaluate --gene BRCA1 --limit 50 --evidence retrieval   # + rerank
```

## Status

🚧 **Phase 3 in progress** (Phases 1–2 complete). Done: data ingestion (ClinVar/gnomAD/
PubMed), typed domain models, gold-set builder, eval-metrics harness, trivial baselines, the
**single-agent LLM classifier** (Anthropic tool-use → structured ACMG output, abstain path,
concurrent batch, token/cost/latency telemetry), the **two-stage retrieval layer** (PubMed
corpus → embeddings → LanceDB vector index → cross-encoder rerank), the **LangGraph
multi-agent classifier**, a **GitHub Actions CI eval gate**, **calibration analysis** (ECE +
reliability bins), and **visual guideline-PDF retrieval** (ColPali). 58 tests; everything is
covered without torch or an API key (a dependency-free hashing embedder + a mocked LLM
client + a synthetic PDF fixture).

### Phase 3 — guideline-PDF (visual) retrieval + calibration

ACMG/VCEP guideline PDFs are ingested page-by-page and retrieved as evidence alongside the
literature, via two interchangeable backends:

- **text** — page text-layer → LanceDB (dependency-free; runs anywhere).
- **colpali** — page *image* → ColPali multi-vector → **MaxSim late interaction**, so
  layout, tables, and figures are searchable, not just the text layer. (`colpali` extra.)

```bash
uv run variantscribe build-guidelines --pdf-dir ./guidelines --embedder text   # or colpali
uv run variantscribe evaluate --gene BRCA1 --evidence retrieval --guidelines
```

No copyrighted guideline PDFs are committed — point `--pdf-dir` at PDFs you hold a licence to.
The eval harness now also reports **Expected Calibration Error** and a reliability table:
does the classifier's confidence track its real accuracy?

Remaining (Phase 3): a thin review UI, and committing real agent-vs-graph / evidence-lift /
calibration numbers (needs an API key + the MedCPT/ColPali extras).

### Phase 2 — multi-agent ACMG classifier

Four specialist nodes run **in parallel**, each owning one slice of the ACMG criteria and
reasoning only over that slice; their criteria are merged and a **deterministic combiner**
applies the published ACMG rules (Richards et al. 2015) to decide the final tier — a
transparent, reviewable function, not a black-box model output.

```
START ─┬─▶ null_variant (PVS1) ──────────────────┐
       ├─▶ population_frequency (BA1/BS1/BS2/PM2) ─┤
       ├─▶ computational (PP3/BP4/BP1/BP7) ────────┼─▶ combine (ACMG rules) ─▶ Classification
       └─▶ functional_literature (PS3/PS1/…) ──────┘
```

It outputs the same `Classification` type as the single-agent path, so the eval harness,
baselines, and CI gate guard it unchanged. Pick the classifier with
`evaluate --classifier agent|graph`. **Langfuse tracing** is wired in (opt-in: a no-op
unless `VARIANTSCRIBE_LANGFUSE_*` keys are set), giving per-node traces and token costs.

The combiner is conservative by clinical-safety design: any co-occurrence of pathogenic
**and** benign criteria is flagged *Uncertain* for human review, never silently called
benign.

**Retrieval design:** a pluggable `Embedder` (production: NCBI **MedCPT** asymmetric
article/query encoders; fallback: deterministic hashing) and `Reranker` (MedCPT
cross-encoder vs. a no-op ablation). Swapping LanceDB → pgvector is the planned Phase-2
data-layer milestone. Retrieval quality is measured *downstream* — classification macro-F1
with `--evidence none` vs `retrieval` vs `retrieval --no-rerank` — rather than via
unlabelled IR metrics.

The multi-agent classifier (Phase 2) consumes this same retrieval evidence — see the
multi-agent section below. Remaining: run the agent-vs-graph and evidence-lift measurements
with MedCPT + a real LLM key.

> **Honest caveat:** the no-evidence run is an LLM-only ablation on a heavily-documented
> gene (BRCA1), so some apparent skill may reflect training-data familiarity with ClinVar.
> The retrieval phase — and held-out / less-documented genes — separate genuine evidence
> reasoning from recall. This is tracked explicitly rather than hidden.

### Baselines to beat (BRCA1, 800 gold variants, live ClinVar)

| Classifier | Macro-F1 |
|------------|----------|
| always-VUS | 0.080 |
| majority (always Pathogenic) | 0.121 |
| LLM agent (no evidence) | *run with an API key* |

## License

MIT © Eduardo Ramos Alves
