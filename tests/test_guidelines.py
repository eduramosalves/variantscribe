"""Guideline PDF ingestion + text-backend retrieval, with a synthetic PDF (no copyright)."""

from fpdf import FPDF

from variantscribe.models import Variant


def _make_pdf(path, pages: list[str]) -> None:
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


_PAGES = [
    "PVS1 applies to null variants: nonsense, frameshift, and canonical splice changes "
    "causing loss of function in a gene where loss of function is a known disease mechanism.",
    "BA1 is a stand-alone benign criterion for variants with high population allele "
    "frequency above five percent in population databases such as gnomAD.",
]


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("VARIANTSCRIBE_DATA_DIR", str(tmp_path))
    import variantscribe.retrieval.guidelines as gl
    from variantscribe.config import Settings

    monkeypatch.setattr(gl, "settings", Settings())
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf(pdf_dir / "acmg_guideline.pdf", _PAGES)
    return gl, pdf_dir


def test_pdf_text_extraction(tmp_path):
    from variantscribe.retrieval.pdf import load_pdf_pages

    _make_pdf(tmp_path / "g.pdf", _PAGES)
    pages = load_pdf_pages(tmp_path / "g.pdf")
    assert len(pages) == 2
    assert "PVS1" in pages[0].text
    assert pages[1].citation == "g p.2"


def test_build_and_retrieve_guidelines(tmp_path, monkeypatch):
    gl, pdf_dir = _setup(tmp_path, monkeypatch)

    meta = gl.build_guideline_index(pdf_dir, embedder="text")
    assert meta["n_pages"] == 2 and meta["embedder"] == "text"
    assert meta["docs"] == ["acmg_guideline"]

    retriever = gl.load_guideline_retriever(k_final=1)
    fn = gl.guideline_evidence_fn(retriever)
    items = fn(Variant(gene="BRCA1", variation_id="1", name="c.68_69del frameshift"))
    assert items and items[0].kind == "guideline"
    assert items[0].source == "guideline"
    # the loss-of-function page should win for a frameshift variant query
    assert "PVS1" in items[0].text


def test_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("VARIANTSCRIBE_DATA_DIR", str(tmp_path))
    import variantscribe.retrieval.guidelines as gl
    from variantscribe.config import Settings

    monkeypatch.setattr(gl, "settings", Settings())
    try:
        gl.load_guideline_retriever()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
