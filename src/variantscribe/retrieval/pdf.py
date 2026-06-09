"""Load guideline PDFs into pages (text layer + optional rendered image).

Text-layer pages feed the dependency-free document retriever; rendered page images feed
the ColPali visual retriever, which can read tables and figures the text layer misses.
pypdfium2 (Apache-2.0) handles both rendering and text extraction with no system deps.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pydantic import BaseModel


class PdfPage(BaseModel):
    doc: str  # source document stem
    page_no: int  # 1-based
    text: str = ""
    image_png: bytes | None = None  # rendered page, only when with_images=True

    @property
    def citation(self) -> str:
        return f"{self.doc} p.{self.page_no}"


def load_pdf_pages(
    path: str | Path, *, with_images: bool = False, render_dpi: int = 120
) -> list[PdfPage]:
    import pypdfium2 as pdfium

    path = Path(path)
    pdf = pdfium.PdfDocument(str(path))
    pages: list[PdfPage] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            text = page.get_textpage().get_text_range().strip()
            image_png = None
            if with_images:
                bitmap = page.render(scale=render_dpi / 72)
                buf = BytesIO()
                bitmap.to_pil().save(buf, format="PNG")
                image_png = buf.getvalue()
            pages.append(
                PdfPage(doc=path.stem, page_no=i + 1, text=text, image_png=image_png)
            )
    finally:
        pdf.close()
    return pages


def load_pdf_dir(
    pdf_dir: str | Path, *, with_images: bool = False, render_dpi: int = 120
) -> list[PdfPage]:
    pdf_dir = Path(pdf_dir)
    pages: list[PdfPage] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages.extend(
            load_pdf_pages(pdf_path, with_images=with_images, render_dpi=render_dpi)
        )
    return pages
