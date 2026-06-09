"""PubMed ingestion: search and fetch abstracts to build the literature evidence corpus."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic import BaseModel

from variantscribe.ingest.eutils import EutilsClient


class Article(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: int | None = None

    @property
    def citation(self) -> str:
        return f"PMID:{self.pmid}"


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _parse_articles(xml_text: str) -> list[Article]:
    if not xml_text.strip():
        return []
    root = ET.fromstring(xml_text)
    articles: list[Article] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))
        # Abstracts can be split into labelled sections; join them in order.
        sections = art.findall(".//Abstract/AbstractText")
        abstract = " ".join(_text(s) for s in sections).strip()
        journal = _text(art.find(".//Journal/Title")) or None
        year_txt = _text(art.find(".//JournalIssue/PubDate/Year"))
        year = int(year_txt) if year_txt.isdigit() else None
        if pmid and (title or abstract):
            articles.append(
                Article(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    journal=journal,
                    year=year,
                )
            )
    return articles


def search_articles(
    query: str,
    *,
    retmax: int = 50,
    client: EutilsClient | None = None,
) -> list[Article]:
    """Search PubMed and return parsed articles (only those with title/abstract)."""
    own = client is None
    client = client or EutilsClient()
    try:
        pmids = client.esearch("pubmed", query, retmax=retmax)
        if not pmids:
            return []
        articles: list[Article] = []
        for i in range(0, len(pmids), 100):
            chunk = pmids[i : i + 100]
            xml_text = client.efetch_text("pubmed", chunk, rettype="abstract", retmode="xml")
            articles.extend(_parse_articles(xml_text))
        return articles
    finally:
        if own:
            client.close()


def gene_literature_query(gene: str, extra: str | None = None) -> str:
    """A focused query for clinically relevant literature on a gene's variants."""
    base = (
        f"{gene}[Title/Abstract] AND "
        "(variant[Title/Abstract] OR mutation[Title/Abstract] OR pathogenic[Title/Abstract])"
    )
    return f"{base} AND {extra}" if extra else base
