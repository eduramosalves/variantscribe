"""Thin client for NCBI E-utilities (Entrez), shared by ClinVar and PubMed.

Handles the things NCBI's usage policy and reliability demand: identifying tool+email
on every call, an optional API key, client-side rate limiting (3 req/s without a key,
10 with), and retries with backoff on transient 429/5xx responses.
"""

from __future__ import annotations

import threading
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from variantscribe.config import settings

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "variantscribe"


class _RateLimiter:
    """Minimal thread-safe spacing between requests."""

    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


class EutilsClient:
    def __init__(self, timeout: float = 30.0) -> None:
        rate = 10.0 if settings.ncbi_api_key else 3.0
        self._limiter = _RateLimiter(rate)
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout)

    def _common_params(self) -> dict[str, str]:
        params = {"tool": TOOL_NAME, "email": settings.ncbi_email}
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key
        return params

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1.5, min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> httpx.Response:
        self._limiter.wait()
        resp = self._client.get(endpoint, params={**self._common_params(), **params})
        # 429 / 5xx are worth retrying; raise so tenacity catches them.
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()
        resp.raise_for_status()
        return resp

    def esearch(self, db: str, term: str, retmax: int = 200, retstart: int = 0) -> list[str]:
        """Return a list of UIDs matching `term` in `db`."""
        params = {
            "db": db,
            "term": term,
            "retmax": str(retmax),
            "retstart": str(retstart),
            "retmode": "json",
        }
        data = self._get("/esearch.fcgi", params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def esummary(self, db: str, ids: list[str]) -> dict:
        """Return the `result` map of document summaries for the given UIDs."""
        if not ids:
            return {}
        params = {"db": db, "id": ",".join(ids), "retmode": "json"}
        return self._get("/esummary.fcgi", params).json().get("result", {})

    def efetch_text(self, db: str, ids: list[str], rettype: str, retmode: str = "xml") -> str:
        """Return raw efetch payload (XML or text) for the given UIDs."""
        if not ids:
            return ""
        params = {"db": db, "id": ",".join(ids), "rettype": rettype, "retmode": retmode}
        return self._get("/efetch.fcgi", params).text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EutilsClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
