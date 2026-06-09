"""gnomAD ingestion: population allele frequencies via the public GraphQL API.

Frequency is the backbone of several ACMG criteria: a high popmax AF supports benign
(BA1/BS1), while absence from large cohorts is supporting-pathogenic (PM2).
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from variantscribe.models import GnomadFrequency

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

# Query by rsID against a gnomAD dataset; returns joint/exome/genome frequencies.
_VARIANT_QUERY = """
query VariantFreq($rsid: String!, $dataset: DatasetId!) {
  variant(rsid: $rsid, dataset: $dataset) {
    genome { ac an af populations { id ac an } }
    exome  { ac an af populations { id ac an } }
  }
}
"""


def _popmax(populations: list[dict]) -> tuple[float | None, str | None]:
    best_af, best_pop = None, None
    for p in populations or []:
        pid = p.get("id", "")
        # Skip sex- and subcontinental aggregate buckets; keep top-level ancestries.
        if "_" in pid or pid in {"XX", "XY"}:
            continue
        an = p.get("an") or 0
        ac = p.get("ac") or 0
        if an <= 0:
            continue
        af = ac / an
        if best_af is None or af > best_af:
            best_af, best_pop = af, pid
    return best_af, best_pop


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def fetch_frequency(
    rsid: str,
    *,
    dataset: str = "gnomad_r4",
    client: httpx.Client | None = None,
) -> GnomadFrequency:
    """Fetch allele frequency for a variant by rsID. Returns found=False if absent —
    which is itself evidence (PM2)."""
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(
            GNOMAD_API,
            json={"query": _VARIANT_QUERY, "variables": {"rsid": rsid, "dataset": dataset}},
        )
        resp.raise_for_status()
        payload = resp.json()
        variant = (payload.get("data") or {}).get("variant")
        if not variant:
            return GnomadFrequency(found=False)

        # Prefer the larger genome callset, fall back to exome.
        block = variant.get("genome") or variant.get("exome") or {}
        af = block.get("af")
        popmax_af, popmax_pop = _popmax(block.get("populations") or [])
        return GnomadFrequency(
            allele_frequency=af,
            popmax_af=popmax_af,
            popmax_population=popmax_pop,
            allele_count=block.get("ac"),
            allele_number=block.get("an"),
            found=af is not None,
        )
    finally:
        if own:
            client.close()
