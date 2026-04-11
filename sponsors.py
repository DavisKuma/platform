import csv
from io import StringIO

import requests
from bs4 import BeautifulSoup

from config import (
    log,
    GOV_UK_CONTENT_API,
    GOV_UK_SPONSORS_PAGE,
)


def normalise_name(name: str) -> str:
    """Lower-case and strip common suffixes for better matching."""
    n = name.lower().strip()
    for suffix in (" ltd", " limited", " plc", " llp", " inc", " corp"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    n = n.replace("&", "and").replace(",", " ").replace(".", " ")
    return " ".join(n.split())


def get_sponsor_csv_url() -> str:
    """Get the latest download URL for the sponsors register from gov.uk."""
    log.info("Fetching latest sponsor register URL from gov.uk ...")

    try:
        resp = requests.get(GOV_UK_CONTENT_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for doc in data.get("details", {}).get("documents", []):
            if isinstance(doc, str):
                soup = BeautifulSoup(doc, "html.parser")
                link = soup.find("a", href=True)
                if link and link["href"].endswith(".csv"):
                    return link["href"]

        for att in data.get("details", {}).get("attachments", []):
            url = att.get("url", "")
            if url.endswith(".csv"):
                return url
    except Exception as e:
        log.warning("Content API approach failed: %s — trying HTML scrape", e)

    resp = requests.get(GOV_UK_SPONSORS_PAGE, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if ".csv" in href.lower() and "worker" in href.lower():
            if href.startswith("/"):
                return f"https://www.gov.uk{href}"
            return href

    raise RuntimeError(
        "Could not find the sponsor register CSV on gov.uk. "
        "Please download it manually from:\n" + GOV_UK_SPONSORS_PAGE
    )


_cached_sponsors: set[str] | None = None
_cached_sponsors_url: str | None = None


def download_sponsor_list() -> set[str]:
    """Download the register and return a set of normalised company names.
    Caches in memory — skips re-download if the CSV URL hasn't changed."""
    global _cached_sponsors, _cached_sponsors_url

    url = get_sponsor_csv_url()

    if _cached_sponsors is not None and _cached_sponsors_url == url:
        log.info("Sponsor list already cached (%d companies, same URL) — skipping download", len(_cached_sponsors))
        return _cached_sponsors

    log.info("Downloading sponsor register from: %s", url)

    resp = requests.get(url, timeout=(15, 180), stream=True)
    resp.raise_for_status()

    chunks = []
    for chunk in resp.iter_content(chunk_size=1024 * 64):
        chunks.append(chunk)
    raw = b"".join(chunks)

    content = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(StringIO(content))

    sponsors = set()
    for row in reader:
        name = row.get("Organisation Name", "").strip()
        if name:
            sponsors.add(normalise_name(name))

    log.info("Loaded %d sponsor-licensed companies", len(sponsors))
    _cached_sponsors = sponsors
    _cached_sponsors_url = url
    return sponsors
