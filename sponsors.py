import csv
import os
import json
from io import StringIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import (
    log,
    GOV_UK_CONTENT_API,
    GOV_UK_SPONSORS_PAGE,
)

# Disk cache directory
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_CSV = _CACHE_DIR / "sponsors.csv"
_CACHE_META = _CACHE_DIR / "sponsors_meta.json"


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


def _parse_sponsors_csv(content: str) -> set[str]:
    """Parse sponsor CSV content into a set of normalised company names."""
    reader = csv.DictReader(StringIO(content))
    sponsors = set()
    for row in reader:
        name = row.get("Organisation Name", "").strip()
        if name:
            sponsors.add(normalise_name(name))
    return sponsors


def _load_cached_meta() -> dict:
    """Load cached metadata (URL, Last-Modified, ETag)."""
    try:
        if _CACHE_META.exists():
            return json.loads(_CACHE_META.read_text())
    except Exception:
        pass
    return {}


def _save_cache(content: str, url: str, last_modified: str | None, etag: str | None):
    """Save CSV and metadata to disk cache."""
    _CACHE_DIR.mkdir(exist_ok=True)
    _CACHE_CSV.write_text(content, encoding="utf-8")
    _CACHE_META.write_text(json.dumps({
        "url": url,
        "last_modified": last_modified,
        "etag": etag,
    }))


def download_sponsor_list() -> set[str]:
    """Download the register and return a set of normalised company names.

    Uses disk cache + HTTP conditional requests (If-Modified-Since / If-None-Match)
    so the CSV is only re-downloaded when gov.uk publishes a new version.
    """
    global _cached_sponsors

    # 1. Return in-memory cache if available
    if _cached_sponsors is not None:
        log.info("Sponsor list in memory (%d companies) — skipping download", len(_cached_sponsors))
        return _cached_sponsors

    url = get_sponsor_csv_url()
    meta = _load_cached_meta()

    # 2. Try conditional request if we have a disk cache for the same URL
    if _CACHE_CSV.exists() and meta.get("url") == url:
        headers = {}
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]

        if headers:
            try:
                resp = requests.get(url, headers=headers, timeout=(15, 30), stream=True)
                if resp.status_code == 304:
                    log.info("Sponsor register not modified (304) — using disk cache")
                    content = _CACHE_CSV.read_text(encoding="utf-8")
                    _cached_sponsors = _parse_sponsors_csv(content)
                    log.info("Loaded %d sponsor-licensed companies from cache", len(_cached_sponsors))
                    return _cached_sponsors
            except Exception as e:
                log.warning("Conditional request failed: %s — will try full download", e)

        # Same URL but no conditional headers — just load from disk
        if not headers:
            log.info("Loading sponsor register from disk cache")
            content = _CACHE_CSV.read_text(encoding="utf-8")
            _cached_sponsors = _parse_sponsors_csv(content)
            log.info("Loaded %d sponsor-licensed companies from cache", len(_cached_sponsors))
            return _cached_sponsors

    # 3. Full download
    log.info("Downloading sponsor register from: %s", url)
    resp = requests.get(url, timeout=(15, 180), stream=True)
    resp.raise_for_status()

    chunks = []
    for chunk in resp.iter_content(chunk_size=1024 * 64):
        chunks.append(chunk)
    raw = b"".join(chunks)

    content = raw.decode("utf-8-sig", errors="replace")
    _cached_sponsors = _parse_sponsors_csv(content)
    log.info("Loaded %d sponsor-licensed companies", len(_cached_sponsors))

    # Save to disk cache
    last_modified = resp.headers.get("Last-Modified")
    etag = resp.headers.get("ETag")
    _save_cache(content, url, last_modified, etag)
    log.info("Sponsor register cached to disk")

    return _cached_sponsors
