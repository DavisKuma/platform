"""
Adzuna UK Entry-Level Job Scraper
Finds jobs posted in the last 24 hours from UK Sponsorship License companies.

Usage:
    1. Get free API credentials at https://developer.adzuna.com/
    2. Copy .env.example to .env and fill in your credentials
    3. pip install -r requirements.txt
    4. python scraper.py
"""

import os
import csv
import math
import time
import logging
from io import StringIO
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search"
RESULTS_PER_PAGE = 50
FUZZY_MATCH_THRESHOLD = 85  # minimum similarity score (0-100)

ENTRY_LEVEL_KEYWORDS = "junior graduate trainee apprentice entry-level"
SALARY_MAX = 35000  # GBP — cap to target entry-level roles

GOV_UK_SPONSORS_PAGE = (
    "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"
)
GOV_UK_CONTENT_API = (
    "https://www.gov.uk/api/content/government/publications/"
    "register-of-licensed-sponsors-workers"
)

OUTPUT_CSV = f"adzuna_sponsored_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ── 1. Download Sponsorship License Register ────────────────────────────────
def get_sponsor_csv_url() -> str:
    """Get the latest download URL for the sponsors register from gov.uk."""
    log.info("Fetching latest sponsor register URL from gov.uk ...")

    # Try the Content API first
    try:
        resp = requests.get(GOV_UK_CONTENT_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Look through documents/attachments for the CSV
        for doc in data.get("details", {}).get("documents", []):
            if isinstance(doc, str):
                soup = BeautifulSoup(doc, "html.parser")
                link = soup.find("a", href=True)
                if link and link["href"].endswith(".csv"):
                    return link["href"]

        # Fallback: check attachments list
        for att in data.get("details", {}).get("attachments", []):
            url = att.get("url", "")
            if url.endswith(".csv"):
                return url
    except Exception as e:
        log.warning("Content API approach failed: %s — trying HTML scrape", e)

    # Fallback: scrape the HTML page
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


def download_sponsor_list() -> set[str]:
    """Download the register and return a set of normalised company names."""
    url = get_sponsor_csv_url()
    log.info("Downloading sponsor register from: %s", url)

    # Stream download — the file is large (~125k rows)
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
    return sponsors


def normalise_name(name: str) -> str:
    """Lower-case and strip common suffixes for better matching."""
    n = name.lower().strip()
    for suffix in (" ltd", " limited", " plc", " llp", " inc", " corp"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # normalise punctuation
    n = n.replace("&", "and").replace(",", " ").replace(".", " ")
    # collapse whitespace
    return " ".join(n.split())


# ── 2. Query Adzuna API ─────────────────────────────────────────────────────
def fetch_adzuna_jobs() -> list[dict]:
    """Fetch all entry-level UK jobs posted in the last 24 hours."""
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        raise RuntimeError(
            "Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY "
            "in your .env file. Get free keys at https://developer.adzuna.com/"
        )

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "max_days_old": 1,
        "what_or": ENTRY_LEVEL_KEYWORDS,
        "salary_max": SALARY_MAX,
        "sort_by": "date",
        "content-type": "application/json",
    }

    all_jobs = []
    page = 1

    # First request to get total count
    log.info("Querying Adzuna API for entry-level UK jobs (last 24h) ...")
    resp = requests.get(f"{ADZUNA_BASE_URL}/{page}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    total = data.get("count", 0)
    total_pages = min(math.ceil(total / RESULTS_PER_PAGE), 20)  # cap at 20 pages
    log.info("Found %d total jobs across %d pages", total, total_pages)

    all_jobs.extend(data.get("results", []))

    # Fetch remaining pages
    for page in range(2, total_pages + 1):
        time.sleep(0.3)  # respect rate limits
        resp = requests.get(f"{ADZUNA_BASE_URL}/{page}", params=params, timeout=30)
        resp.raise_for_status()
        all_jobs.extend(resp.json().get("results", []))
        log.info("  Fetched page %d/%d (%d jobs so far)", page, total_pages, len(all_jobs))

    log.info("Retrieved %d jobs from Adzuna", len(all_jobs))
    return all_jobs


# ── 3. Match jobs to sponsors ───────────────────────────────────────────────
def match_jobs_to_sponsors(
    jobs: list[dict], sponsors: set[str]
) -> list[dict]:
    """Cross-reference Adzuna jobs with sponsorship license holders."""
    matched = []
    sponsors_list = list(sponsors)  # for fuzzy search

    log.info("Cross-referencing %d jobs against %d sponsors ...", len(jobs), len(sponsors))

    for i, job in enumerate(jobs):
        company_raw = (job.get("company") or {}).get("display_name", "")
        if not company_raw:
            continue

        company_norm = normalise_name(company_raw)

        # Exact match first (fast O(1) set lookup)
        if company_norm in sponsors:
            matched.append(build_row(job, company_raw, "exact"))
            continue

        # Fuzzy match using rapidfuzz extractOne (much faster than manual loop)
        result = process.extractOne(
            company_norm, sponsors_list,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if result:
            _, score, _ = result
            matched.append(build_row(job, company_raw, f"fuzzy ({score:.0f}%)"))

        if (i + 1) % 500 == 0:
            log.info("  Processed %d/%d jobs (%d matched so far)", i + 1, len(jobs), len(matched))

    log.info(
        "Matched %d / %d jobs to sponsorship license companies", len(matched), len(jobs)
    )
    return matched


def build_row(job: dict, company: str, match_type: str) -> dict:
    """Extract a flat dict from an Adzuna job result."""
    location = (job.get("location") or {}).get("display_name", "")
    category = (job.get("category") or {}).get("label", "")
    salary_min = job.get("salary_min", "")
    salary_max = job.get("salary_max", "")
    salary_predicted = job.get("salary_is_predicted", "")

    salary_display = ""
    if salary_min and salary_max:
        tag = " (estimated)" if str(salary_predicted) == "1" else ""
        salary_display = f"£{salary_min:,.0f} - £{salary_max:,.0f}{tag}"

    return {
        "Job Title": job.get("title", ""),
        "Company": company,
        "Location": location,
        "Category": category,
        "Salary": salary_display,
        "Contract Type": job.get("contract_type", ""),
        "Contract Time": job.get("contract_time", ""),
        "Posted": job.get("created", ""),
        "Match Type": match_type,
        "URL": job.get("redirect_url", ""),
        "Description": (job.get("description", "") or "")[:300],
    }


# ── 4. Save to CSV ──────────────────────────────────────────────────────────
def save_csv(rows: list[dict], filepath: str):
    """Write matched jobs to a CSV file."""
    if not rows:
        log.warning("No matching jobs found — no CSV created.")
        return

    fieldnames = [
        "Job Title", "Company", "Location", "Category", "Salary",
        "Contract Type", "Contract Time", "Posted", "Match Type",
        "URL", "Description",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Saved %d jobs to %s", len(rows), filepath)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Adzuna UK Sponsorship Jobs Scraper")
    log.info("=" * 60)

    sponsors = download_sponsor_list()
    jobs = fetch_adzuna_jobs()
    matched = match_jobs_to_sponsors(jobs, sponsors)
    save_csv(matched, OUTPUT_CSV)

    log.info("=" * 60)
    if matched:
        log.info("Done! %d entry-level jobs from sponsored companies.", len(matched))
        log.info("Output: %s", OUTPUT_CSV)
    else:
        log.info("No sponsored entry-level jobs found in the last 24 hours.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
