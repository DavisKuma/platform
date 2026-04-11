import math
import time

import requests

from config import (
    log,
    ADZUNA_APP_ID,
    ADZUNA_APP_KEY,
    ADZUNA_BASE_URL,
    RESULTS_PER_PAGE,
    ENTRY_LEVEL_KEYWORDS,
    SALARY_MAX,
)


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

    log.info("Querying Adzuna API for entry-level UK jobs (last 24h) ...")
    resp = requests.get(f"{ADZUNA_BASE_URL}/{page}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    total = data.get("count", 0)
    total_pages = min(math.ceil(total / RESULTS_PER_PAGE), 20)
    log.info("Found %d total jobs across %d pages", total, total_pages)

    all_jobs.extend(data.get("results", []))

    for page in range(2, total_pages + 1):
        time.sleep(0.3)
        resp = requests.get(f"{ADZUNA_BASE_URL}/{page}", params=params, timeout=30)
        resp.raise_for_status()
        all_jobs.extend(resp.json().get("results", []))
        log.info("  Fetched page %d/%d (%d jobs so far)", page, total_pages, len(all_jobs))

    log.info("Retrieved %d jobs from Adzuna", len(all_jobs))
    return all_jobs


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
        salary_display = f"\u00a3{salary_min:,.0f} - \u00a3{salary_max:,.0f}{tag}"

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
        "Source": "adzuna",
    }
