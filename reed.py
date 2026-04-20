import time
from datetime import datetime, timedelta

import requests

from config import log, REED_API_KEY, ENTRY_LEVEL_KEYWORDS, SALARY_MAX


REED_SEARCH_URL = "https://www.reed.co.uk/api/1.0/search"
REED_RESULTS_PER_PAGE = 100


def fetch_reed_jobs() -> list[dict]:
    """Fetch entry-level UK jobs from Reed, filtered to last 24 hours."""
    if not REED_API_KEY:
        log.warning("REED_API_KEY not set — skipping Reed")
        return []

    keywords = ENTRY_LEVEL_KEYWORDS.replace(" ", " OR ")

    all_jobs = []
    skip = 0

    log.info("Querying Reed API for entry-level UK jobs ...")

    while True:
        params = {
            "keywords": keywords,
            "maximumSalary": SALARY_MAX,
            "resultsToTake": REED_RESULTS_PER_PAGE,
            "resultsToSkip": skip,
            "graduate": "true",
        }

        resp = requests.get(
            REED_SEARCH_URL,
            params=params,
            auth=(REED_API_KEY, ""),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        all_jobs.extend(results)
        log.info("  Reed: fetched %d jobs so far", len(all_jobs))

        # Reed caps at 100 per request; stop after 300 to keep it fast
        if len(results) < REED_RESULTS_PER_PAGE or len(all_jobs) >= 300:
            break

        skip += REED_RESULTS_PER_PAGE
        time.sleep(0.1)

    # Filter to last 24 hours client-side (compare dates only, not times)
    cutoff_date = (datetime.now() - timedelta(hours=24)).date()
    recent = []
    for job in all_jobs:
        date_str = job.get("date", "")
        if not date_str:
            recent.append(job)  # keep if no date field
            continue
        try:
            posted = datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            try:
                posted = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
            except ValueError:
                recent.append(job)  # keep if date can't be parsed
                continue
        if posted >= cutoff_date:
            recent.append(job)

    log.info("Reed: %d jobs total, %d from last 24h", len(all_jobs), len(recent))
    return recent


def build_reed_row(job: dict, company: str, match_type: str) -> dict:
    """Convert a Reed API job result to our standard row format."""
    salary_min = job.get("minimumSalary")
    salary_max = job.get("maximumSalary")

    salary_display = ""
    if salary_min and salary_max:
        salary_display = f"\u00a3{salary_min:,.0f} - \u00a3{salary_max:,.0f}"
    elif salary_min:
        salary_display = f"\u00a3{salary_min:,.0f}+"
    elif salary_max:
        salary_display = f"Up to \u00a3{salary_max:,.0f}"

    return {
        "Job Title": job.get("jobTitle", ""),
        "Company": company,
        "Location": job.get("locationName", ""),
        "Category": "",
        "Salary": salary_display,
        "Contract Type": "permanent" if job.get("permanent") else "contract" if job.get("contract") else "",
        "Contract Time": "full_time" if job.get("fullTime") else "part_time" if job.get("partTime") else "",
        "Posted": job.get("date", ""),
        "Match Type": match_type,
        "URL": job.get("jobUrl", ""),
        "Description": (job.get("jobDescription", "") or "")[:300],
        "Source": "reed",
    }
