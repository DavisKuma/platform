import csv

from rapidfuzz import fuzz, process

from config import log, FUZZY_MATCH_THRESHOLD
from sponsors import normalise_name
from adzuna import build_row as build_adzuna_row
from reed import build_reed_row


def match_adzuna_to_sponsors(
    jobs: list[dict], sponsors: set[str]
) -> list[dict]:
    """Cross-reference Adzuna jobs with sponsorship license holders."""
    matched = []
    sponsors_list = list(sponsors)

    log.info("Cross-referencing %d Adzuna jobs against %d sponsors ...", len(jobs), len(sponsors))

    for i, job in enumerate(jobs):
        company_raw = (job.get("company") or {}).get("display_name", "")
        if not company_raw:
            continue

        company_norm = normalise_name(company_raw)

        if company_norm in sponsors:
            matched.append(build_adzuna_row(job, company_raw, "exact"))
            continue

        result = process.extractOne(
            company_norm, sponsors_list,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if result:
            _, score, _ = result
            matched.append(build_adzuna_row(job, company_raw, f"fuzzy ({score:.0f}%)"))

        if (i + 1) % 500 == 0:
            log.info("  Processed %d/%d Adzuna jobs (%d matched)", i + 1, len(jobs), len(matched))

    log.info("Adzuna: matched %d / %d jobs", len(matched), len(jobs))
    return matched


def match_reed_to_sponsors(
    jobs: list[dict], sponsors: set[str]
) -> list[dict]:
    """Cross-reference Reed jobs with sponsorship license holders."""
    matched = []
    sponsors_list = list(sponsors)

    log.info("Cross-referencing %d Reed jobs against %d sponsors ...", len(jobs), len(sponsors))

    for i, job in enumerate(jobs):
        company_raw = job.get("employerName", "")
        if not company_raw:
            continue

        company_norm = normalise_name(company_raw)

        if company_norm in sponsors:
            matched.append(build_reed_row(job, company_raw, "exact"))
            continue

        result = process.extractOne(
            company_norm, sponsors_list,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if result:
            _, score, _ = result
            matched.append(build_reed_row(job, company_raw, f"fuzzy ({score:.0f}%)"))

        if (i + 1) % 500 == 0:
            log.info("  Processed %d/%d Reed jobs (%d matched)", i + 1, len(jobs), len(matched))

    log.info("Reed: matched %d / %d jobs", len(matched), len(jobs))
    return matched


def match_jobs_to_sponsors(
    adzuna_jobs: list[dict], reed_jobs: list[dict], sponsors: set[str]
) -> list[dict]:
    """Match both Adzuna and Reed jobs against sponsors, return combined list."""
    adzuna_matched = match_adzuna_to_sponsors(adzuna_jobs, sponsors)
    reed_matched = match_reed_to_sponsors(reed_jobs, sponsors)

    combined = adzuna_matched + reed_matched
    log.info("Total matched: %d (Adzuna: %d, Reed: %d)",
             len(combined), len(adzuna_matched), len(reed_matched))
    return combined


def save_csv(rows: list[dict], filepath: str):
    """Write matched jobs to a CSV file."""
    if not rows:
        log.warning("No matching jobs found — no CSV created.")
        return

    fieldnames = [
        "Job Title", "Company", "Location", "Category", "Salary",
        "Contract Type", "Contract Time", "Posted", "Match Type",
        "URL", "Description", "Source",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("Saved %d jobs to %s", len(rows), filepath)
