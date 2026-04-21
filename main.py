"""
Adzuna UK Entry-Level Job Scraper + CV Matcher

Usage:
    python main.py                          # Scrape jobs only
    python main.py --cv path/to/resume.pdf  # Scrape + recommend jobs for your CV
    python main.py --cv resume.pdf --csv existing_jobs.csv  # Skip scraping, match CV against existing CSV
"""

import argparse
import glob
import os
from datetime import datetime

from config import log, SUPABASE_URL
from sponsors import download_sponsor_list
from adzuna import fetch_adzuna_jobs
from reed import fetch_reed_jobs
from matcher import match_jobs_to_sponsors, save_csv
from job_recommender import recommend_jobs, save_recommendations


def _save_jobs_to_supabase(matched: list[dict]):
    """Save jobs to Supabase if configured. Non-fatal on failure."""
    if not SUPABASE_URL:
        log.info("Supabase not configured — skipping DB save (CSV still saved)")
        return

    try:
        from database import save_jobs_to_db
        save_jobs_to_db(matched)
    except Exception as e:
        log.warning("Failed to save jobs to Supabase: %s", e)


def _save_recs_to_supabase(recommendations: list[dict], cv_path: str):
    """Save recommendations to Supabase if configured. Non-fatal on failure."""
    if not SUPABASE_URL:
        return

    try:
        from database import save_recommendations_to_db
        save_recommendations_to_db(recommendations, cv_path)
    except Exception as e:
        log.warning("Failed to save recommendations to Supabase: %s", e)


def _cleanup_old_csvs(prefix: str, keep: str):
    """Delete old CSVs matching prefix, keeping only the new file."""
    for old in glob.glob(f"{prefix}_*.csv"):
        if old != keep:
            try:
                os.remove(old)
                log.info("Deleted old CSV: %s", old)
            except OSError:
                pass


def run_scraper() -> tuple[str, list[dict]]:
    """Run the full scraping pipeline. Returns (csv_path, matched_jobs)."""
    output_csv = f"adzuna_sponsored_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    sponsors = download_sponsor_list()
    adzuna_jobs = fetch_adzuna_jobs()
    reed_jobs = fetch_reed_jobs()
    matched = match_jobs_to_sponsors(adzuna_jobs, reed_jobs, sponsors)

    # Save to CSV (delete old ones first)
    _cleanup_old_csvs("adzuna_sponsored_jobs", output_csv)
    save_csv(matched, output_csv)

    # Save to Supabase
    _save_jobs_to_supabase(matched)

    if matched:
        log.info("Scraper done! %d jobs saved to %s", len(matched), output_csv)
    else:
        log.info("No sponsored entry-level jobs found in the last 24 hours.")

    return output_csv, matched


def run_recommender(csv_path: str, cv_path: str):
    """Run the CV recommendation pipeline."""
    rec_csv = f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    log.info("=" * 60)
    log.info("CV Job Recommender")
    log.info("=" * 60)

    recommendations = recommend_jobs(csv_path, cv_path, top_n=20)

    # Save to CSV (delete old ones first)
    _cleanup_old_csvs("recommendations", rec_csv)
    save_recommendations(recommendations, rec_csv)

    # Save to Supabase
    _save_recs_to_supabase(recommendations, cv_path)

    if recommendations:
        log.info("=" * 60)
        log.info("Top %d job recommendations:", len(recommendations))
        for i, rec in enumerate(recommendations, 1):
            log.info(
                "  %2d. [%d%%] %s at %s — %s",
                i,
                rec.get("Relevance Score", 0),
                rec.get("Job Title", ""),
                rec.get("Company", ""),
                rec.get("Why", "")[:80],
            )
        log.info("Full results: %s", rec_csv)
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Adzuna UK Sponsorship Jobs Scraper + CV Matcher"
    )
    parser.add_argument(
        "--cv",
        help="Path to your CV (PDF or DOCX) for job recommendations",
    )
    parser.add_argument(
        "--csv",
        help="Path to an existing jobs CSV (skip scraping, only run CV matcher)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Adzuna + Reed UK Sponsorship Jobs Scraper")
    log.info("=" * 60)

    if args.csv and args.cv:
        # Skip scraping, just run recommendations on existing CSV
        run_recommender(args.csv, args.cv)
    elif args.cv:
        # Scrape first, then recommend
        csv_path, _ = run_scraper()
        run_recommender(csv_path, args.cv)
    else:
        # Scrape only
        run_scraper()


if __name__ == "__main__":
    main()
