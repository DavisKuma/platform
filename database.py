import os

from supabase import create_client, Client

from config import log, SUPABASE_URL, SUPABASE_KEY


_client: Client | None = None


def get_supabase() -> Client:
    """Create or return the Supabase client singleton."""
    global _client
    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY "
            "in your .env file."
        )

    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _job_row_to_db(row: dict) -> dict:
    """Convert a matched job dict (CSV-style keys) to DB column names."""
    return {
        "job_title": row.get("Job Title", ""),
        "company": row.get("Company", ""),
        "location": row.get("Location", ""),
        "category": row.get("Category", ""),
        "salary": row.get("Salary", ""),
        "contract_type": row.get("Contract Type", ""),
        "contract_time": row.get("Contract Time", ""),
        "posted": row.get("Posted", None) or None,
        "match_type": row.get("Match Type", ""),
        "url": row.get("URL", ""),
        "description": row.get("Description", ""),
    }


def save_jobs_to_db(jobs: list[dict]) -> int:
    """Upsert matched jobs into the `jobs` table. Returns count saved."""
    if not jobs:
        return 0

    client = get_supabase()
    db_rows = [_job_row_to_db(j) for j in jobs]

    # Filter out rows without a URL (needed for upsert on unique constraint)
    db_rows = [r for r in db_rows if r["url"]]

    # Upsert in batches of 100
    saved = 0
    batch_size = 100
    for i in range(0, len(db_rows), batch_size):
        batch = db_rows[i : i + batch_size]
        client.table("jobs").upsert(
            batch, on_conflict="url"
        ).execute()
        saved += len(batch)

    log.info("Saved %d jobs to Supabase", saved)
    return saved


def _rec_row_to_db(row: dict, cv_filename: str) -> dict:
    """Convert a recommendation dict to DB column names."""
    return {
        "cv_filename": cv_filename,
        "relevance_score": row.get("Relevance Score", 0),
        "reason": row.get("Why", ""),
        "job_title": row.get("Job Title", ""),
        "company": row.get("Company", ""),
        "location": row.get("Location", ""),
        "category": row.get("Category", ""),
        "salary": row.get("Salary", ""),
        "contract_type": row.get("Contract Type", ""),
        "contract_time": row.get("Contract Time", ""),
        "posted": row.get("Posted", ""),
        "url": row.get("URL", ""),
    }


def save_recommendations_to_db(
    recommendations: list[dict], cv_path: str
) -> int:
    """Insert recommendations into the `recommendations` table. Returns count saved."""
    if not recommendations:
        return 0

    client = get_supabase()
    cv_filename = os.path.basename(cv_path)
    db_rows = [_rec_row_to_db(r, cv_filename) for r in recommendations]

    client.table("recommendations").insert(db_rows).execute()

    log.info("Saved %d recommendations to Supabase", len(db_rows))
    return len(db_rows)
