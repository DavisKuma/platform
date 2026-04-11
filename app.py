"""
Job Scraper REST API

Endpoints:
    POST /api/match-jobs    Upload CV → get matched + recommended jobs
    GET  /api/scrape        Scrape jobs only (no CV matching)
    GET  /api/health        Health check

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import time
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import log, SUPABASE_URL
from sponsors import download_sponsor_list
from adzuna import fetch_adzuna_jobs
from reed import fetch_reed_jobs
from matcher import match_jobs_to_sponsors
from cv_reader import read_cv
from job_recommender import recommend_jobs_from_list

# ── In-memory cache for scraped+matched jobs (avoids re-scraping every request)
_jobs_cache: list[dict] = []
_jobs_cache_time: float = 0
JOBS_CACHE_TTL = 3600  # 1 hour

app = FastAPI(
    title="Job Scraper API",
    description="Upload a CV and get matched UK sponsorship jobs from Adzuna + Reed",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_jobs_to_supabase(matched: list[dict]):
    if not SUPABASE_URL:
        return
    try:
        from database import save_jobs_to_db
        save_jobs_to_db(matched)
    except Exception as e:
        log.warning("Failed to save jobs to Supabase: %s", e)


def _save_recs_to_supabase(recommendations: list[dict], cv_filename: str):
    if not SUPABASE_URL:
        return
    try:
        from database import save_recommendations_to_db
        save_recommendations_to_db(recommendations, cv_filename)
    except Exception as e:
        log.warning("Failed to save recommendations to Supabase: %s", e)


def _scrape_and_match(force: bool = False) -> list[dict]:
    """Fetch jobs from both APIs and match against sponsors.
    Results are cached in memory for JOBS_CACHE_TTL seconds."""
    global _jobs_cache, _jobs_cache_time

    if not force and _jobs_cache and (time.time() - _jobs_cache_time) < JOBS_CACHE_TTL:
        log.info("Using cached jobs (%d jobs, %.0fs old)", len(_jobs_cache), time.time() - _jobs_cache_time)
        return _jobs_cache

    sponsors = download_sponsor_list()
    adzuna_jobs = fetch_adzuna_jobs()
    reed_jobs = fetch_reed_jobs()
    matched = match_jobs_to_sponsors(adzuna_jobs, reed_jobs, sponsors)
    _save_jobs_to_supabase(matched)

    _jobs_cache = matched
    _jobs_cache_time = time.time()
    return matched


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/scrape")
def scrape_jobs():
    """Scrape jobs from Adzuna + Reed, match against UK sponsors. No CV needed."""
    try:
        matched = _scrape_and_match()
        return {
            "total_jobs": len(matched),
            "sources": {
                "adzuna": sum(1 for j in matched if j.get("Source") == "adzuna"),
                "reed": sum(1 for j in matched if j.get("Source") == "reed"),
            },
            "jobs": matched,
        }
    except Exception as e:
        log.exception("Scrape failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match-jobs")
def match_jobs(
    cv: UploadFile = File(..., description="CV file (PDF or DOCX)"),
    top_n: int = Query(20, ge=1, le=100, description="Number of top recommendations"),
):
    """Upload a CV and get matched + AI-recommended jobs from Adzuna + Reed."""
    # Validate file type
    ext = os.path.splitext(cv.filename or "")[1].lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Upload a .pdf or .docx file.",
        )

    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(cv.file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        # 1. Read CV text
        cv_text = read_cv(tmp_path)

        # 2. Scrape & match jobs from both APIs
        matched = _scrape_and_match()

        if not matched:
            return {
                "total_matched_jobs": 0,
                "recommendations": [],
                "message": "No sponsored jobs found in the last 24 hours.",
            }

        # 3. Score matched jobs against CV using OpenAI
        recommendations = recommend_jobs_from_list(cv_text, matched, top_n=top_n)

        # 4. Save to Supabase
        _save_recs_to_supabase(recommendations, cv.filename or "unknown")

        return {
            "total_matched_jobs": len(matched),
            "sources": {
                "adzuna": sum(1 for j in matched if j.get("Source") == "adzuna"),
                "reed": sum(1 for j in matched if j.get("Source") == "reed"),
            },
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }

    except Exception as e:
        log.exception("Match-jobs failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
