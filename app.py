"""
Job Scraper REST API

Endpoints:
    POST /api/match-jobs          Upload CV → returns task_id immediately (<500ms)
    GET  /api/match-jobs/{id}     Poll for task status / results
    GET  /api/scrape              Scrape jobs only (no CV matching)
    GET  /api/health              Health check

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import time
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from config import log, SUPABASE_URL
from sponsors import download_sponsor_list
from adzuna import fetch_adzuna_jobs
from reed import fetch_reed_jobs
from matcher import match_jobs_to_sponsors
from cv_reader import read_cv
from job_recommender import extract_cv_profile, recommend_jobs_from_list

# ── In-memory cache for scraped+matched jobs (avoids re-scraping every request)
_jobs_cache: list[dict] = []
_jobs_cache_time: float = 0
JOBS_CACHE_TTL = 3600  # 1 hour

# ── In-memory task store for async processing ────────────────────────────────
_tasks: dict[str, dict] = {}
TASK_TTL = 3600  # clean up tasks older than 1 hour

app = FastAPI(
    title="Job Scraper API",
    description="Upload a CV and get matched UK sponsorship jobs from Adzuna + Reed",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-warm jobs cache in background ───────────────────────────────

@app.on_event("startup")
def startup_warmup():
    """Pre-warm the jobs cache so the first request doesn't wait for scraping."""
    def _warmup():
        try:
            log.info("Startup: pre-warming jobs cache ...")
            _scrape_and_match(force=True)
            log.info("Startup: jobs cache ready (%d jobs)", len(_jobs_cache))
        except Exception as e:
            log.warning("Startup warmup failed (will retry on first request): %s", e)
    Thread(target=_warmup, daemon=True).start()


# ── Helpers ──────────────────────────────────────────────────────────────────

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

    # Fetch all three data sources in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_sponsors = pool.submit(download_sponsor_list)
        f_adzuna = pool.submit(fetch_adzuna_jobs)
        f_reed = pool.submit(fetch_reed_jobs)

    sponsors = f_sponsors.result()
    adzuna_jobs = f_adzuna.result()
    reed_jobs = f_reed.result()
    matched = match_jobs_to_sponsors(adzuna_jobs, reed_jobs, sponsors)
    _save_jobs_to_supabase(matched)

    _jobs_cache = matched
    _jobs_cache_time = time.time()
    return matched


def _cleanup_old_tasks():
    """Remove tasks older than TASK_TTL to prevent memory leaks."""
    now = time.time()
    expired = [tid for tid, t in _tasks.items() if now - t.get("created_at", now) > TASK_TTL]
    for tid in expired:
        del _tasks[tid]
    if expired:
        log.info("Cleaned up %d expired tasks", len(expired))


def _process_match(task_id: str, cv_text: str, cv_filename: str, top_n: int):
    """Background worker: scrape, score, and store results for a task."""
    try:
        task = _tasks[task_id]

        def update_progress(msg: str):
            task["progress"] = msg

        # 1. Scrape jobs AND extract CV profile in parallel
        #    (they're independent — no reason to wait)
        update_progress("Scraping jobs & analysing CV in parallel...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_jobs = pool.submit(_scrape_and_match)
            f_profile = pool.submit(extract_cv_profile, cv_text)

        matched = f_jobs.result()
        profile = f_profile.result()

        if not matched:
            task["status"] = "completed"
            task["result"] = {
                "total_matched_jobs": 0,
                "recommendations": [],
                "message": "No sponsored jobs found in the last 24 hours.",
            }
            return

        # 2. Score matched jobs against CV using OpenAI (parallel batches)
        update_progress("Scoring jobs with AI...")
        recommendations = recommend_jobs_from_list(
            cv_text, matched, top_n=top_n, progress_cb=update_progress,
            profile=profile,
        )

        # 3. Save to Supabase (non-blocking from user's perspective)
        _save_recs_to_supabase(recommendations, cv_filename)

        # 4. Store final result
        task["status"] = "completed"
        task["progress"] = "Done"
        task["result"] = {
            "total_matched_jobs": len(matched),
            "sources": {
                "adzuna": sum(1 for j in matched if j.get("Source") == "adzuna"),
                "reed": sum(1 for j in matched if j.get("Source") == "reed"),
            },
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }
        log.info("Task %s completed: %d recommendations", task_id, len(recommendations))

    except Exception as e:
        log.exception("Task %s failed", task_id)
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


# ── Endpoints ────────────────────────────────────────────────────────────────

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
    background_tasks: BackgroundTasks,
    cv: UploadFile = File(..., description="CV file (PDF or DOCX)"),
    top_n: int = Query(20, ge=1, le=100, description="Number of top recommendations"),
):
    """Upload a CV and start async matching. Returns a task_id immediately (<500ms).
    Poll GET /api/match-jobs/{task_id} for progress and results."""
    # Validate file type
    ext = os.path.splitext(cv.filename or "")[1].lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Upload a .pdf or .docx file.",
        )

    # Save uploaded file and extract text immediately (fast — no API calls)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(cv.file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        cv_text = read_cv(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract CV text: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Clean up old tasks periodically
    _cleanup_old_tasks()

    # Create task and kick off background processing
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "progress": "Starting...",
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(_process_match, task_id, cv_text, cv.filename or "unknown", top_n)

    return {"task_id": task_id}


@app.get("/api/match-jobs/{task_id}")
def get_match_result(task_id: str):
    """Poll for task status and results."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    response = {
        "status": task["status"],
        "progress": task["progress"],
    }

    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
