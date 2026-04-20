import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI

from config import log, OPENAI_API_KEY, OPENAI_MODEL
from cv_reader import read_cv

# ── CV profile cache (hash of CV text → extracted profile) ──────────────────
_profile_cache: dict[str, dict] = {}


def _sanitize(text: str) -> str:
    """Strip control characters and null bytes that break JSON serialization."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ch >= " " and ord(ch) != 0x7F))


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to your .env file.\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def extract_cv_profile(cv_text: str) -> dict:
    """Use OpenAI to extract a structured profile from CV text.
    Results are cached by content hash to avoid redundant API calls."""
    cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()[:16]
    if cv_hash in _profile_cache:
        log.info("CV profile cache hit (hash=%s)", cv_hash)
        return _profile_cache[cv_hash]

    client = _get_client()

    log.info("Extracting CV profile via OpenAI (hash=%s) ...", cv_hash)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CV analysis assistant. Extract a structured profile "
                    "from the CV text. Return JSON with these fields:\n"
                    '- "skills": list of technical and soft skills\n'
                    '- "job_titles": list of job titles the candidate has held or is suited for\n'
                    '- "industries": list of industries the candidate has experience in\n'
                    '- "experience_level": one of "entry", "mid", "senior"\n'
                    '- "education": list of degrees/qualifications\n'
                    '- "summary": 2-3 sentence summary of the candidate'
                ),
            },
            {"role": "user", "content": _sanitize(cv_text[:8000])},
        ],
    )

    profile = json.loads(resp.choices[0].message.content)
    log.info(
        "CV profile: %d skills, %d job titles, level=%s",
        len(profile.get("skills", [])),
        len(profile.get("job_titles", [])),
        profile.get("experience_level", "unknown"),
    )
    _profile_cache[cv_hash] = profile
    return profile


def score_jobs_batch(profile: dict, jobs: list[dict]) -> list[dict]:
    """Send a batch of jobs to OpenAI for relevance scoring against the CV profile."""
    client = _get_client()

    jobs_summary = []
    for i, job in enumerate(jobs):
        jobs_summary.append(
            f"[{i}] {_sanitize(job['Job Title'])} at {_sanitize(job['Company'])} "
            f"| {_sanitize(job['Location'])} | {_sanitize(job.get('Category', ''))} "
            f"| {_sanitize(str(job.get('Salary', 'N/A')))}"
        )

    jobs_text = "\n".join(jobs_summary)
    profile_text = json.dumps(profile, indent=2)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict job matching assistant. Given a candidate profile and "
                    "a list of jobs, score each job's relevance from 0-100.\n\n"
                    "SCORING RULES (be strict):\n"
                    "- 80-100: Direct skills match AND relevant job title (e.g. data scientist CV → data analyst role)\n"
                    "- 60-79: Strong skills overlap with related field (e.g. marketing CV → content role)\n"
                    "- 40-59: Some transferable skills but different field\n"
                    "- 20-39: Minimal relevance, mostly unrelated\n"
                    "- 0-19: Completely unrelated\n\n"
                    "IMPORTANT: A job in an unrelated field should score BELOW 40 even if it's entry-level. "
                    "Do NOT inflate scores just because a job exists in the same broad industry. "
                    "Focus on whether the candidate's specific SKILLS and EXPERIENCE match the job requirements.\n\n"
                    "Return JSON: {\"scores\": [{\"index\": 0, \"score\": 85, "
                    "\"reason\": \"Strong match because...\"}]}\n"
                    "Include ALL jobs in the output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Candidate Profile\n{profile_text}\n\n"
                    f"## Jobs\n{jobs_text}"
                ),
            },
        ],
    )

    result = json.loads(resp.choices[0].message.content)
    scores = {s["index"]: s for s in result.get("scores", [])}

    scored_jobs = []
    for i, job in enumerate(jobs):
        score_data = scores.get(i, {"score": 0, "reason": "Not scored"})
        scored_job = dict(job)
        scored_job["Relevance Score"] = score_data["score"]
        scored_job["Why"] = score_data.get("reason", "")
        scored_jobs.append(scored_job)

    return scored_jobs


def _keyword_prefilter(profile: dict, jobs: list[dict], max_jobs: int = 75) -> list[dict]:
    """Fast keyword pre-filter: keep only jobs whose title/category overlap with CV skills/titles.
    This reduces the number of jobs sent to OpenAI from 300+ to ~75."""
    keywords = set()
    for skill in profile.get("skills", []):
        for word in skill.lower().split():
            if len(word) > 2:
                keywords.add(word)
    for title in profile.get("job_titles", []):
        for word in title.lower().split():
            if len(word) > 2:
                keywords.add(word)

    scored = []
    for job in jobs:
        text = f"{job.get('Job Title', '')} {job.get('Category', '')}".lower()
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored.append((hits, job))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [job for _, job in scored[:max_jobs]]
    log.info("Pre-filter: %d/%d jobs matched CV keywords (keeping top %d)", len(scored), len(jobs), len(filtered))
    return filtered if filtered else jobs[:max_jobs]


def recommend_jobs_from_list(
    cv_text: str, all_jobs: list[dict], top_n: int = 20, progress_cb=None,
    profile: dict | None = None,
) -> list[dict]:
    """Score a list of jobs against CV text. Used by the REST API.
    If profile is already extracted (e.g. done in parallel), pass it to skip re-extraction."""
    if profile is None:
        if progress_cb:
            progress_cb("Extracting CV profile...")
        profile = extract_cv_profile(cv_text)
    if progress_cb:
        progress_cb("Filtering jobs by keywords...")
    filtered = _keyword_prefilter(profile, all_jobs)
    if progress_cb:
        progress_cb("Scoring jobs with AI...")
    return _score_and_rank(profile, filtered, top_n, progress_cb=progress_cb)


def recommend_jobs(csv_path: str, cv_path: str, top_n: int = 20) -> list[dict]:
    """Full pipeline: read CV, load jobs CSV, score with OpenAI, return top N."""
    # Read CV
    cv_text = read_cv(cv_path)
    profile = extract_cv_profile(cv_text)

    # Load jobs from CSV
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_jobs = list(reader)

    log.info("Loaded %d jobs from %s", len(all_jobs), csv_path)

    return _score_and_rank(profile, all_jobs, top_n)


def _score_and_rank(
    profile: dict,
    all_jobs: list[dict],
    top_n: int,
    progress_cb=None,
) -> list[dict]:
    """Score jobs against a profile and return top N.
    Batches are scored in parallel via ThreadPoolExecutor."""
    batch_size = 50
    batches = [all_jobs[i : i + batch_size] for i in range(0, len(all_jobs), batch_size)]
    total_batches = len(batches)
    all_scored = []

    log.info("Scoring %d jobs in %d parallel batches ...", len(all_jobs), total_batches)

    with ThreadPoolExecutor(max_workers=min(4, total_batches)) as pool:
        futures = {
            pool.submit(score_jobs_batch, profile, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            scored = future.result()
            all_scored.extend(scored)
            log.info("  Batch %d/%d done (%d scored)", idx + 1, total_batches, len(scored))
            if progress_cb:
                progress_cb(f"Scored {len(all_scored)}/{len(all_jobs)} jobs")

    MIN_SCORE = 50
    all_scored = [j for j in all_scored if j.get("Relevance Score", 0) >= MIN_SCORE]
    all_scored.sort(key=lambda x: x.get("Relevance Score", 0), reverse=True)
    top = all_scored[:top_n]

    log.info("Top %d recommendations ready (scores: %d-%d)",
             len(top),
             top[-1].get("Relevance Score", 0) if top else 0,
             top[0].get("Relevance Score", 0) if top else 0)

    return top


def save_recommendations(recommendations: list[dict], filepath: str):
    """Save recommendations to a CSV file."""
    if not recommendations:
        log.warning("No recommendations to save.")
        return

    fieldnames = [
        "Relevance Score", "Why", "Job Title", "Company", "Location",
        "Category", "Salary", "Contract Type", "Contract Time",
        "Posted", "URL",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(recommendations)

    log.info("Saved %d recommendations to %s", len(recommendations), filepath)
