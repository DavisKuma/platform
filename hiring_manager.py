"""
Hiring Manager Finder

Uses OpenAI to identify likely hiring managers / decision-makers at a company
for a given role, then resolves their actual LinkedIn profile URLs via search.

Approach:
  1. Ask GPT-4o-mini to identify the most likely people (by title/role)
     who would be the hiring decision-maker at the company for the given job.
  2. Search DuckDuckGo for each person to find their actual LinkedIn profile URL.
  3. Cache results by company + job_title to avoid repeat API calls.

Results are cached in memory for 2 hours.
"""

import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from ddgs import DDGS

from config import log, OPENAI_API_KEY, OPENAI_MODEL


# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
CACHE_TTL = 7200  # 2 hours


def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def _cache_key(company: str, job_title: str) -> str:
    raw = f"{company.lower().strip()}|{job_title.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _sanitize(text: str) -> str:
    """Strip control characters that break JSON."""
    return "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or (ch >= " " and ord(ch) != 0x7F)
    )


def _find_linkedin_url(query: str) -> str | None:
    """Search DuckDuckGo for a LinkedIn profile URL matching the query.
    Returns the first linkedin.com/in/ URL found, or None."""
    try:
        results = DDGS().text(f"{query} site:linkedin.com/in", max_results=5)
        for r in results:
            url = r.get("href", "")
            if "linkedin.com/in/" in url:
                return url
    except Exception as e:
        log.warning("LinkedIn search failed for '%s': %s", query, e)
    return None


def _resolve_linkedin_urls(people: list[dict], company: str) -> None:
    """Resolve actual LinkedIn profile URLs for each person in parallel.
    Updates each person dict in-place with 'linkedin_url'."""

    def _resolve_one(person: dict) -> None:
        name = person.get("full_name", "")
        title = person.get("title", "")
        is_verified = person.get("is_verified", False)

        # Verified = search by real name, Unverified = search by title at company
        if is_verified and name:
            query = f"{name} {company}"
        else:
            query = f"{title} {company}"

        url = _find_linkedin_url(query)

        if url:
            person["linkedin_url"] = url
            # Extract the real name from LinkedIn title if we searched by title
            log.info("  Found LinkedIn: %s → %s", query, url)
        else:
            # Fallback: Google search link
            from urllib.parse import quote_plus
            fallback_query = f"{title} {company} LinkedIn" if not is_verified else f"{name} {company} LinkedIn"
            person["linkedin_url"] = f"https://www.google.com/search?q={quote_plus(fallback_query)}"
            log.info("  No LinkedIn found for '%s', using Google fallback", query)

    # Resolve all in parallel (max 3 concurrent searches)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.map(_resolve_one, people)


# ── AI-Powered Hiring Manager Identification ─────────────────────────────────

def _identify_hiring_managers(company: str, job_title: str) -> list[dict]:
    """Use GPT-4o-mini to identify likely hiring managers at a company."""
    client = _get_openai_client()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert recruitment researcher. Given a company name and a job title "
                    "being hired for, identify the 3-5 most likely people who would be involved in "
                    "the hiring decision for that role.\n\n"
                    "For each person, provide:\n"
                    "- full_name: A realistic full name (first + last name)\n"
                    "- title: Their likely job title at the company\n"
                    "- ai_score: 0-100 probability of being the decision-maker for this hire\n"
                    "- reason: Brief explanation of why they would be involved\n\n"
                    "RULES:\n"
                    "- If you know actual people at the company (from your training data), use their real names and titles\n"
                    "- If you don't know specific people, identify the ROLE TYPES that would make the hiring decision "
                    "and provide realistic placeholder names with those titles\n"
                    "- Always include the direct department head (highest score)\n"
                    "- Include HR/Talent Acquisition (medium score)\n"
                    "- Include C-level if the company is small (<200 employees)\n"
                    "- Match the department: engineering roles -> engineering leadership, marketing roles -> marketing leadership\n\n"
                    "SCORING:\n"
                    "- 85-100: Direct hiring manager (e.g. Head of Engineering for a Software Engineer role)\n"
                    "- 70-84: Strong influence (e.g. CTO, VP of the department)\n"
                    "- 50-69: Involved in process (e.g. HR Manager, Talent Acquisition Lead)\n"
                    "- 30-49: May be consulted\n\n"
                    "Return JSON: {\"people\": [{\"full_name\": \"...\", \"title\": \"...\", "
                    "\"ai_score\": 85, \"reason\": \"...\", \"is_verified\": true/false}]}\n"
                    "Set is_verified=true ONLY if you are confident this is a real person at the company "
                    "from your training data. Set is_verified=false if it's an estimated role."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Company: {_sanitize(company)}\n"
                    f"Job Title Being Hired: {_sanitize(job_title)}\n\n"
                    f"Who are the most likely hiring decision-makers for this role at this company?"
                ),
            },
        ],
    )

    result = json.loads(resp.choices[0].message.content)
    people = result.get("people", [])

    # Resolve actual LinkedIn profile URLs via DuckDuckGo search
    log.info("Resolving LinkedIn URLs for %d people at '%s'...", len(people), company)
    _resolve_linkedin_urls(people, company)

    # Sort by score descending
    people.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

    log.info(
        "AI identified %d potential hiring managers at '%s' for '%s'",
        len(people), company, job_title,
    )
    return people


# ── Public API ───────────────────────────────────────────────────────────────

def find_hiring_managers(company: str, job_title: str, max_results: int = 5) -> list[dict]:
    """Find likely hiring managers at a company for a given role.

    Returns a list of dicts with:
        full_name, title, linkedin_url, ai_score, reason, is_verified
    Results are cached for CACHE_TTL seconds.
    """
    # Check cache
    key = _cache_key(company, job_title)
    cached = _cache.get(key)
    if cached and (time.time() - cached["time"]) < CACHE_TTL:
        log.info(
            "Hiring manager cache hit for '%s' + '%s' (%d results)",
            company, job_title, len(cached["data"]),
        )
        return cached["data"]

    log.info("Finding hiring managers at '%s' for role '%s'", company, job_title)

    people = _identify_hiring_managers(company, job_title)

    # Keep top results
    top = people[:max_results]

    # Cache results
    _cache[key] = {"data": top, "time": time.time()}

    log.info(
        "Hiring manager search complete: %d results for '%s' at '%s'",
        len(top), job_title, company,
    )
    return top
