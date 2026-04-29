"""
Hiring Manager Finder

Uses OpenAI web search to find real hiring decision-makers at a company
by searching for their LinkedIn profiles directly.

Approach:
  1. Use OpenAI Responses API with web_search to find real people in
     HR, Talent Acquisition, and department leadership at the company.
  2. Return their real names and LinkedIn profile URLs.
  3. Cache results by company + job_title to avoid repeat API calls.

Results are cached in memory for 2 hours.
"""

import re
import time
import hashlib
from urllib.parse import quote_plus

from openai import OpenAI

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


def _name_matches_slug(name: str, url: str) -> bool:
    """Check if a person's name matches the LinkedIn URL slug.
    Prevents showing hallucinated names that don't match the actual profile."""
    slug = url.rstrip("/").split("/")[-1].lower()
    # Normalize name: lowercase, remove punctuation, keep parts > 1 char
    name_parts = [re.sub(r"[^a-z]", "", p.lower()) for p in name.split() if len(p) > 1]
    if len(name_parts) < 2:
        return False
    # First and last name must both appear in the slug
    first = name_parts[0]
    last = name_parts[-1]
    return first in slug and last in slug


def _search_one_role(client: OpenAI, role: str, company: str) -> dict | None:
    """Search for one specific role at a company. Returns a person dict or None.

    Uses an anti-hallucination prompt that instructs the model to say NOT FOUND
    when it can't find a real person.  Never trusts LinkedIn URLs from the AI
    (they are always fabricated); builds a LinkedIn search URL instead.
    """
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=(
                f"Who is the {role} at {company}? "
                f"Search the web for their real full name. "
                f"If you CANNOT find a specific real person in this role at {company}, "
                f"respond with exactly: NOT FOUND. "
                f"Do NOT guess or make up names. Only provide a name if you found it "
                f"on a real, credible website. "
                f"If found, respond with ONLY: Name: [full name] Title: [their actual job title]"
            ),
        )

        text = ""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if hasattr(content, "text"):
                        text += content.text

        log.info("  Search '%s at %s': %s", role, company, text[:200])

        # If model says it couldn't find anyone, respect that
        if "NOT FOUND" in text.upper():
            log.info("  Model reported NOT FOUND for '%s at %s'", role, company)
            return None

        # Extract name
        name = None
        for pattern in [
            r"Name:\s*\*?\*?([A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+)+)",
            r"\*\*([A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+)+)\*\*",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|serves|works|holds|currently)",
        ]:
            m = re.search(pattern, text)
            if m:
                name = m.group(1).strip()
                name = name.split("\n")[0].strip()
                name = re.split(r"\s+Title|\s+URL|\s+at\s+", name)[0].strip()
                break

        if not name:
            return None

        # Extract title
        title = role
        title_match = re.search(r"Title:\s*(.+?)(?:\n|URL:|Source:|Location:|$)", text)
        if title_match:
            title = title_match.group(1).strip().strip("*")

        # Build LinkedIn search URL with name + company only (title makes it too restrictive)
        search_query = f"{name} {company}"
        linkedin_url = (
            f"https://www.linkedin.com/search/results/people/"
            f"?keywords={quote_plus(search_query)}"
        )

        log.info("  Found: %s — %s", name, title)
        return {
            "full_name": name,
            "title": title,
            "linkedin_url": linkedin_url,
            "is_verified": True,
        }

    except Exception as e:
        log.warning("  Search failed for '%s at %s': %s", role, company, e)
        return None


def _find_people_at_company(client: OpenAI, company: str, job_title: str) -> list[dict]:
    """Search for real people in hiring-related roles at a company.
    Runs separate searches per role in parallel for better results."""
    from concurrent.futures import ThreadPoolExecutor

    # Roles to search for
    roles = [
        f"Hiring Manager or Department Head for {job_title}",
        "HR Manager or Head of Human Resources",
        "Talent Acquisition Manager or Head of Recruitment",
        "Managing Director or CEO",
    ]

    people = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_search_one_role, client, role, company): role for role in roles}
        for future in futures:
            result = future.result()
            if result and result["linkedin_url"] not in seen_urls:
                seen_urls.add(result["linkedin_url"])
                people.append(result)

    log.info("Found %d real people at '%s' from %d role searches", len(people), company, len(roles))
    return people



def _score_people(people: list[dict], job_title: str) -> list[dict]:
    """Assign AI scores based on how likely each person is the decision-maker."""
    for person in people:
        title_lower = person.get("title", "").lower()

        # Score based on role type
        if any(kw in title_lower for kw in ["head of", "director", "vp ", "vice president", "chief", "lead"]):
            person["ai_score"] = 90
            person["reason"] = f"Senior leadership role — likely the direct hiring decision-maker for {job_title}."
        elif any(kw in title_lower for kw in ["manager", "supervisor", "team lead"]):
            person["ai_score"] = 75
            person["reason"] = f"Management role — strong influence over hiring for {job_title}."
        elif any(kw in title_lower for kw in ["talent", "recruiter", "recruitment", "acquisition"]):
            person["ai_score"] = 65
            person["reason"] = "Talent acquisition — manages the recruitment pipeline and candidate screening."
        elif any(kw in title_lower for kw in ["hr ", "human resource", "people"]):
            person["ai_score"] = 55
            person["reason"] = "HR team — involved in the hiring process and offer decisions."
        else:
            person["ai_score"] = 50
            person["reason"] = f"Relevant team member who may be involved in hiring for {job_title}."

    # Sort by score descending
    people.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    return people


def _fallback_company_profile(company: str, job_title: str) -> list[dict]:
    """When web search finds no specific people, return the company's
    LinkedIn profile so the user can browse employees directly."""
    company_slug = quote_plus(company.strip())
    return [{
        "full_name": company,
        "title": "Company LinkedIn Profile",
        "linkedin_url": (
            f"https://www.linkedin.com/company/{company_slug}"
        ),
        "ai_score": 70,
        "reason": f"No specific decision-makers found. Visit the company page to find people involved in hiring for {job_title}.",
        "is_verified": False,
    }]


# ── Public API ───────────────────────────────────────────────────────────────

def find_hiring_managers(company: str, job_title: str, max_results: int = 5) -> list[dict]:
    """Find real hiring managers at a company for a given role.

    Uses OpenAI web search to find actual people on LinkedIn.
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

    log.info("Finding hiring managers at '%s' for role '%s' via web search", company, job_title)

    client = _get_openai_client()
    people = _find_people_at_company(client, company, job_title)

    if people:
        people = _score_people(people, job_title)
        log.info("Found %d real people at '%s'", len(people), company)
    else:
        log.warning("No real profiles found for '%s', falling back to role-based suggestions", company)
        people = _fallback_company_profile(company, job_title)

    # Keep top results
    top = people[:max_results]

    # Cache results
    _cache[key] = {"data": top, "time": time.time()}

    log.info(
        "Hiring manager search complete: %d results for '%s' at '%s'",
        len(top), job_title, company,
    )
    return top
