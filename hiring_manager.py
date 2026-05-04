"""
Hiring Manager Finder

Uses OpenAI web search to find real decision-makers at a company
(C-level, founders, owners, senior managers) and their LinkedIn pages.

Approach:
  1. Find the company's verified LinkedIn page via web search.
  2. Search for C-level / founders / senior managers at the company.
  3. Return real names with LinkedIn search URLs + company page.
  4. Cache results by company + job_title to avoid repeat API calls.

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


def _search_company_linkedin(client: OpenAI, company: str) -> str | None:
    """Find the verified LinkedIn company page URL via web search."""
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=(
                f"What is the official LinkedIn company page URL for {company}? "
                f"Find the URL in the format linkedin.com/company/... "
                f"Return ONLY the URL, nothing else."
            ),
        )
        text = ""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if hasattr(content, "text"):
                        text += content.text

        match = re.search(r"https?://(?:www\.)?linkedin\.com/company/[A-Za-z0-9_-]+", text)
        if match:
            url = match.group(0).rstrip("/")
            log.info("  Company LinkedIn page: %s", url)
            return url
        return None
    except Exception as e:
        log.warning("  Company LinkedIn search failed: %s", e)
        return None


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
            # Remove markdown links like ([source](url))
            title = re.sub(r"\s*\(?\[.*?\]\(.*?\)\)?", "", title).strip()

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


def _find_people_at_company(client: OpenAI, company: str, job_title: str) -> tuple[list[dict], str | None]:
    """Search for real decision-makers and the company LinkedIn page.
    Runs all searches in parallel. Returns (people_list, company_url)."""
    from concurrent.futures import ThreadPoolExecutor

    # Target C-level, owners, founders, senior managers
    roles = [
        "CEO or Managing Director or Owner",
        "Founder or Co-Founder",
        f"Senior Manager or Director of {job_title}",
        "COO or Operations Director",
    ]

    people = []
    seen_lastnames = set()
    company_url = None

    with ThreadPoolExecutor(max_workers=5) as pool:
        # Search for people + company page in parallel
        people_futures = {pool.submit(_search_one_role, client, role, company): role for role in roles}
        company_future = pool.submit(_search_company_linkedin, client, company)

        for future in people_futures:
            result = future.result()
            if result:
                # Deduplicate by last name (catches "Nik Storonsky" vs "Nikolay Storonsky")
                lastname = result["full_name"].split()[-1].lower()
                if lastname not in seen_lastnames:
                    seen_lastnames.add(lastname)
                    people.append(result)

        company_url = company_future.result()

    log.info("Found %d real people at '%s' from %d role searches", len(people), company, len(roles))
    return people, company_url



def _score_people(people: list[dict], job_title: str) -> list[dict]:
    """Assign AI scores based on seniority and decision-making authority."""
    for person in people:
        title_lower = person.get("title", "").lower()

        if any(kw in title_lower for kw in ["ceo", "owner", "founder", "co-founder", "managing director", "chairman"]):
            person["ai_score"] = 95
            person["reason"] = f"Top-level decision-maker — ultimate authority over hiring for {job_title}."
        elif any(kw in title_lower for kw in ["coo", "cfo", "cto", "chief", "president"]):
            person["ai_score"] = 90
            person["reason"] = f"C-suite executive — key decision-maker for {job_title}."
        elif any(kw in title_lower for kw in ["director", "vp ", "vice president", "head of", "partner"]):
            person["ai_score"] = 85
            person["reason"] = f"Senior leadership — likely involved in hiring decisions for {job_title}."
        elif any(kw in title_lower for kw in ["senior manager", "general manager", "regional manager"]):
            person["ai_score"] = 75
            person["reason"] = f"Senior management — strong influence over hiring for {job_title}."
        elif any(kw in title_lower for kw in ["manager", "lead", "supervisor"]):
            person["ai_score"] = 65
            person["reason"] = f"Management role — may influence hiring for {job_title}."
        else:
            person["ai_score"] = 50
            person["reason"] = f"Relevant team member who may be involved in hiring for {job_title}."

    people.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
    return people



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
    people, company_url = _find_people_at_company(client, company, job_title)

    if people:
        people = _score_people(people, job_title)
        log.info("Found %d real people at '%s'", len(people), company)

    # Always append company LinkedIn page (verified or guessed)
    company_entry = {
        "full_name": company,
        "title": "Company LinkedIn Page",
        "linkedin_url": company_url or f"https://www.linkedin.com/company/{quote_plus(company.strip())}",
        "ai_score": 70 if not people else 40,
        "reason": f"Browse {company}'s LinkedIn page to find employees and open positions.",
        "is_verified": False,
    }
    people.append(company_entry)

    # Keep top results
    top = people[:max_results]

    # Cache results
    _cache[key] = {"data": top, "time": time.time()}

    log.info(
        "Hiring manager search complete: %d results for '%s' at '%s'",
        len(top), job_title, company,
    )
    return top
