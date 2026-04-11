import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("APIscraper")

# ── Adzuna API ───────────────────────────────────────────────────────────────
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search"
RESULTS_PER_PAGE = 50

# ── Reed API ─────────────────────────────────────────────────────────────────
REED_API_KEY = os.getenv("REED_API_KEY")

# ── Job filters ──────────────────────────────────────────────────────────────
ENTRY_LEVEL_KEYWORDS = "junior graduate trainee apprentice entry-level"
SALARY_MAX = 35000  # GBP — cap to target entry-level roles

# ── Sponsor matching ────────────────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 85  # minimum similarity score (0-100)
GOV_UK_SPONSORS_PAGE = (
    "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"
)
GOV_UK_CONTENT_API = (
    "https://www.gov.uk/api/content/government/publications/"
    "register-of-licensed-sponsors-workers"
)

# ── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # cost-effective for batch scoring

# ── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
