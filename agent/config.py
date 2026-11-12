"""Local settings loaded from the project's .env file (never committed)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Values already set in the shell take precedence over .env.
load_dotenv(ENV_FILE, override=False)


def tavily_api_key() -> str | None:
    """Return the Tavily API key, or None if it is not configured."""
    value = os.environ.get("TAVILY_API_KEY", "").strip()
    return value or None
