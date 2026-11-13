"""Settings loaded from the project's .env file (never committed).

All provider choices, keys, limits and prices come from `.env` or the shell
environment, never from source code or UI text fields. Invalid settings are
reported as errors; the app never silently switches to another provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Values already set in the shell take precedence over .env.
load_dotenv(ENV_FILE, override=False)

PROVIDERS = ("ollama", "gemini")
GEMINI_TIERS = ("free", "paid")


@dataclass(frozen=True)
class Settings:
    model_provider: str = "ollama"
    ollama_model: str = "qwen3:8b"
    gemini_api_key: str | None = None
    gemini_model: str = ""
    gemini_tier: str = "free"
    tavily_api_key: str | None = None
    max_model_calls: int = 3
    max_searches: int = 3
    max_fetches: int = 4
    max_output_tokens: int = 1024
    max_input_chars: int = 40_000
    request_timeout_seconds: int = 60
    gemini_input_price_per_m: float | None = None
    gemini_output_price_per_m: float | None = None
    max_estimated_spend_per_question: float | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    def gemini_problems(self) -> list[str]:
        """Reasons Gemini cannot be used with the current settings."""
        problems = []
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set in .env.")
        if not self.gemini_model:
            problems.append("GEMINI_MODEL is not set in .env.")
        if self.gemini_tier == "paid":
            if self.gemini_input_price_per_m is None or self.gemini_output_price_per_m is None:
                problems.append("GEMINI_TIER=paid requires GEMINI_INPUT_PRICE_PER_M and GEMINI_OUTPUT_PRICE_PER_M.")
            if self.max_estimated_spend_per_question is None:
                problems.append("GEMINI_TIER=paid requires MAX_ESTIMATED_SPEND_PER_QUESTION.")
        return problems


def _text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int, low: int, high: int, errors: list[str]) -> int:
    raw = _text(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name} must be a whole number (got {raw!r}); using {default}.")
        return default
    if not low <= value <= high:
        errors.append(f"{name} must be between {low} and {high} (got {value}); using {default}.")
        return default
    return value


def _price(name: str, errors: list[str]) -> float | None:
    raw = _text(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{name} must be a number in USD (got {raw!r}); it is ignored.")
        return None
    if value < 0:
        errors.append(f"{name} cannot be negative; it is ignored.")
        return None
    return value


def load_settings() -> Settings:
    """Read and validate settings. Problems are collected, never hidden."""
    errors: list[str] = []
    provider = _text("MODEL_PROVIDER", "ollama").lower() or "ollama"
    if provider not in PROVIDERS:
        errors.append(f"MODEL_PROVIDER must be 'ollama' or 'gemini' (got {provider!r}). No model will be used until this is fixed.")
    tier = _text("GEMINI_TIER", "free").lower() or "free"
    if tier not in GEMINI_TIERS:
        errors.append(f"GEMINI_TIER must be 'free' or 'paid' (got {tier!r}); treating it as 'free'.")
        tier = "free"
    return Settings(
        model_provider=provider,
        ollama_model=_text("OLLAMA_MODEL", "qwen3:8b") or "qwen3:8b",
        gemini_api_key=_text("GEMINI_API_KEY") or None,
        gemini_model=_text("GEMINI_MODEL"),
        gemini_tier=tier,
        tavily_api_key=_text("TAVILY_API_KEY") or None,
        max_model_calls=_int("MAX_MODEL_CALLS_PER_QUESTION", 3, 1, 10, errors),
        max_searches=_int("MAX_SEARCHES_PER_QUESTION", 3, 0, 5, errors),
        max_fetches=_int("MAX_FETCHES_PER_QUESTION", 4, 0, 8, errors),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS_PER_CALL", 1024, 64, 8192, errors),
        max_input_chars=_int("MAX_INPUT_CHARS_PER_CALL", 40_000, 2_000, 200_000, errors),
        request_timeout_seconds=_int("MODEL_REQUEST_TIMEOUT_SECONDS", 60, 5, 300, errors),
        gemini_input_price_per_m=_price("GEMINI_INPUT_PRICE_PER_M", errors),
        gemini_output_price_per_m=_price("GEMINI_OUTPUT_PRICE_PER_M", errors),
        max_estimated_spend_per_question=_price("MAX_ESTIMATED_SPEND_PER_QUESTION", errors),
        errors=tuple(errors),
    )


def secrets(settings: Settings) -> list[str]:
    """Configured secret values, so they can be removed from any displayed text."""
    return [value for value in (settings.gemini_api_key, settings.tavily_api_key) if value]


def redact(text: str, secret_values: list[str]) -> str:
    for value in secret_values:
        if value:
            text = text.replace(value, "[redacted]")
    return text


def tavily_api_key() -> str | None:
    """Return the Tavily API key, or None if it is not configured."""
    value = os.environ.get("TAVILY_API_KEY", "").strip()
    return value or None
