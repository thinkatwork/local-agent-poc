"""Model providers: one small interface for local Ollama and cloud Gemini.

The research workflow only needs single-turn text generation. Tools stay in
Python, so no provider is given function-calling access to anything.
Neither provider retries a failed request, and errors never include API keys.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from ..jobs import check_cancel


class ProviderError(Exception):
    """A model request failed. The message is safe to show to the user."""


class ProviderQuotaError(ProviderError):
    """The provider reported a quota, rate or billing limit."""


@dataclass
class ModelResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    truncated: bool = False


class ModelProvider(Protocol):
    name: str
    model: str
    is_cloud: bool

    def generate(self, system: str, prompt: str, max_output_tokens: int) -> ModelResult: ...


class OllamaProvider:
    """Local inference through the Ollama service on this Mac.

    Replies are streamed so a cancel can stop generation part-way: closing the
    stream drops the HTTP connection and Ollama stops generating.
    """

    name = "ollama"
    is_cloud = False

    def __init__(self, model: str, chat_fn: Any = None, cancel: threading.Event | None = None):
        self.model = model
        if chat_fn is None:
            from ollama import chat as chat_fn
        self._chat = chat_fn
        self._cancel = cancel

    def _failed(self, exc: Exception) -> ProviderError:
        return ProviderError(
            f"Ollama request failed ({type(exc).__name__}). Check that Ollama is running and "
            f"`{self.model}` is installed (`ollama pull {self.model}`)."
        )

    def generate(self, system: str, prompt: str, max_output_tokens: int) -> ModelResult:
        check_cancel(self._cancel)
        try:
            stream = self._chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                think=False,
                stream=True,
                options={"temperature": 0, "num_predict": max_output_tokens},
            )
        except Exception as exc:
            raise self._failed(exc) from None
        parts: list[str] = []
        last = None
        try:
            for chunk in stream:
                if self._cancel is not None and self._cancel.is_set():
                    break
                parts.append(chunk.message.content or "")
                last = chunk
        except Exception as exc:
            raise self._failed(exc) from None
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        check_cancel(self._cancel)
        return ModelResult(
            text="".join(parts).strip(),
            input_tokens=getattr(last, "prompt_eval_count", None),
            output_tokens=getattr(last, "eval_count", None),
            truncated=getattr(last, "done_reason", None) == "length",
        )


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


class GeminiProvider:
    """Google Gemini through the Interactions API (official google-genai SDK).

    - store=False: Google is asked not to keep the interaction for later retrieval.
    - automatic_continuation=False: one request per call, so the call count is exact.
    - retry attempts = 1: a failed paid request is never retried.
    - max_output_tokens and a request timeout are always set.
    """

    name = "gemini"
    is_cloud = True

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60, tier: str = "free", client: Any = None):
        if not api_key or not model:
            raise ProviderError("Gemini is not configured: set GEMINI_API_KEY and GEMINI_MODEL in .env.")
        self.model = model
        self.tier = tier
        self._key = api_key
        self._timeout = timeout_seconds
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        self._client = client

    def _safe(self, text: str) -> str:
        return text.replace(self._key, "[redacted]")[:300]

    def generate(self, system: str, prompt: str, max_output_tokens: int) -> ModelResult:
        try:
            interaction = self._client.interactions.create(
                model=self.model,
                input=prompt,
                system_instruction=system,
                store=False,
                generation_config={"max_output_tokens": max_output_tokens, "thinking_level": "minimal"},
                automatic_continuation=False,
                timeout=self._timeout,
            )
        except Exception as exc:
            code = _status_code(exc)
            if code == 429:
                tier_hint = (
                    "You are on the free-tier setting: wait for the quota to reset, or read the README before "
                    "deciding whether to enable paid use. The app will not continue or upgrade billing."
                    if self.tier == "free"
                    else "Check your rate limits and billing in Google AI Studio. The app will not retry."
                )
                raise ProviderQuotaError(f"Gemini quota or rate limit reached (HTTP 429). {tier_hint}") from None
            if code in (401, 403):
                raise ProviderError(
                    "Gemini rejected the request (HTTP %s). Check GEMINI_API_KEY in .env and that the key's "
                    "project can use the Gemini API." % code
                ) from None
            if code == 404:
                raise ProviderError(f"Gemini model `{self.model}` was not found. Check GEMINI_MODEL in .env.") from None
            detail = self._safe(str(exc))
            raise ProviderError(f"Gemini request failed ({type(exc).__name__}, HTTP {code or 'n/a'}): {detail}") from None

        status = str(getattr(interaction, "status", "") or "")
        if status == "failed":
            raise ProviderError("Gemini reported that the interaction failed. Nothing was retried.")
        usage = getattr(interaction, "usage", None)
        return ModelResult(
            text=(getattr(interaction, "output_text", None) or "").strip(),
            input_tokens=getattr(usage, "total_input_tokens", None),
            output_tokens=getattr(usage, "total_output_tokens", None),
            thought_tokens=getattr(usage, "total_thought_tokens", None),
            truncated=status in ("incomplete", "requires_continuation"),
        )
