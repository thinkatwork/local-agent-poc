"""Web search providers. Tavily is the first one; swap it by adding another class.

Only Tavily's basic search endpoint is used (1 credit per request at the time
of writing). Tavily extract/crawl are never called: pages are fetched by the
app itself with the read-only fetcher in agent/web.py.
"""
from __future__ import annotations

from typing import Protocol

from ..web import TAVILY_LIMIT_MESSAGE, search_web


class SearchQuotaError(Exception):
    """The search provider's credit or rate limit was reached."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str) -> tuple[str, list[dict[str, str]]]: ...


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str, transport=None):
        self._key = api_key
        self._transport = transport

    def search(self, query: str) -> tuple[str, list[dict[str, str]]]:
        text, results = search_web(query, self._key, transport=self._transport)
        if text == TAVILY_LIMIT_MESSAGE:
            raise SearchQuotaError(
                "Tavily's rate or credit limit was reached, so searching stopped. Check usage at "
                "https://app.tavily.com/home. The app does not switch on pay-as-you-go."
            )
        return text, results
