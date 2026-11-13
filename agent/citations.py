"""Check the writer's citations against the pages the app actually fetched.

The writer is asked to cite each source with a short exact quote. Python then
looks for that quote in the text of the cited page. This needs no model call,
search or network request: it only compares text the app already holds.

A found quote shows that the text is on the cited page. It does NOT show that
the quote supports the claim it is attached to.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from .web import UNTRUSTED_END, UNTRUSTED_START

VERIFIED = "verified"  # quote found on the cited page (ignoring case, punctuation and spacing)
CLOSE = "close"  # nearly all words found in order; small differences
NOT_FOUND = "not_found"  # page was fetched, but the quote is not on it
NOT_FETCHED = "not_fetched"  # the app never read this URL
NO_QUOTE = "no_quote"  # URL cited without a quote, or a quote too short to check

STATUS_LABELS = {
    VERIFIED: "✅ Quote found on the page",
    CLOSE: "🟡 Quote nearly matches the page (small differences)",
    NOT_FOUND: "⚠️ Quote not found on the cited page",
    NOT_FETCHED: "❌ Page was not read by the app",
    NO_QUOTE: "➖ No quote to check",
}

MIN_QUOTE_WORDS = 4
CLOSE_THRESHOLD = 0.85  # share of quote words matched, in order, within one window of the page
MAX_ANCHOR_POSITIONS = 200  # bounds the fuzzy search on long pages

_URL_RE = re.compile(r"https?://[^\s<>\"“”\]]+")
# After a URL: up to 80 characters (for example ", 2026-01-01, ") and then a quoted passage.
_QUOTE_AFTER_URL_RE = re.compile(r"[^\"“\n]{0,80}?[\"“]([^\"”\n]{6,400})[\"”]")
_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"'})


def _clean_url(url: str) -> str:
    """Strip trailing punctuation and unbalanced closing brackets picked up from prose."""
    while True:
        before = url
        url = url.rstrip(".,;:!?'*_")
        if url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        if url == before:
            return url


def url_key(url: str) -> str:
    """Compare URLs ignoring scheme, `www.`, fragment and a trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path.rstrip("/")
    return f"{host}{path}" + (f"?{parts.query}" if parts.query else "")


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS).lower()
    return re.findall(r"[^\W_]+", text)


def page_body(fetched_text: str) -> str:
    """The page text from `fetch_page` output, without the title/URL header."""
    start = fetched_text.find(UNTRUSTED_START)
    if start == -1:
        return fetched_text
    body = fetched_text[start + len(UNTRUSTED_START) :]
    end = body.rfind(UNTRUSTED_END)
    return body[:end] if end != -1 else body


def extract_citations(answer: str) -> list[dict[str, str]]:
    """Find each cited URL and the quote that follows it, in order, without duplicates."""
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _URL_RE.finditer(answer):
        url = _clean_url(match.group(0))
        tail = answer[match.start() + len(url) :]
        quote_match = _QUOTE_AFTER_URL_RE.match(tail)
        quote = ""
        if quote_match and "http" not in tail[: quote_match.start(1)]:
            quote = quote_match.group(1).strip()
        key = (url_key(url), " ".join(_words(quote)))
        if key in seen:
            continue
        seen.add(key)
        found.append({"url": url, "quote": quote})
    return found


def _matched_share(quote: list[str], page: list[str]) -> float:
    """Best share of quote words found in order within one window of the page."""
    n = len(quote)
    best = 0.0
    # Anchor on the longest (usually rarest) words, so one changed word does not hide the match.
    anchors = sorted({(len(w), w) for w in quote}, reverse=True)[:3]
    for _, anchor in anchors:
        offset = quote.index(anchor)
        positions = [i for i, w in enumerate(page) if w == anchor][:MAX_ANCHOR_POSITIONS]
        for i in positions:
            start = max(0, i - offset - 3)
            window = page[start : i - offset + n + 3]
            matcher = SequenceMatcher(None, quote, window, autojunk=False)
            share = sum(block.size for block in matcher.get_matching_blocks()) / n
            best = max(best, share)
            if best == 1.0:
                return best
    return best


def check_quote(quote: str, page_text: str) -> str:
    quote_words = _words(quote)
    if len(quote_words) < MIN_QUOTE_WORDS:
        return NO_QUOTE
    page_words = _words(page_text)
    if f" {' '.join(quote_words)} " in f" {' '.join(page_words)} ":
        return VERIFIED
    return CLOSE if _matched_share(quote_words, page_words) >= CLOSE_THRESHOLD else NOT_FOUND


def check_citations(answer: str, pages: dict[str, str]) -> list[dict[str, str]]:
    """Check every citation in `answer`.

    `pages` maps each URL the app fetched (requested and final URL) to the text it got back.
    """
    by_key = {url_key(url): text for url, text in pages.items()}
    results = []
    for citation in extract_citations(answer):
        text = by_key.get(url_key(citation["url"]))
        if text is None:
            status = NOT_FETCHED
        elif not citation["quote"]:
            status = NO_QUOTE
        else:
            status = check_quote(citation["quote"], page_body(text))
        results.append({**citation, "status": status})
    return results


def summarize(citations: list[dict[str, str]]) -> str:
    if not citations:
        return "Citation check: the answer cites no web pages."
    found = sum(c["status"] in (VERIFIED, CLOSE) for c in citations)
    quoted = sum(c["status"] != NO_QUOTE for c in citations)
    return f"Citation check: {found}/{len(citations)} citations have a quote found on the cited page" + (
        "" if quoted == len(citations) else f" ({len(citations) - quoted} without a checkable quote)"
    )
