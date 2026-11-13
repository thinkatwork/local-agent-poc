"""Offline tests for the citation check. No network or model access is needed."""
from __future__ import annotations

import unittest

from agent import citations
from agent.citations import CLOSE, NO_QUOTE, NOT_FETCHED, NOT_FOUND, VERIFIED, check_citations, extract_citations
from agent.providers import ModelResult
from agent.web import UNTRUSTED_END, UNTRUSTED_START
from agent.workflow import Limits, run_research

PAGE_TEXT = (
    "The Lumen bridge opened to traffic on 3 March 2025 after six years of construction. "
    "It carries roughly 40,000 vehicles a day, according to the regional transport authority. "
    "Critics argued the final cost—€1.2 billion—was nearly double the original estimate."
)


def fetched(url: str, text: str = PAGE_TEXT) -> str:
    return f"Title: Bridge\nURL: {url}\nPublished: 2025-03-04\n{UNTRUSTED_START}\n{text}\n{UNTRUSTED_END}"


class ExtractTests(unittest.TestCase):
    def test_parses_writer_format(self):
        answer = (
            'It opened in 2025 (Bridge news, https://news.example.com/bridge, 2025-03-04, '
            '"opened to traffic on 3 March 2025"). Traffic is high (Other, https://b.example.org/x?id=1, '
            'date not stated, “roughly 40,000 vehicles a day”).'
        )
        self.assertEqual(
            extract_citations(answer),
            [
                {"url": "https://news.example.com/bridge", "quote": "opened to traffic on 3 March 2025"},
                {"url": "https://b.example.org/x?id=1", "quote": "roughly 40,000 vehicles a day"},
            ],
        )

    def test_quote_is_not_borrowed_from_the_next_citation(self):
        answer = 'A (T, https://a.example.com/1, 2025). B (T, https://b.example.com/2, 2025, "some quoted words here")'
        found = extract_citations(answer)
        self.assertEqual(found[0], {"url": "https://a.example.com/1", "quote": ""})
        self.assertEqual(found[1]["quote"], "some quoted words here")

    def test_url_cleanup_keeps_balanced_parentheses(self):
        answer = "See (https://en.wikipedia.org/wiki/Bridge_(structure)). Also [x](https://a.example.com/p)."
        urls = [c["url"] for c in extract_citations(answer)]
        self.assertEqual(urls, ["https://en.wikipedia.org/wiki/Bridge_(structure)", "https://a.example.com/p"])

    def test_duplicates_are_dropped(self):
        answer = 'x (https://a.example.com/p, "one two three four") y (https://www.a.example.com/p/, "One, two three four")'
        self.assertEqual(len(extract_citations(answer)), 1)


class CheckTests(unittest.TestCase):
    URL = "https://news.example.com/bridge"

    def status(self, quote: str, url: str | None = None) -> str:
        answer = f'Claim ({url or self.URL}, 2025, "{quote}")'
        return check_citations(answer, {self.URL: fetched(self.URL)})[0]["status"]

    def test_exact_quote_is_verified_ignoring_case_punctuation_and_spacing(self):
        self.assertEqual(self.status("It carries roughly 40,000 vehicles a day"), VERIFIED)
        self.assertEqual(self.status("the final cost - €1.2 billion - was nearly double"), VERIFIED)
        self.assertEqual(self.status("OPENED  to traffic on 3 March 2025"), VERIFIED)

    def test_small_difference_is_close(self):
        self.assertEqual(self.status("Critics argued that the final cost was nearly double the original estimate"), CLOSE)

    def test_invented_or_changed_quote_is_not_found(self):
        self.assertEqual(self.status("the bridge was opened by the prime minister in a ceremony"), NOT_FOUND)
        self.assertEqual(self.status("It carries roughly 90,000 trucks every single week"), NOT_FOUND)

    def test_url_not_fetched(self):
        self.assertEqual(self.status("opened to traffic on 3 March 2025", "https://elsewhere.example.com/"), NOT_FETCHED)

    def test_short_or_missing_quote_is_not_checked(self):
        self.assertEqual(self.status("six years"), NO_QUOTE)
        result = check_citations(f"Claim ({self.URL}, 2025).", {self.URL: fetched(self.URL)})
        self.assertEqual(result[0]["status"], NO_QUOTE)

    def test_page_header_does_not_count_as_page_text(self):
        self.assertEqual(self.status("Title: Bridge URL: https://news.example.com/bridge"), NOT_FOUND)

    def test_url_variants_match(self):
        self.assertEqual(self.status("opened to traffic on 3 March 2025", "http://www.news.example.com/bridge/#top"), VERIFIED)

    def test_long_page_is_bounded(self):
        page = fetched(self.URL, "word filler " * 50_000 + PAGE_TEXT)
        answer = f'x ({self.URL}, "Critics argued that the final cost was nearly double the original estimate")'
        self.assertEqual(check_citations(answer, {self.URL: page})[0]["status"], CLOSE)

    def test_summary(self):
        results = [{"status": VERIFIED}, {"status": CLOSE}, {"status": NOT_FOUND}, {"status": NO_QUOTE}]
        self.assertEqual(
            citations.summarize(results),
            "Citation check: 2/4 citations have a quote found on the cited page (1 without a checkable quote)",
        )
        self.assertIn("cites no web pages", citations.summarize([]))


class WorkflowCitationTests(unittest.TestCase):
    def test_workflow_checks_writer_answer_without_extra_calls(self):
        url = "https://site0.example.com/1"
        writer_answer = (
            f'It opened in 2025 (Bridge, {url}, 2025-03-04, "opened to traffic on 3 March 2025"). '
            f'It was cheap (Bridge, {url}, 2025-03-04, "the bridge came in well under its budget"). '
            '(Other, https://never-fetched.example.com/, 2025, "opened to traffic on 3 March 2025")'
        )

        class Model:
            name, is_cloud, calls = "fake", False, 0

            def generate(self, system, prompt, max_output_tokens):
                Model.calls += 1
                return ModelResult(text=writer_answer if "answer writer" in system else "notes", input_tokens=1, output_tokens=1)

        class Search:
            name = "fake-search"

            def search(self, query):
                return "results", [{"title": "Bridge", "url": url, "date": "2025-03-04"}]

        fetches = []

        def fetcher(u):
            fetches.append(u)
            return fetched(u), {"title": "Bridge", "url": u, "date": "2025-03-04"}

        result = run_research("When did it open?", Model(), Search(), Limits(max_model_calls=2), review=False, fetcher=fetcher)
        self.assertEqual(Model.calls, 2)  # the check adds no model call
        self.assertEqual(fetches, [url])  # and no extra fetch
        self.assertEqual([c["status"] for c in result.citations], [VERIFIED, NOT_FOUND, NOT_FETCHED])
        self.assertTrue(any(step["name"].startswith("citation_check") for step in result.trace))

    def test_no_citations_when_stopped_before_writer(self):
        class Search:
            name = "fake-search"

            def search(self, query):
                return "none", []

        model = type("Model", (), {"name": "fake", "is_cloud": False})()
        result = run_research("q", model, Search(), Limits(), fetcher=lambda u: ("", None))
        self.assertIsNone(result.citations)


if __name__ == "__main__":
    unittest.main()
