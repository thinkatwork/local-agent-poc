"""Offline tests for settings, providers and the bounded research workflow.

Run with:  python -m unittest discover -s tests
No network, Ollama, Gemini or Tavily access is needed; everything is faked.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import config, workflow
from agent.providers import GeminiProvider, ModelResult, ProviderError, ProviderQuotaError, SearchQuotaError
from agent.workflow import Limits, Pricing, run_research

SECRET_DOC_TEXT = "PRIVATE-DOCUMENT-TEXT-9911"


class FakeModel:
    name = "fake"

    def __init__(self, is_cloud=True, fail_on=None, usage=(1000, 100, 10)):
        self.is_cloud = is_cloud
        self.model = "fake-model"
        self.prompts: list[tuple[str, str, int]] = []
        self.fail_on = fail_on
        self.usage = usage

    def generate(self, system, prompt, max_output_tokens):
        self.prompts.append((system, prompt, max_output_tokens))
        if self.fail_on is not None and len(self.prompts) == self.fail_on:
            raise ProviderQuotaError("Gemini quota or rate limit reached (HTTP 429).")
        i, o, t = self.usage
        return ModelResult(text=f"output {len(self.prompts)}", input_tokens=i, output_tokens=o, thought_tokens=t)


class FakeSearch:
    name = "fake-search"

    def __init__(self, quota_after=None):
        self.queries: list[str] = []
        self.quota_after = quota_after

    def search(self, query):
        if self.quota_after is not None and len(self.queries) >= self.quota_after:
            raise SearchQuotaError("Tavily's rate or credit limit was reached.")
        self.queries.append(query)
        n = len(self.queries)
        results = [{"title": f"R{n}-{i}", "url": f"https://site{i}.example.com/{n}", "date": "2026-01-01"} for i in range(5)]
        return "untrusted results", results


def fake_fetch(fetched):
    def fetch(url):
        fetched.append(url)
        return f"page text for {url}", {"title": "Page", "url": url, "date": "2026-01-01"}
    return fetch


class WorkflowTests(unittest.TestCase):
    def test_three_calls_with_review_and_limits_enforced(self):
        model, search, fetched = FakeModel(), FakeSearch(), []
        limits = Limits(max_model_calls=3, max_searches=3, max_fetches=4, max_output_tokens=500)
        result = run_research("What happened?", model, search, limits, review=True, fetcher=fake_fetch(fetched))
        self.assertEqual(len(model.prompts), 3)  # researcher, reviewer, writer; no room for planning
        self.assertEqual(len(search.queries), 1)
        self.assertEqual(len(fetched), 4)
        self.assertEqual(len({u.split("/")[2] for u in fetched}), 4)  # distinct sites preferred
        self.assertTrue(all(p[2] == 500 for p in model.prompts))
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.usage["fetches"], 4)
        self.assertIsNone(result.stopped)
        self.assertIn("Sources · Pages read", result.answer)

    def test_one_pass_mode_uses_two_calls(self):
        model = FakeModel()
        run_research("q", model, FakeSearch(), Limits(max_model_calls=2), review=False, fetcher=fake_fetch([]))
        self.assertEqual(len(model.prompts), 2)

    def test_planner_runs_only_when_budget_allows(self):
        model, search = FakeModel(), FakeSearch()
        model.generate = mock.Mock(side_effect=[ModelResult("a\nb\nc\nd\ne", 1, 1)] + [ModelResult("x", 1, 1)] * 3)
        run_research("q", model, search, Limits(max_model_calls=4, max_searches=3), review=True, fetcher=fake_fetch([]))
        self.assertEqual(search.queries, ["a", "b", "c"])  # capped at max_searches
        self.assertEqual(model.generate.call_count, 4)

    def test_call_limit_stops_before_extra_calls(self):
        model = FakeModel()
        result = run_research("q", model, FakeSearch(), Limits(max_model_calls=2), review=True, fetcher=fake_fetch([]))
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("model-call limit reached", result.stopped)
        self.assertIn("Nothing was retried", result.answer)

    def test_quota_error_stops_without_retry(self):
        model = FakeModel(fail_on=2)
        result = run_research("q", model, FakeSearch(), Limits(), review=True, fetcher=fake_fetch([]))
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("429", result.stopped)

    def test_search_quota_stops_search(self):
        result = run_research("q", FakeModel(), FakeSearch(quota_after=0), Limits(), fetcher=fake_fetch([]))
        self.assertIn("credit limit", result.answer)
        self.assertEqual(result.usage["model_calls"], 0)

    def test_spend_cap_blocks_call_before_sending(self):
        model = FakeModel(usage=(1_000_000, 0, 0))  # first call "costs" $1.00
        pricing = Pricing(input_per_m=1.0, output_per_m=1.0, max_spend=0.5)
        result = run_research("q", model, FakeSearch(), Limits(), pricing=pricing, fetcher=fake_fetch([]))
        self.assertEqual(len(model.prompts), 1)
        self.assertIn("MAX_ESTIMATED_SPEND_PER_QUESTION", result.stopped)

    def test_cost_estimate_counts_thinking_tokens(self):
        model = FakeModel(usage=(1_000_000, 1_000_000, 1_000_000))
        result = run_research("q", model, FakeSearch(), Limits(max_model_calls=2), pricing=Pricing(1.0, 2.0), review=False, fetcher=fake_fetch([]))
        self.assertAlmostEqual(result.usage["estimated_cost"], 2 * (1.0 + 2.0 * 2))

    def test_local_model_has_no_cost(self):
        result = run_research("q", FakeModel(is_cloud=False), FakeSearch(), Limits(), pricing=Pricing(1.0, 1.0), fetcher=fake_fetch([]))
        self.assertIsNone(result.usage["estimated_cost"])

    def test_local_documents_never_reach_the_model(self):
        with tempfile.TemporaryDirectory() as docs:
            with open(os.path.join(docs, "secret.md"), "w") as handle:
                handle.write(SECRET_DOC_TEXT)
            model = FakeModel()
            run_research(f"Summarise {docs}/secret.md", model, FakeSearch(), Limits(), fetcher=fake_fetch([]))
        self.assertFalse(any(SECRET_DOC_TEXT in s + p for s, p, _ in model.prompts))

    def test_input_is_capped_per_call(self):
        def big_fetch(url):
            return "x" * 100_000, {"title": "T", "url": url, "date": ""}
        model = FakeModel()
        run_research("q", model, FakeSearch(), Limits(max_input_chars=5_000), fetcher=big_fetch)
        self.assertTrue(all(len(p) <= 5_200 for _, p, _ in model.prompts))


class GeminiProviderTests(unittest.TestCase):
    def make(self, create):
        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        return GeminiProvider("AIza-secret-key", "gemini-test", timeout_seconds=7, client=client)

    def test_request_settings_and_usage(self):
        create = mock.Mock(return_value=SimpleNamespace(
            status="completed",
            output_text=" answer ",
            usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5, total_thought_tokens=2),
        ))
        result = self.make(create).generate("sys", "prompt", 321)
        kwargs = create.call_args.kwargs
        self.assertFalse(kwargs["store"])
        self.assertFalse(kwargs["automatic_continuation"])
        self.assertEqual(kwargs["generation_config"]["max_output_tokens"], 321)
        self.assertEqual(kwargs["timeout"], 7)
        self.assertNotIn("tools", kwargs)
        self.assertEqual((result.text, result.input_tokens, result.output_tokens, result.thought_tokens), ("answer", 10, 5, 2))

    def test_429_is_quota_error(self):
        error = type("RateLimitError", (Exception,), {"status_code": 429})("too many")
        with self.assertRaises(ProviderQuotaError):
            self.make(mock.Mock(side_effect=error)).generate("s", "p", 10)

    def test_errors_never_include_key(self):
        error = type("APIError", (Exception,), {"status_code": 500})("bad key AIza-secret-key")
        with self.assertRaises(ProviderError) as ctx:
            self.make(mock.Mock(side_effect=error)).generate("s", "p", 10)
        self.assertNotIn("AIza-secret-key", str(ctx.exception))

    def test_missing_key_is_refused(self):
        with self.assertRaises(ProviderError):
            GeminiProvider("", "m", client=object())


class SettingsTests(unittest.TestCase):
    def load(self, **env):
        keys = [
            "MODEL_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_TIER", "MAX_MODEL_CALLS_PER_QUESTION",
            "GEMINI_INPUT_PRICE_PER_M", "GEMINI_OUTPUT_PRICE_PER_M", "MAX_ESTIMATED_SPEND_PER_QUESTION",
        ]
        clean = {k: "" for k in keys}
        clean.update(env)
        with mock.patch.dict(os.environ, clean):
            return config.load_settings()

    def test_defaults_to_ollama(self):
        settings = self.load()
        self.assertEqual(settings.model_provider, "ollama")
        self.assertEqual(settings.max_model_calls, 3)

    def test_invalid_provider_is_an_error_not_a_fallback(self):
        settings = self.load(MODEL_PROVIDER="openai")
        self.assertTrue(any("MODEL_PROVIDER" in e for e in settings.errors))
        self.assertEqual(settings.model_provider, "openai")

    def test_gemini_needs_key_and_model(self):
        self.assertEqual(len(self.load(MODEL_PROVIDER="gemini").gemini_problems()), 2)

    def test_paid_tier_requires_prices_and_cap(self):
        settings = self.load(GEMINI_API_KEY="k", GEMINI_MODEL="m", GEMINI_TIER="paid")
        self.assertEqual(len(settings.gemini_problems()), 2)

    def test_out_of_range_limit_rejected(self):
        settings = self.load(MAX_MODEL_CALLS_PER_QUESTION="500")
        self.assertEqual(settings.max_model_calls, 3)
        self.assertTrue(settings.errors)

    def test_redact(self):
        self.assertEqual(config.redact("key=abc123", ["abc123"]), "key=[redacted]")


if __name__ == "__main__":
    unittest.main()
