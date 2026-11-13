"""Offline tests for cancelling a question.

Run with:  python -m unittest discover -s tests
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import core
from agent.jobs import Cancelled, start_job
from agent.providers import OllamaProvider
from agent.workflow import Limits, run_research
from tests.test_workflow import FakeModel, FakeSearch, fake_fetch


class CancelWorkflowTests(unittest.TestCase):
    def test_cancel_before_start_makes_no_calls(self):
        cancel = threading.Event()
        cancel.set()
        model, search, fetched = FakeModel(), FakeSearch(), []
        result = run_research("q", model, search, Limits(), cancel=cancel, fetcher=fake_fetch(fetched))
        self.assertTrue(result.cancelled)
        self.assertEqual((model.prompts, search.queries, fetched), ([], [], []))
        self.assertIn("Cancelled by you", result.answer)

    def test_cancel_during_researcher_stops_later_calls(self):
        cancel = threading.Event()
        model, fetched = FakeModel(), []
        original = model.generate

        def generate_then_cancel(*args):
            result = original(*args)
            cancel.set()  # user presses Cancel while the researcher call is in flight
            return result

        model.generate = generate_then_cancel
        result = run_research("q", model, FakeSearch(), Limits(), review=True, cancel=cancel, fetcher=fake_fetch(fetched))
        self.assertTrue(result.cancelled)
        self.assertEqual(len(model.prompts), 1)  # reviewer and writer never sent
        self.assertEqual(result.usage["model_calls"], 1)  # the in-flight call is still counted
        self.assertIn("Researcher notes gathered before stopping", result.answer)

    def test_cancel_during_fetching_stops_fetches(self):
        cancel = threading.Event()
        fetched = []

        def fetch(url):
            fetched.append(url)
            cancel.set()
            return "page", {"title": "T", "url": url, "date": ""}

        model = FakeModel()
        result = run_research("q", model, FakeSearch(), Limits(max_fetches=4), cancel=cancel, fetcher=fetch)
        self.assertTrue(result.cancelled)
        self.assertEqual(len(fetched), 1)
        self.assertEqual(model.prompts, [])

    def test_progress_steps_are_reported(self):
        steps = []
        run_research("q", FakeModel(), FakeSearch(), Limits(), review=True, fetcher=fake_fetch([]), on_step=steps.append)
        self.assertTrue(steps[0].startswith("Searching the web"))
        self.assertTrue(any(s.startswith("Reading page") for s in steps))
        self.assertTrue(steps[-1].startswith("Writer"))


def chunk(text, done=False):
    extra = {"prompt_eval_count": 7, "eval_count": 3, "done_reason": "stop"} if done else {}
    return SimpleNamespace(message=SimpleNamespace(content=text), **extra)


class FakeStream:
    def __init__(self, chunks, on_chunk=None):
        self.chunks, self.on_chunk, self.closed, self.read = chunks, on_chunk, False, 0

    def __iter__(self):
        for item in self.chunks:
            self.read += 1
            if self.on_chunk:
                self.on_chunk(self.read)
            yield item

    def close(self):
        self.closed = True


class OllamaCancelTests(unittest.TestCase):
    def test_streams_and_reports_usage(self):
        stream = FakeStream([chunk("Hel"), chunk("lo", done=True)])
        chat = mock.Mock(return_value=stream)
        result = OllamaProvider("m", chat_fn=chat).generate("s", "p", 50)
        self.assertTrue(chat.call_args.kwargs["stream"])
        self.assertEqual((result.text, result.input_tokens, result.output_tokens), ("Hello", 7, 3))
        self.assertTrue(stream.closed)

    def test_cancel_mid_stream_closes_connection(self):
        cancel = threading.Event()
        stream = FakeStream([chunk("a"), chunk("b"), chunk("c"), chunk("d", done=True)],
                            on_chunk=lambda n: cancel.set() if n == 2 else None)
        with self.assertRaises(Cancelled):
            OllamaProvider("m", chat_fn=mock.Mock(return_value=stream), cancel=cancel).generate("s", "p", 50)
        self.assertTrue(stream.closed)
        self.assertLess(stream.read, 4)

    def test_cancelled_before_request_sends_nothing(self):
        cancel = threading.Event()
        cancel.set()
        chat = mock.Mock()
        with self.assertRaises(Cancelled):
            OllamaProvider("m", chat_fn=chat, cancel=cancel).generate("s", "p", 50)
        chat.assert_not_called()


class DocumentsAgentCancelTests(unittest.TestCase):
    def test_cancel_raises_before_model_call(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(core, "chat") as chat, self.assertRaises(Cancelled):
            core.run_agent("q", "m", "/tmp/docs-test", cancel=cancel)
        chat.assert_not_called()


class BackgroundJobTests(unittest.TestCase):
    def test_result_and_error_are_captured(self):
        ok = start_job(lambda cancel, report: (report("working"), 42)[1])
        ok.thread.join(2)
        self.assertTrue(ok.done)
        self.assertEqual((ok.result, ok.step, ok.error), (42, "working", None))

        def fail(cancel, report):
            raise ValueError("boom")

        bad = start_job(fail)
        bad.thread.join(2)
        self.assertIsInstance(bad.error, ValueError)

    def test_cancel_flag_reaches_worker(self):
        started = threading.Event()

        def work(cancel, report):
            started.set()
            if not cancel.wait(2):
                return "not cancelled"
            raise Cancelled("Cancelled by you.")

        job = start_job(work)
        started.wait(2)
        job.cancel.set()
        job.thread.join(2)
        self.assertTrue(job.cancelled)
        self.assertIsInstance(job.error, Cancelled)


if __name__ == "__main__":
    unittest.main()
