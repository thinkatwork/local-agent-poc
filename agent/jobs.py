"""Run one question in a background thread so the UI can cancel it.

Cancelling sets an Event. The workflow checks it before every search, page
fetch and model call, and the Ollama provider checks it while a reply streams
in, so nothing new is started after you cancel. A request that is already in
flight to a cloud provider (Gemini) cannot be recalled: it may still finish and
be billed, and its result is discarded.

The worker never calls Streamlit; the UI thread reads `step`, `result` and
`error` from this object.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class Cancelled(Exception):
    """The user cancelled the question."""


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled by you.")


@dataclass
class BackgroundJob:
    cancel: threading.Event = field(default_factory=threading.Event)
    step: str = "Starting…"
    result: Any = None
    error: BaseException | None = None
    started: float = field(default_factory=time.monotonic)
    thread: threading.Thread | None = None

    @property
    def done(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def set_step(self, step: str) -> None:
        self.step = step


def start_job(work: Callable[[threading.Event, Callable[[str], None]], Any]) -> BackgroundJob:
    """Start `work(cancel_event, report_step)` in a daemon thread."""
    job = BackgroundJob()

    def run() -> None:
        try:
            job.result = work(job.cancel, job.set_step)
        except BaseException as exc:  # reported to the UI, never re-raised in the thread
            job.error = exc

    job.thread = threading.Thread(target=run, name="research-question", daemon=True)
    job.thread.start()
    return job
