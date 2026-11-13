from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from agent.config import load_settings, redact, secrets
from agent.core import run_agent
from agent.jobs import BackgroundJob, Cancelled, start_job
from agent.providers import GeminiProvider, OllamaProvider, TavilySearch
from agent.workflow import Limits, Pricing, run_research

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = os.environ.get("AGENT_DOCS_DIR", str(APP_ROOT / "documents"))

SETTINGS = load_settings()
SECRETS = secrets(SETTINGS)
LIMITS = Limits(
    max_model_calls=SETTINGS.max_model_calls,
    max_searches=SETTINGS.max_searches,
    max_fetches=SETTINGS.max_fetches,
    max_output_tokens=SETTINGS.max_output_tokens,
    max_input_chars=SETTINGS.max_input_chars,
)
PROVIDER_LABELS = {"ollama": "Ollama (local, on this Mac)", "gemini": "Gemini (Google cloud)"}
MODE_WEB, MODE_DOCS = "Web research", "Local documents"
CANCEL_GRACE_SECONDS = 3  # how long the UI waits for a cancelled question to wind down before moving on

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="centered")
st.title("Research Assistant")
st.caption(
    "A user-directed, read-only research assistant. The UI and app run on this Mac; depending on your settings, "
    "the model may run locally (Ollama) or at Google (Gemini). It can search and read, but cannot edit files, "
    "run commands, sign in, submit forms or post. Its multi-step review reduces, but does not remove, mistakes."
)


def show_trace(trace: list[dict[str, str]]) -> None:
    with st.expander(f"Workflow trace: {len(trace)} step(s)"):
        for step in trace:
            st.markdown(f"**{step['name']}** · `{redact(step['arguments'], SECRETS)}`")
            st.code(redact(step["result"], SECRETS)[:2500], language="text")


def show_usage(usage: dict, provider: str) -> None:
    parts = [f"Model calls: {usage['model_calls']}"]
    tokens = f"tokens in/out/thinking: {usage['input_tokens']:,} / {usage['output_tokens']:,} / {usage['thought_tokens']:,}"
    parts.append(tokens + ("" if usage["tokens_reported"] else " (some counts estimated; provider did not report them)"))
    parts.append(f"Tavily searches: {usage['searches']} (basic search, 1 credit each)")
    parts.append(f"pages fetched by the app: {usage['fetches']}")
    if provider == "gemini":
        if usage.get("estimated_cost") is None:
            parts.append("Gemini cost: cannot be estimated (set GEMINI_*_PRICE_PER_M in .env)")
        else:
            label = "if billed at paid-tier rates" if SETTINGS.gemini_tier == "free" else "estimate"
            parts.append(f"Gemini cost {label}: ≈ ${usage['estimated_cost']:.5f} (estimate, not a bill)")
    else:
        parts.append("model cost: none (local)")
    st.caption(" · ".join(parts))


with st.sidebar:
    st.header("Settings")
    for problem in SETTINGS.errors:
        st.error(problem)

    provider_options = list(PROVIDER_LABELS)
    default_provider = SETTINGS.model_provider if SETTINGS.model_provider in provider_options else "ollama"
    provider = st.radio(
        "Model provider",
        provider_options,
        index=provider_options.index(st.session_state.get("provider", default_provider)),
        format_func=PROVIDER_LABELS.get,
        help="Default comes from MODEL_PROVIDER in .env. The app never switches provider by itself.",
    )
    st.session_state.provider = provider
    provider_ready = SETTINGS.model_provider in provider_options

    if provider == "ollama":
        st.caption(f"Model: `{SETTINGS.ollama_model}` (OLLAMA_MODEL in .env)")
        try:
            with urlopen("http://localhost:11434/api/tags", timeout=1) as response:
                st.success("Ollama is running" if response.status == 200 else "Ollama did not respond")
        except (URLError, TimeoutError, OSError):
            st.error("Ollama is not responding. Open the Ollama app first.")
    else:
        st.caption(f"Model: `{SETTINGS.gemini_model or 'not set'}` (GEMINI_MODEL) · tier setting: {SETTINGS.gemini_tier}")
        problems = SETTINGS.gemini_problems()
        for problem in problems:
            st.error(problem)
        if problems:
            st.info("Fix .env and restart, or switch back to Ollama. The app will not fall back to another model.")
        provider_ready = provider_ready and not problems

    st.subheader("Mode")
    mode = st.radio("What should the assistant use?", [MODE_WEB, MODE_DOCS], index=[MODE_WEB, MODE_DOCS].index(st.session_state.get("mode", MODE_WEB)))
    st.session_state.mode = mode
    mode_ready = True
    if mode == MODE_WEB:
        if SETTINGS.tavily_api_key:
            st.caption("Tavily search key loaded from `.env`.")
        else:
            st.warning("Web research is off: no `TAVILY_API_KEY` in `.env`. Local document chat with Ollama still works.")
            mode_ready = False
        review = st.toggle("Reviewer step (one extra model call)", value=st.session_state.get("review", True))
        st.session_state.review = review
        calls = 2 + int(review)
        plan_note = " plus a query-planning call" if LIMITS.max_model_calls > calls and LIMITS.max_searches > 1 else ""
        st.caption(
            f"Per question: {calls} model calls{plan_note} (max {LIMITS.max_model_calls}), up to {LIMITS.max_searches} "
            f"searches, {LIMITS.max_fetches} pages, {LIMITS.max_output_tokens} output tokens per call. "
            "Public http/https pages only; local and private-network addresses are blocked."
        )
    else:
        if provider == "gemini":
            st.warning("Local documents are only available with Ollama, so their contents never leave this Mac.")
            mode_ready = False

    st.subheader("Local documents")
    docs_path_value = st.text_input("Documents folder", value=st.session_state.get("docs_path", DEFAULT_DOCS))
    docs_path = Path(docs_path_value).expanduser().resolve()
    docs_path.mkdir(parents=True, exist_ok=True)
    st.session_state.docs_path = str(docs_path)
    st.markdown("**Supported files:** `.txt`, `.md`, `.pdf` (text-based PDFs), up to 5 MB each")
    uploads = st.file_uploader(
        "Add documents to this folder",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        max_upload_size=5,
    )
    if uploads:
        added_or_changed = False
        for upload in uploads:
            if upload.size > 5 * 1024 * 1024:
                st.warning(f"Skipped {upload.name}: each file must be under 5 MB.")
                continue
            clean_name = Path(upload.name).name
            destination = docs_path / clean_name
            digest = hashlib.sha256(upload.getvalue()).hexdigest()
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
                continue
            destination.write_bytes(upload.getvalue())
            added_or_changed = True
        if added_or_changed:
            st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        if st.session_state.get("active"):
            st.session_state.active["job"].cancel.set()
            st.session_state.active = None
        st.session_state.api_messages = []
        st.session_state.display_messages = []
        st.session_state.research_history = []
        st.rerun()

    st.caption("Documents are read only. Scanned PDFs need OCR before they can be searched.")

for key in ("api_messages", "display_messages", "research_history"):
    if key not in st.session_state:
        st.session_state[key] = []

for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("tools"):
            show_trace(item["tools"])
        if item.get("usage"):
            show_usage(item["usage"], item.get("provider", "ollama"))

if provider == "gemini" and mode == MODE_WEB:
    tier_note = (
        " Free-tier setting: Google may use free-tier content to improve its products, so do not ask about personal, "
        "confidential, client, work or family matters."
        if SETTINGS.gemini_tier == "free"
        else " Paid-tier setting: token charges apply; see the usage line under each answer."
    )
    st.warning(
        f"**Gemini cloud mode** (`{SETTINGS.gemini_model}`): your question and the retrieved web excerpts are sent to "
        f"Google. Local documents are not sent. Search queries go to Tavily.{tier_note}"
    )
elif mode == MODE_WEB:
    st.info("Ollama mode: the model runs on this Mac. Search queries still go to Tavily and pages are fetched from the web.")

ready = provider_ready and mode_ready
active = st.session_state.get("active")
prompt = st.chat_input(
    "Ask a research question" if mode == MODE_WEB else "Ask about your documents",
    disabled=not ready or active is not None,
)


def make_work(question: str, question_mode: str, question_provider: str):
    """Build the function the background thread runs. It must not call Streamlit."""
    docs_dir = st.session_state.docs_path
    api_messages = list(st.session_state.api_messages)
    history = list(st.session_state.research_history)
    review = st.session_state.review

    def work(cancel, report):
        if question_mode == MODE_DOCS:
            return run_agent(
                question=question,
                model=SETTINGS.ollama_model,
                docs_dir=docs_dir,
                messages=api_messages,
                web_enabled=False,
                cancel=cancel,
                on_step=report,
            )
        if question_provider == "gemini":
            model = GeminiProvider(
                SETTINGS.gemini_api_key or "",
                SETTINGS.gemini_model,
                timeout_seconds=SETTINGS.request_timeout_seconds,
                tier=SETTINGS.gemini_tier,
            )
        else:
            model = OllamaProvider(SETTINGS.ollama_model, cancel=cancel)
        pricing = Pricing(
            SETTINGS.gemini_input_price_per_m,
            SETTINGS.gemini_output_price_per_m,
            SETTINGS.max_estimated_spend_per_question,
        )
        return run_research(
            question=question,
            model=model,
            search=TavilySearch(SETTINGS.tavily_api_key or ""),
            limits=LIMITS,
            pricing=pricing,
            review=review,
            history=history,
            cancel=cancel,
            on_step=report,
        )

    return work


def cancel_active() -> None:
    if st.session_state.get("active"):
        st.session_state.active["job"].cancel.set()


def finish_active(item: dict) -> None:
    """Turn a finished (or abandoned) background job into a chat message."""
    job: BackgroundJob = item["job"]
    question, question_mode, question_provider = item["question"], item["mode"], item["provider"]
    messages = st.session_state.display_messages
    if not job.done:
        # Cancelled, but a request already in flight has not returned yet. Abandon it: the thread
        # stops at its next check, starts nothing new, and its result is discarded.
        if question_mode == MODE_DOCS:
            note = (
                "No further model or document tool calls will be made. The local model may finish its current "
                "reply in the background on this Mac; the result is discarded."
            )
        else:
            note = "No further searches, page fetches or model calls will be made."
        if question_provider == "gemini" and question_mode == MODE_WEB:
            note += (
                " A Gemini request that was already in progress cannot be recalled: it may still finish at Google "
                "and be billed, and its result will be discarded."
            )
        messages.append({"role": "assistant", "content": f"**Cancelled by you.** {note}"})
    elif isinstance(job.error, Cancelled):
        messages.append(
            {"role": "assistant", "content": "**Cancelled by you.** No further model calls or tool calls were made."}
        )
    elif job.error is not None:
        exc = job.error
        message = redact(
            f"The request failed and was not retried.\n\nDetails: `{type(exc).__name__}: {str(exc)[:300]}`", SECRETS
        )
        if question_provider == "ollama" or question_mode == MODE_DOCS:
            message += f"\n\nCheck that Ollama is running and `{SETTINGS.ollama_model}` is installed."
        messages.append({"role": "assistant", "content": message})
    elif question_mode == MODE_DOCS:
        answer, api_messages, trace = job.result
        st.session_state.api_messages = api_messages
        messages.append({"role": "assistant", "content": redact(answer, SECRETS), "tools": trace})
    else:
        result = job.result
        answer = redact(result.answer, SECRETS)
        if not result.cancelled:
            st.session_state.research_history.append({"question": question, "answer": answer})
        messages.append(
            {"role": "assistant", "content": answer, "tools": result.trace, "usage": result.usage, "provider": question_provider}
        )
    st.session_state.active = None


if prompt and active is None:
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.active = {
        "job": start_job(make_work(prompt, mode, provider)),
        "question": prompt,
        "mode": mode,
        "provider": provider,
    }
    st.rerun()

if active is not None:
    job: BackgroundJob = active["job"]
    with st.chat_message("assistant"):
        status = st.empty()
        st.button(
            "Cancelling…" if job.cancelled else "Cancel",
            on_click=cancel_active,
            disabled=job.cancelled,
            key="cancel_active_question",
            icon=":material/stop_circle:",
            help="Stop this question. Nothing new is searched, fetched or sent to the model after you cancel.",
        )
        cancel_deadline = None
        while not job.done:
            elapsed = int(time.monotonic() - job.started)
            if job.cancelled:
                cancel_deadline = cancel_deadline or time.monotonic() + CANCEL_GRACE_SECONDS
                if time.monotonic() > cancel_deadline:
                    break
                status.warning("Cancelling: waiting for the current step to stop…")
            else:
                status.info(f"{job.step}… ({elapsed}s)")
            time.sleep(0.25)
    finish_active(active)
    st.rerun()
