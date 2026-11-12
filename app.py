from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from agent.config import tavily_api_key
from agent.core import MAX_PAGES_PER_QUESTION, MAX_SEARCHES_PER_QUESTION, run_agent

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = os.environ.get("AGENT_DOCS_DIR", str(APP_ROOT / "documents"))

TAVILY_API_KEY = tavily_api_key()

st.set_page_config(page_title="Local Research Assistant", page_icon="🔎", layout="centered")
st.title("Local Research Assistant")
st.caption(
    "Ask about files in a folder on this Mac, or research the public web. "
    "The assistant is read-only: it can search and read, but cannot edit files, run commands, sign in, submit forms or post."
)

with st.sidebar:
    st.header("Settings")
    docs_path_value = st.text_input("Documents folder", value=st.session_state.get("docs_path", DEFAULT_DOCS))
    docs_path = Path(docs_path_value).expanduser().resolve()
    docs_path.mkdir(parents=True, exist_ok=True)
    st.session_state.docs_path = str(docs_path)
    model = st.text_input("Ollama model", value=st.session_state.get("model", "qwen3:8b"))
    st.session_state.model = model.strip()

    try:
        with urlopen("http://localhost:11434/api/tags", timeout=1) as response:
            st.success("Ollama is running" if response.status == 200 else "Ollama did not respond")
    except (URLError, TimeoutError, OSError):
        st.error("Ollama is not responding. Open the Ollama app first.")

    st.subheader("Web research")
    if TAVILY_API_KEY:
        st.caption("Tavily search API key loaded from `.env`.")
    else:
        st.warning("No `TAVILY_API_KEY` found. Copy `.env.example` to `.env`, add your key, then restart the app.")
    web_enabled = st.toggle(
        "Allow web search and page reading",
        value=st.session_state.get("web_enabled", bool(TAVILY_API_KEY)) and bool(TAVILY_API_KEY),
        disabled=not TAVILY_API_KEY,
    )
    st.session_state.web_enabled = bool(web_enabled and TAVILY_API_KEY)
    st.caption(
        f"Per question: up to {MAX_SEARCHES_PER_QUESTION} searches and {MAX_PAGES_PER_QUESTION} pages. "
        "Public http/https pages only; local and private-network addresses are blocked. "
        "Search queries are sent to Tavily; each search uses 1 credit."
    )

    st.subheader("Local documents")
    st.markdown("**Supported files:** `.txt`, `.md`, `.pdf` (text-based PDFs)")
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
        st.session_state.api_messages = []
        st.session_state.display_messages = []
        st.rerun()

    st.caption("Documents are read only. Scanned PDFs need OCR before they can be searched.")

if "api_messages" not in st.session_state:
    st.session_state.api_messages = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("tools"):
            with st.expander(f"Agent used {len(item['tools'])} tool call(s)"):
                for tool in item["tools"]:
                    st.markdown(f"**{tool['name']}** · `{tool['arguments']}`")
                    st.code(tool["result"][:2500], language="text")

prompt = st.chat_input("Ask about your documents or research the web")
if prompt:
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        spinner_text = "Researching your documents and the web…" if st.session_state.web_enabled else "Searching your local documents…"
        with st.spinner(spinner_text):
            try:
                answer, api_messages, trace = run_agent(
                    question=prompt,
                    model=st.session_state.model,
                    docs_dir=st.session_state.docs_path,
                    messages=st.session_state.api_messages,
                    web_enabled=st.session_state.web_enabled,
                    api_key=TAVILY_API_KEY,
                )
                st.session_state.api_messages = api_messages
                st.markdown(answer)
                if trace:
                    with st.expander(f"Agent used {len(trace)} tool call(s)"):
                        for tool in trace:
                            st.markdown(f"**{tool['name']}** · `{tool['arguments']}`")
                            st.code(tool["result"][:2500], language="text")
                st.session_state.display_messages.append({"role": "assistant", "content": answer, "tools": trace})
            except Exception as exc:
                message = f"Could not contact the model. Check that Ollama is running and `{st.session_state.model}` is installed.\n\nDetails: `{type(exc).__name__}: {exc}`"
                st.error(message)
                st.session_state.display_messages.append({"role": "assistant", "content": message})
