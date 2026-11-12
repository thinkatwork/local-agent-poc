from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from agent.core import run_agent

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = os.environ.get("AGENT_DOCS_DIR", str(APP_ROOT / "documents"))

st.set_page_config(page_title="Local Document Agent", page_icon="🗂️", layout="centered")
st.title("Local Document Agent")
st.caption("Ask questions about files in a folder on this Mac. The agent can search and read; it cannot edit files or run commands.")

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

prompt = st.chat_input("Ask about the documents in your folder")
if prompt:
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching your local documents…"):
            try:
                answer, api_messages, trace = run_agent(
                    question=prompt,
                    model=st.session_state.model,
                    docs_dir=st.session_state.docs_path,
                    messages=st.session_state.api_messages,
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
