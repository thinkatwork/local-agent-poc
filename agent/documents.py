"""Read-only document search tools, restricted to one configured directory."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_CHARS = 18_000


def _safe_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iter_documents(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                if path.stat().st_size <= MAX_FILE_BYTES:
                    yield path
            except OSError:
                continue


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf. Run: pip install -r requirements.txt") from exc
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_number}]\n{text}")
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8", errors="replace")


def list_documents(docs_dir: str) -> str:
    """List supported documents in the configured local folder.

    Args:
        docs_dir: The configured documents folder; do not change this path.
    """
    root = _safe_root(docs_dir)
    paths = list(_iter_documents(root))
    if not paths:
        return "No supported documents were found in the configured folder."
    names = [path.relative_to(root).as_posix() for path in paths[:100]]
    result = "Available documents:\n" + "\n".join(f"- {name}" for name in names)
    if len(paths) > len(names):
        result += f"\n[Showing {len(names)} of {len(paths)} documents.]"
    return result


def search_documents(query: str, docs_dir: str) -> str:
    """Search the configured local document folder and return relevant excerpts.

    Args:
        query: Words or phrases to find in local documents.
        docs_dir: The configured documents folder; do not change this path.
    """
    root = _safe_root(docs_dir)
    terms = [term for term in re.findall(r"[\w'-]+", query.casefold()) if len(term) > 1]
    if not terms:
        return "Please provide one or more meaningful search terms."

    results: list[tuple[int, str, str]] = []
    for path in _iter_documents(root):
        try:
            text = _extract(path)
        except Exception:
            continue
        folded = text.casefold()
        score = sum(folded.count(term) for term in terms)
        if not score:
            continue
        snippets = []
        for term in terms:
            start = 0
            while len(snippets) < 3:
                position = folded.find(term, start)
                if position < 0:
                    break
                left, right = max(0, position - 220), min(len(text), position + 320)
                snippet = " ".join(text[left:right].split())
                if snippet and snippet not in snippets:
                    snippets.append(snippet)
                start = position + len(term)
        rel = path.relative_to(root).as_posix()
        results.append((score, rel, "\n  • ".join(snippets)))

    if not results:
        return "No matching passages found in the configured documents folder."
    results.sort(key=lambda item: (-item[0], item[1].casefold()))
    output = [f"Search results for: {query}"]
    for score, filename, excerpts in results[:5]:
        output.append(f"\nSource: {filename} (match score {score})\n  • {excerpts}")
    return "\n".join(output)


def read_document(filename: str, docs_dir: str) -> str:
    """Read one named document from the configured folder, never outside it.

    Args:
        filename: A relative file name returned by search_documents.
        docs_dir: The configured documents folder; do not change this path.
    """
    root = _safe_root(docs_dir)
    candidate = (root / filename).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return "Document not found in the configured folder. Use search_documents first."
    if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
        return "That file type is not supported."
    if candidate.stat().st_size > MAX_FILE_BYTES:
        return "That document exceeds the 5 MB POC limit."
    try:
        content = _extract(candidate)
    except Exception as exc:
        return f"Could not read document: {type(exc).__name__}: {exc}"
    rel = candidate.relative_to(root).as_posix()
    if len(content) > MAX_DOCUMENT_CHARS:
        content = content[:MAX_DOCUMENT_CHARS] + "\n[Document truncated for this POC.]"
    return f"Source: {rel}\n\n{content or '[No extractable text found.]'}"
