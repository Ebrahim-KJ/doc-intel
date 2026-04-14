"""
document_loader.py
------------------
Loads documents from PDFs, URLs, or plain text into a
normalised list of page-level dicts ready for chunking.

Each page dict:
  { "text": str, "source": str, "page": int, "doc_id": str }
"""

import os
import uuid
import hashlib
import requests
from typing import List, Dict, Any
from pathlib import Path

# PDF parsing
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# HTML/URL parsing
from bs4 import BeautifulSoup


def _make_doc_id(source: str) -> str:
    """Stable deterministic ID from source path/URL."""
    return hashlib.md5(source.encode()).hexdigest()[:12]


def load_pdf(path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF."""
    if not HAS_FITZ:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc_id = _make_doc_id(path)
    pages  = []

    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < 20:          # Skip nearly-empty pages (cover, ToC)
                continue
            pages.append({
                "text":   text,
                "source": os.path.basename(path),
                "page":   i + 1,
                "doc_id": doc_id,
            })

    print(f"✅ Loaded PDF: {os.path.basename(path)} — {len(pages)} pages")
    return pages


def load_url(url: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """Fetch and parse a web page into a single page dict."""
    doc_id   = _make_doc_id(url)
    headers  = {"User-Agent": "Mozilla/5.0 (DocIntel Research Assistant)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup  = BeautifulSoup(response.text, "lxml")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if len(line.strip()) > 20)

    print(f"✅ Loaded URL: {url[:60]}... — {len(text)} chars")
    return [{"text": text, "source": url, "page": 1, "doc_id": doc_id}]


def load_text(text: str, source: str = "manual_input") -> List[Dict[str, Any]]:
    """Wrap raw text into a page dict."""
    doc_id = _make_doc_id(source + text[:50])
    return [{"text": text.strip(), "source": source, "page": 1, "doc_id": doc_id}]


def load_document(path_or_url: str) -> List[Dict[str, Any]]:
    """Auto-detect type and load."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return load_url(path_or_url)
    elif path_or_url.lower().endswith(".pdf"):
        return load_pdf(path_or_url)
    elif os.path.isfile(path_or_url):
        with open(path_or_url, "r", encoding="utf-8") as f:
            return load_text(f.read(), source=path_or_url)
    else:
        raise ValueError(f"Cannot load: {path_or_url}")


if __name__ == "__main__":
    # Quick smoke test
    pages = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print(f"Loaded {len(pages)} page(s), first 200 chars:")
    print(pages[0]["text"][:200])
