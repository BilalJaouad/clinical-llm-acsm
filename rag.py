"""
rag.py
------
Minimal, economical RAG (section 9, Phase 2).

Deliberately NOT using an embedding model or a vector store: the corpus is
tiny (a handful of .txt files, section 10), so a simple keyword-overlap
score over paragraphs is enough and keeps the project dependency-free and
GPU-free (criterion in section 22: "runs without a local GPU").

Only 1-3 short passages are ever returned — never full documents.
"""

import os
import re
from functools import lru_cache

DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")


def _tokenize(text: str):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


@lru_cache(maxsize=1)
def _load_paragraphs():
    """
    Loads every .txt file in data/documents and splits it into paragraphs.
    Cached so the small corpus is only read from disk once per session.
    Returns a list of (source_filename, paragraph_text).
    """
    paragraphs = []
    if not os.path.isdir(DOCS_DIR):
        return paragraphs

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for para in content.split("\n\n"):
            para = para.strip()
            if len(para) > 30:  # skip empty/near-empty fragments
                paragraphs.append((filename, para))
    return paragraphs


def get_relevant_passages(query: str, top_k: int = 2, min_overlap: int = 1):
    """
    Returns up to `top_k` (source, passage) tuples ranked by simple word
    overlap with the query. Returns an empty list if nothing overlaps
    meaningfully, so the caller can fall back to no-RAG behaviour.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored = []
    for source, paragraph in _load_paragraphs():
        para_words = _tokenize(paragraph)
        overlap = len(query_words & para_words)
        if overlap >= min_overlap:
            scored.append((overlap, source, paragraph))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [(source, paragraph) for _, source, paragraph in top]
