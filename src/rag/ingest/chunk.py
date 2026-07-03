"""Heading-aware chunking of Wikipedia plaintext extracts.

Extracts mark sections as `== Heading ==` lines. We split by section, drop boilerplate
sections (references etc.), then pack sentences into ~chunk_tokens-sized chunks with a
sentence-aligned overlap so retrieval never lands mid-thought.
"""

import re
import uuid
from dataclasses import dataclass

DROP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "citations",
    "sources",
    "footnotes",
    "works cited",
    "gallery",
}

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # stable chunk ids


@dataclass
class Chunk:
    id: str
    title: str
    section: str
    text: str
    url: str

    @property
    def source_label(self) -> str:
        return f"{self.title} — {self.section}" if self.section != "Introduction" else self.title


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4/3 tokens per word for English prose)."""
    return max(1, round(len(text.split()) * 4 / 3))


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split extract plaintext into (top_level_section, body) pairs."""
    sections: list[tuple[str, str]] = []
    current = "Introduction"
    buf: list[str] = []
    pos = 0
    for m in _HEADING_RE.finditer(text):
        body = text[pos : m.start()].strip()
        if body:
            buf.append(body)
        if buf:
            sections.append((current, "\n\n".join(buf)))
            buf = []
        if len(m.group(1)) == 2:  # top-level section: becomes the label
            current = m.group(2)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        buf.append(tail)
    if buf:
        sections.append((current, "\n\n".join(buf)))
    return [(name, body) for name, body in sections if name.lower() not in DROP_SECTIONS]


def _pack_sentences(sentences: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    window_tokens = 0
    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if window and window_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(window))
            # keep trailing sentences as overlap for the next chunk
            kept: list[str] = []
            kept_tokens = 0
            for s in reversed(window):
                kept_tokens += estimate_tokens(s)
                kept.insert(0, s)
                if kept_tokens >= overlap_tokens:
                    break
            window, window_tokens = kept, kept_tokens
        window.append(sent)
        window_tokens += sent_tokens
    if window:
        chunks.append(" ".join(window))
    return chunks


def chunk_article(
    article: dict, max_tokens: int = 400, overlap_tokens: int = 60, min_tokens: int = 30
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(article["text"]):
        sentences = [s.strip() for s in _SENTENCE_RE.split(body) if s.strip()]
        for i, text in enumerate(_pack_sentences(sentences, max_tokens, overlap_tokens)):
            if estimate_tokens(text) < min_tokens:
                continue  # skip stubs (e.g. one-line sections)
            chunk_id = str(uuid.uuid5(_NAMESPACE, f"{article['title']}|{section}|{i}"))
            chunks.append(
                Chunk(
                    id=chunk_id,
                    title=article["title"],
                    section=section,
                    text=text,
                    url=article["url"],
                )
            )
    return chunks
