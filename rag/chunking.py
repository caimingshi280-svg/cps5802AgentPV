"""Document chunking for RAG.

Strategy (MVP, rule §27 minimal viable):

* **Markdown-aware**: split first by Markdown headings (so each ``## ``
  section becomes its own chunk root), then secondarily by paragraph
  boundary if a section grows past ``max_chars``.
* **Title carry-forward**: the document's title (first ``# `` heading)
  prefixes every chunk so retrieval results carry source context.

The polish-phase upgrade can swap in semantic chunking
(LangChain's RecursiveCharacterTextSplitter) without changing the
:class:`Chunk` shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text."""

    text: str
    source: str  # filename or document id
    title: str
    section: str | None  # optional section heading


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_by_heading(content: str) -> list[tuple[str, str | None]]:
    """Return [(section_heading_or_None, body), ...] in document order."""

    matches = list(_HEADING_RE.finditer(content))
    if not matches:
        return [(None, content.strip())]

    sections: list[tuple[str, str | None]] = []
    pre_text = content[: matches[0].start()].strip()
    if pre_text:
        sections.append((pre_text, None))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        # 把 heading + body 作为单块的 text；保留 heading 用于 section 元数据。
        full = f"{heading}\n\n{body}".strip() if body else heading
        sections.append((full, heading))
    return sections


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split overly long text by paragraph until each fragment ≤ max_chars."""

    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current.strip())
            current = paragraph
    if current:
        parts.append(current.strip())
    return parts


def chunk_markdown_document(
    content: str,
    source: str,
    *,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Chunk a Markdown document.

    Parameters
    ----------
    content : str
        The full Markdown text.
    source : str
        Document identifier (typically the filename).
    max_chars : int
        Target maximum length for any chunk's body. Sections longer than
        this are further split by paragraph.
    """

    if not content.strip():
        return []

    title = "untitled"
    title_match = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    sections = _split_by_heading(content)
    chunks: list[Chunk] = []
    for body, section_heading in sections:
        for piece in _split_long(body, max_chars=max_chars):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    title=title,
                    section=section_heading,
                )
            )
    return chunks


def chunk_directory(
    documents_dir: Path,
    *,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Chunk every ``*.md`` file in a directory, returning all chunks."""

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    chunks: list[Chunk] = []
    for path in sorted(documents_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown_document(text, source=path.name, max_chars=max_chars))
    return chunks
