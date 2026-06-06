"""Unit tests for :mod:`rag.chunking`."""
from __future__ import annotations

from rag.chunking import Chunk, chunk_directory, chunk_markdown_document


def test_empty_document_returns_no_chunks() -> None:
    assert chunk_markdown_document("", source="x.md") == []
    assert chunk_markdown_document("   \n  \n", source="x.md") == []


def test_single_section_picks_up_title() -> None:
    md = "# My Title\n\nBody paragraph here."
    chunks = chunk_markdown_document(md, source="t.md")
    assert len(chunks) == 1
    assert chunks[0].title == "My Title"
    assert "Body paragraph here." in chunks[0].text
    assert chunks[0].section == "My Title"


def test_multiple_sections_split_by_heading() -> None:
    md = (
        "# Doc\n\n"
        "## Section A\nContent A\n\n"
        "## Section B\nContent B\n\n"
        "## Section C\nContent C\n"
    )
    chunks = chunk_markdown_document(md, source="d.md")
    headings = [c.section for c in chunks]
    assert "Doc" in headings  # title 也是一级 heading
    assert "Section A" in headings
    assert "Section B" in headings
    assert "Section C" in headings


def test_long_section_split_by_paragraph() -> None:
    long_para_a = "Para one. " * 80  # ~880 chars
    long_para_b = "Para two. " * 80
    long_para_c = "Para three. " * 80
    md = f"# T\n\n## S\n{long_para_a}\n\n{long_para_b}\n\n{long_para_c}"
    chunks = chunk_markdown_document(md, source="long.md", max_chars=500)
    assert len(chunks) >= 2  # 长 section 必须被切成多块
    for c in chunks:
        assert c.title == "T"


def test_chunk_directory_loads_all_md_files(tmp_path) -> None:
    (tmp_path / "a.md").write_text("# A\nbody a", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\nbody b", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("ignore me", encoding="utf-8")
    chunks = chunk_directory(tmp_path)
    titles = {c.title for c in chunks}
    assert titles == {"A", "B"}
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.md"}


def test_chunk_directory_missing_dir_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        chunk_directory(tmp_path / "does_not_exist")


def test_chunk_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    c = Chunk(text="x", source="y.md", title="Y", section=None)
    with pytest.raises(FrozenInstanceError):
        c.text = "z"  # type: ignore[misc]
