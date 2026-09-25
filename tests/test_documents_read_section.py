"""Tests for DocumentManager.read(path, section=...) section retrieval."""

from __future__ import annotations

import threading

import pytest

from markdown_vault_mcp.fts_index import FTSIndex
from markdown_vault_mcp.managers.document import DocumentManager
from markdown_vault_mcp.scanner import HeadingChunker, scan_directory


@pytest.fixture()
def doc_mgr(tmp_path):
    a = tmp_path / "a.md"
    # Use multi-line bodies so the doc clears the 30-line short-doc bypass and
    # actually splits into per-section chunks.
    body = (
        "# A\n"
        + "\n".join(["intro"] * 12)
        + "\n## Section One\n"
        + "\n".join(["first body word"] * 12)
        + "\n## Section Two\n"
        + "\n".join(["second body word"] * 12)
        + "\n"
    )
    a.write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker()
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    return DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
        read_only=False,
    )


def test_read_no_section_returns_full_file(doc_mgr):
    nc = doc_mgr.read("a.md")
    assert nc is not None
    assert "Section One" in nc.content
    assert "Section Two" in nc.content


def test_read_with_section_returns_only_that_chunk(doc_mgr):
    nc = doc_mgr.read("a.md", section="Section One")
    assert nc is not None
    assert "first body word" in nc.content
    assert "second body word" not in nc.content


def test_read_unknown_section_raises(doc_mgr):
    with pytest.raises(ValueError, match="Section"):
        doc_mgr.read("a.md", section="No Such Heading")


def test_read_empty_section_raises(doc_mgr):
    with pytest.raises(ValueError):
        doc_mgr.read("a.md", section="   ")


def test_read_returns_none_when_path_unknown(doc_mgr):
    assert doc_mgr.read("missing.md") is None
    # With section, missing path also raises (cannot resolve section in
    # nonexistent doc).
    with pytest.raises(ValueError):
        doc_mgr.read("missing.md", section="Anything")


def test_read_section_collapses_internal_whitespace(tmp_path):
    """Lookup tolerates whitespace runs differing from storage."""
    a = tmp_path / "a.md"
    # Stored heading has two spaces after the numbering prefix — the kind of
    # editor artefact LLM callers rarely reproduce from a rendered TOC.
    # Doc must clear the 30-line short-doc bypass to get per-section chunks.
    body = (
        "# A\n"
        + "\n".join(["intro"] * 16)
        + "\n## 1.3.  Reducing excessive dependencies\n"
        + "\n".join(["body content"] * 16)
        + "\n"
    )
    a.write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker()
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    mgr = DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
    )

    # Two-spaces stored, one-space lookup — the production failure shape.
    nc = mgr.read("a.md", section="1.3. Reducing excessive dependencies")
    assert nc is not None
    assert "body content" in nc.content
    # Symmetric: two-spaces stored, three-spaces lookup also collapses.
    nc = mgr.read("a.md", section="1.3.   Reducing excessive dependencies")
    assert nc is not None
    assert "body content" in nc.content


def test_read_unknown_section_lists_available_headings(doc_mgr):
    """Miss message includes the actual stored headings so callers can recover."""
    with pytest.raises(ValueError) as excinfo:
        doc_mgr.read("a.md", section="No Such Heading")
    message = str(excinfo.value)
    assert "No Such Heading" in message
    assert "available headings include" in message
    assert "'Section One'" in message
    assert "'Section Two'" in message


def test_read_section_no_headings_message(tmp_path):
    """When the document has no indexed headings, the miss message says so."""
    a = tmp_path / "a.md"
    # Long body with no markdown headings — clears short-doc bypass.
    a.write_text("\n".join(["plain text line"] * 60) + "\n", encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker()
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    mgr = DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
    )

    with pytest.raises(ValueError) as excinfo:
        mgr.read("a.md", section="Anything")
    assert "document has no indexed headings" in str(excinfo.value)


def test_read_section_duplicate_heading_returns_first_by_start_line(tmp_path):
    """When a heading repeats, _read_section returns the first occurrence."""
    a = tmp_path / "a.md"
    # Each section body must be long enough that the total doc exceeds the
    # 30-line short-doc bypass (HeadingChunker default), so we get per-section
    # chunks rather than a single whole-document chunk.
    body = (
        "# A\n## Repeat\n"
        + "\n".join(["first occurrence body"] * 16)
        + "\n## Repeat\n"
        + "\n".join(["second occurrence body"] * 16)
        + "\n"
    )
    a.write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker()
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    mgr = DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
    )

    nc = mgr.read("a.md", section="Repeat")
    assert nc is not None
    assert "first occurrence body" in nc.content
    assert "second occurrence body" not in nc.content


# ---------------------------------------------------------------------------
# AGO-560 (2026-09-25): a long section is several chunks; read(section=) serves them all
# ---------------------------------------------------------------------------


def _long_doc_mgr(tmp_path, *, max_note_read_bytes: int = 0):
    a = tmp_path / "long.md"
    long_words = "\n".join(f"w{i}" for i in range(400))  # 400 words on 400 lines: clears the short-doc bypass, splits at max_chunk_words=50
    body = (
        "# Long\n" + "\n".join(["intro"] * 12) + "\n## Big Section\n" + long_words + "\nLAST-WORD-OF-SECTION\n"
        "## After\n" + "\n".join(["after body"] * 12) + "\n"
    )
    a.write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker(max_chunk_words=50)
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    kwargs = {}
    if max_note_read_bytes:
        kwargs["max_note_read_bytes"] = max_note_read_bytes
    return fts, DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
        read_only=False,
        **kwargs,
    )


def test_long_section_is_stored_as_several_chunks(tmp_path):
    fts, _ = _long_doc_mgr(tmp_path)
    chunks = fts.get_section_chunks("long.md", "Big Section")
    assert len(chunks) > 1, "the fixture must split the section, or the test proves nothing"
    assert fts.get_section("long.md", "Big Section")["content"] == chunks[0]["content"]


def test_read_section_returns_the_whole_section_not_its_first_chunk(tmp_path):
    _, mgr = _long_doc_mgr(tmp_path)
    nc = mgr.read("long.md", section="Big Section")
    assert nc is not None
    assert "w0 " in nc.content or nc.content.startswith("w0")
    assert "LAST-WORD-OF-SECTION" in nc.content, "the section's tail was cut silently (AGO-560)"
    assert "after body" not in nc.content
    assert "continues:" not in nc.content


def test_read_section_announces_a_cut_when_the_cap_binds(tmp_path):
    _, mgr = _long_doc_mgr(tmp_path, max_note_read_bytes=600)
    nc = mgr.read("long.md", section="Big Section")
    assert nc is not None
    assert "LAST-WORD-OF-SECTION" not in nc.content
    assert "[section 'Big Section' continues:" in nc.content
    assert "read('long.md')" in nc.content
