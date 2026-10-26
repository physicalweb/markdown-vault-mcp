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
    assert "all 3 headings in this document" in message
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


def _mgr_for(tmp_path, body):
    (tmp_path / "a.md").write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker()
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    return DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
    )


def test_read_unknown_section_lists_every_heading_past_ten(tmp_path):
    """AGO-188: the miss listing names ALL headings with their count — a cut at ten read as the whole doc."""
    body = "# Title\n" + "\n".join(["intro"] * 12) + "\n"
    for i in range(12):
        body += f"## {i}. Part {i}\n" + "\n".join([f"body {i}"] * 6) + "\n"
    mgr = _mgr_for(tmp_path, body)
    with pytest.raises(ValueError) as excinfo:
        mgr.read("a.md", section="No Such Heading")
    message = str(excinfo.value)
    assert "all 13 headings in this document" in message
    assert "'10. Part 10'" in message
    assert "'11. Part 11'" in message
    assert "include" not in message


def test_read_section_accepts_heading_as_written(tmp_path):
    """AGO-188: "## 7. Title" (markdown as written) and a closing "##" resolve to the same section."""
    body = (
        "# A\n"
        + "\n".join(["intro"] * 16)
        + "\n## 7. What we do not do\n"
        + "\n".join(["seventh body"] * 16)
        + "\n"
    )
    mgr = _mgr_for(tmp_path, body)
    for heading in ("## 7. What we do not do", "##   7. What we do not do ##", "# 7. What we do not do"):
        nc = mgr.read("a.md", section=heading)
        assert nc is not None
        assert "seventh body" in nc.content


def test_read_section_literal_hash_heading_still_exact(tmp_path):
    """A stored heading that itself begins with "# " still matches exactly (stripping is only the fallback)."""
    body = (
        "# A\n"
        + "\n".join(["intro"] * 16)
        + "\n## # hashed\n"
        + "\n".join(["hashed body"] * 16)
        + "\n"
    )
    mgr = _mgr_for(tmp_path, body)
    nc = mgr.read("a.md", section="# hashed")
    assert nc is not None
    assert "hashed body" in nc.content


# ---------------------------------------------------------------------------
# AGO-826 (2026-10-26): a section read in a file WITH frontmatter dropped its continuation chunks
# silently — the continuation test read the raw file (frontmatter included) at a body-relative
# start_line, and _budget_split counted fragments one line short of the heading chunk's content.
# ---------------------------------------------------------------------------


def _frontmatter_list_doc(tmp_path, *, fm_keys: int = 3, max_chunk_words: int = 60):
    """A file shaped like the reported one: frontmatter, numbered sections, a long numbered list whose
    last item carries indented continuation paragraphs, a closing line, then the next section."""
    items = "\n\n".join(
        f"{n}. Item {n} " + " ".join(f"alpha{n}x{k}" for k in range(14)) + "." for n in range(1, 6)
    )
    body = (
        "---\n"
        + "".join(f"key{k}: value {k}\n" for k in range(fm_keys))
        + "---\n"
        "\n"
        "# Interim report\n\n"
        + "\n".join(["preface line"] * 12)
        + "\n\n## 5. Earlier\n\n"
        + "\n".join(["earlier body"] * 8)
        + "\n\n## 6. What comes next (not decided)\n\n"
        + items
        + "\n\n6. Item six opens here and waits with the rest for the checkpoint.\n\n"
        "   *First indented* paragraph " + " ".join(f"beta{k}" for k in range(20)) + ".\n\n"
        "   *Second indented* paragraph " + " ".join(f"gamma{k}" for k in range(20)) + ".\n\n"
        "   *Third indented* paragraph " + " ".join(f"delta{k}" for k in range(20)) + ".\n\n"
        "CLOSING-LINE-OF-SIX, if the checkpoint suggests it.\n\n"
        "## 7. Later\n\n"
        + "\n".join(["later body"] * 8)
        + "\n"
    )
    (tmp_path / "r.md").write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker(max_chunk_words=max_chunk_words)
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    return fts, DocumentManager(
        fts=fts,
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=chunker,
        read_only=False,
    )


@pytest.mark.parametrize("fm_keys", range(1, 13))
def test_section_in_frontmatter_file_returns_every_chunk(tmp_path, fm_keys):
    """Every frontmatter length: the old misread line moved with it, and at some lengths it landed on
    the section's own heading (the reported case) — the continuation was then dropped unannounced."""
    fts, mgr = _frontmatter_list_doc(tmp_path, fm_keys=fm_keys)
    heading = "6. What comes next (not decided)"
    assert len(fts.get_section_chunks("r.md", heading)) > 2, "the fixture must split the section"
    nc = mgr.read("r.md", section=heading)
    assert nc is not None
    for marker in ("Item 1 ", "Item six opens", "*First indented*", "*Second indented*", "*Third indented*", "CLOSING-LINE-OF-SIX"):
        assert marker in nc.content, f"{marker!r} missing — the section was cut short (AGO-826)"
    assert "later body" not in nc.content
    assert "not returned" not in nc.content


def test_section_read_announces_a_same_titled_section_left_out(tmp_path):
    """First-occurrence semantics stay, but the left-out later section is announced, not silent."""
    body = (
        "# A\n"
        + "\n".join(["intro"] * 16)
        + "\n## Repeat\n"
        + "\n".join(["first occurrence body"] * 6)
        + "\n## Middle\n"
        + "\n".join(["middle body"] * 6)
        + "\n## Repeat\n"
        + "\n".join(["second occurrence body"] * 6)
        + "\n"
    )
    mgr = _mgr_for(tmp_path, body)
    nc = mgr.read("a.md", section="Repeat")
    assert "first occurrence body" in nc.content
    assert "second occurrence body" not in nc.content
    # Not adjacent: the index's contiguous run stops at "Middle", so nothing of this run is left out.
    assert "not returned" not in nc.content


def test_section_read_announces_an_adjacent_same_titled_section_left_out(tmp_path):
    body = (
        "---\nk: v\n---\n"
        "# A\n"
        + "\n".join(["intro"] * 16)
        + "\n## Repeat\n"
        + "\n".join(["first occurrence body"] * 6)
        + "\n## Repeat\n"
        + "\n".join(["second occurrence body"] * 6)
        + "\n"
    )
    mgr = _mgr_for(tmp_path, body)
    nc = mgr.read("a.md", section="Repeat")
    assert "first occurrence body" in nc.content
    assert "second occurrence body" not in nc.content
    assert "[section 'Repeat': 1 more chunk(s) under this title not returned" in nc.content
    assert "read('a.md')" in nc.content


def test_section_read_leaves_out_an_adjacent_long_same_titled_section_and_says_so(tmp_path):
    """The later same-titled section is itself split: none of its chunks is taken for a continuation."""
    first = "\n\n".join(f"first para {n} " + " ".join(["one"] * 25) for n in range(4))
    second = "\n\n".join(f"second para {n} " + " ".join(["two"] * 25) for n in range(6))
    body = "---\nk: v\n---\n# A\n" + "\n".join(["intro"] * 31) + "\n## Repeat\n\n" + first + "\n## Repeat\n\n" + second + "\n"
    (tmp_path / "a.md").write_text(body, encoding="utf-8")
    fts = FTSIndex(db_path=":memory:")
    chunker = HeadingChunker(max_chunk_words=60)
    for note in scan_directory(tmp_path, chunk_strategy=chunker):
        fts.upsert_note(note)
    mgr = DocumentManager(fts=fts, source_dir=tmp_path, write_lock=threading.RLock(), chunk_strategy=chunker)
    chunks = fts.get_section_chunks("a.md", "Repeat")
    nc = mgr.read("a.md", section="Repeat")
    assert "first para 3" in nc.content
    assert "second para" not in nc.content
    n_first = sum(1 for c in chunks if "first para" in c["content"])
    assert n_first > 1 and len(chunks) - n_first > 1, "both sections must be split, or the test proves nothing"
    assert f"[section 'Repeat': {len(chunks) - n_first} more chunk(s) under this title not returned" in nc.content
