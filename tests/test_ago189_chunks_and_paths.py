"""AGO-189 defects 3 and 4.

3: the chunks of one long section came back from search with byte-identical
   content — the snippet map was keyed by (path, heading), which every chunk of
   the section shares, so each hit got the same (last-written) snippet.
4: a read that missed on a wrong path ("design/x.md" for "docs/design/x.md")
   answered "document is not indexed or does not exist" — an error that reads
   as evidence the document is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from markdown_vault_mcp.collection import Collection

from .conftest import MockEmbeddingProvider

if TYPE_CHECKING:
    from pathlib import Path

_QUERY = "manifest layers dimensions portable fields"


def _vault(tmp_path: Path) -> Path:
    paras = [
        f"Marker{i} manifest layers dimensions portable fields "
        + " ".join(f"filler{i}w{j}" for j in range(60))
        for i in range(8)
    ]
    body = (
        "# Family\n\n"
        + "\n".join(["intro"] * 35)
        + "\n\n## 2. The manifest\n\n"
        + "\n\n".join(paras)
        + "\n\n## 3. Other\n\nOther text here.\n"
    )
    (tmp_path / "docs" / "design").mkdir(parents=True)
    (tmp_path / "docs" / "design" / "fam.md").write_text(body, encoding="utf-8")
    return tmp_path


def _collection(tmp_path: Path, *, embeddings: bool = False) -> Collection:
    vault = _vault(tmp_path / "vault")
    kwargs: dict = {"source_dir": vault, "max_chunk_words": 80}
    if embeddings:
        kwargs["embedding_provider"] = MockEmbeddingProvider()
        kwargs["embeddings_path"] = tmp_path / "embeddings"
    return Collection(**kwargs)


def _sections(results):
    assert len(results) == 1
    return results[0].sections


@pytest.mark.parametrize("snippet_words", [None, 0, 30])
def test_chunks_of_one_section_are_not_duplicated(
    tmp_path: Path, snippet_words
) -> None:
    coll = _collection(tmp_path)
    kw = {} if snippet_words is None else {"snippet_words": snippet_words}
    secs = _sections(coll.search(_QUERY, chunks_per_file=10, **kw))
    assert len(secs) > 1, "the section must be split for this test to mean anything"
    assert {s.heading for s in secs} == {"2. The manifest"}
    contents = [s.content for s in secs]
    assert len(set(contents)) == len(contents), "two hits returned the same text"
    # Each hit carries its OWN chunk's text: exactly one marker per hit, none repeated.
    markers = [[f"Marker{i}" for i in range(8) if f"Marker{i}" in c] for c in contents]
    assert all(len(m) == 1 for m in markers), markers
    assert len({m[0] for m in markers}) == len(markers)


def test_hybrid_snippet_belongs_to_the_chunk_it_is_shown_for(tmp_path: Path) -> None:
    coll = _collection(tmp_path, embeddings=True)
    coll.build_embeddings()
    full = _sections(
        coll.search(_QUERY, mode="hybrid", chunks_per_file=10, snippet_words=0)
    )
    snip = _sections(
        coll.search(_QUERY, mode="hybrid", chunks_per_file=10, snippet_words=30)
    )
    pairs = [
        (f, s)
        for f, s in zip(full, snip, strict=True)
        if f.heading == "2. The manifest"
    ]
    assert pairs
    for f, s in pairs:
        marker = next(f"Marker{i}" for i in range(8) if f"Marker{i}" in f.content)
        assert marker in s.content


def test_section_read_on_a_wrong_path_names_the_document_it_may_mean(
    tmp_path: Path,
) -> None:
    coll = _collection(tmp_path)
    with pytest.raises(ValueError) as ei:
        coll.read("design/fam.md", section="2. The manifest")
    msg = str(ei.value)
    assert "Document not found: design/fam.md" in msg
    assert "'docs/design/fam.md'" in msg
    assert "not indexed or does not exist" not in msg


def test_whole_read_miss_hint_names_same_file_name_elsewhere(tmp_path: Path) -> None:
    coll = _collection(tmp_path)
    assert coll.read("other/fam.md") is None
    assert "'docs/design/fam.md'" in coll.missing_path_hint("other/fam.md")


def test_miss_with_no_candidate_says_how_paths_are_written(tmp_path: Path) -> None:
    coll = _collection(tmp_path)
    hint = coll.missing_path_hint("nowhere.md")
    assert "relative to the collection root" in hint
    assert "may mean" not in hint


def test_section_read_of_an_unindexed_file_says_so(tmp_path: Path) -> None:
    coll = _collection(tmp_path)
    coll.search(_QUERY)  # initialise the index first
    (tmp_path / "vault" / "docs" / "late.md").write_text(
        "# Late\n\n## A\n\ntext\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exists but is not in the index yet"):
        coll.read("docs/late.md", section="A")
