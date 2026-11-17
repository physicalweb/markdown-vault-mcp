"""Edit read-back: write and edit return what they changed (Agora Plan 234 step 1).

The vectors in ``fixtures/readback_vectors.json`` are shared with the Agora
daemon, whose ``memory_update`` returns the same shape; its tests pin the same
bytes on the same inputs.  Here each vector is pinned, and each diff is checked
independently by ``git apply``: applied to the text before, it gives the text
after.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client

from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.fts_index import FTSIndex
from markdown_vault_mcp.git import GitWriteStrategy
from markdown_vault_mcp.managers.document import DocumentManager
from markdown_vault_mcp.scanner import HeadingChunker
from markdown_vault_mcp.utils.readback import (
    MAX_DIFF_LINES,
    changed_lines,
    readback,
    unified_diff,
)
from tests.fixtures.git import _run_git

if TYPE_CHECKING:
    from tests.fixtures.git import GitRepoPair

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "readback_vectors.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("vector", VECTORS, ids=[v["name"] for v in VECTORS])
def test_vector_is_pinned(vector: dict[str, Any]) -> None:
    expected = {
        k: vector[k] for k in ("diff", "lines", "diff_truncated") if k in vector
    }
    assert readback(vector["path"], vector["before"], vector["after"]) == expected


@pytest.mark.parametrize(
    "vector",
    [v for v in VECTORS if v["diff"] and "diff_truncated" not in v],
    ids=lambda v: v["name"],
)
def test_git_apply_takes_before_to_after(
    vector: dict[str, Any], tmp_path: Path
) -> None:
    note = tmp_path / vector["path"]
    note.parent.mkdir(parents=True)
    note.write_bytes(vector["before"].encode("utf-8"))
    patch = tmp_path / "change.patch"
    patch.write_bytes(vector["diff"].encode("utf-8"))
    subprocess.run(
        ["git", "apply", "--unsafe-paths", "--whitespace=nowarn", str(patch)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert note.read_bytes().decode("utf-8") == vector["after"]


def test_cut_diff_names_the_range_to_read() -> None:
    before = "".join(f"row {i}\n" for i in range(200))
    after = before.replace("row 50\n", "".join(f"new {i}\n" for i in range(150)))
    out = readback("n.md", before, after)
    assert out["diff_truncated"]["shown"] == MAX_DIFF_LINES
    assert out["diff_truncated"]["total"] > MAX_DIFF_LINES
    assert len(out["diff"].split("\n")) == MAX_DIFF_LINES + 1  # trailing newline
    assert out["lines"] == {"first": 51, "last": 200}


def test_a_new_note_returns_its_range_and_no_diff() -> None:
    assert readback("n.md", None, "a\nb\nc") == {"lines": {"first": 1, "last": 3}}
    assert readback("n.md", None, "") == {"lines": None}


def test_large_rewrite_falls_back_to_one_block() -> None:
    before = "".join(f"a{i}\n" for i in range(600))
    after = "".join(f"b{i}\n" for i in range(600))
    diff = unified_diff("n.md", before, after)
    body = diff.split("\n")[3:-1]
    assert body == [f"-a{i}" for i in range(600)] + [f"+b{i}" for i in range(600)]


def test_changed_lines_is_none_when_nothing_changed() -> None:
    assert changed_lines("a\n", "a\n") is None


# --- Through the manager: each edit mode -------------------------------------


@pytest.fixture
def doc_mgr(tmp_path: Path) -> DocumentManager:
    (tmp_path / "note.md").write_text(
        "# Note\n\nalpha\nbeta — “quoted”\ngamma\nalpha\n", encoding="utf-8"
    )
    return DocumentManager(
        fts=FTSIndex(db_path=":memory:"),
        source_dir=tmp_path,
        write_lock=threading.RLock(),
        chunk_strategy=HeadingChunker(),
        read_only=False,
    )


def test_exact_edit_returns_its_diff(doc_mgr: DocumentManager) -> None:
    result = doc_mgr.edit("note.md", old_text="gamma", new_text="delta")
    assert result.match_type == "exact"
    assert result.lines == {"first": 5, "last": 5}
    assert "-gamma\n+delta\n" in (result.diff or "")
    assert result.commit is None  # no git here


def test_normalized_edit_shows_what_it_really_replaced(
    doc_mgr: DocumentManager,
) -> None:
    # Sent with straight quotes; the note has curly ones.
    result = doc_mgr.edit(
        "note.md", old_text='beta — "quoted"', new_text='beta — "said"'
    )
    assert result.match_type == "normalized"
    assert '-beta — “quoted”\n+beta — "said"\n' in (result.diff or "")


def test_line_range_edit(doc_mgr: DocumentManager) -> None:
    result = doc_mgr.edit("note.md", new_text="ALPHA\nBETA", line_start=3, line_end=4)
    assert result.lines == {"first": 3, "last": 4}
    assert "-alpha\n-beta — “quoted”\n+ALPHA\n+BETA\n" in (result.diff or "")


def test_scoped_edit(doc_mgr: DocumentManager) -> None:
    result = doc_mgr.edit(
        "note.md", old_text="alpha", new_text="omega", line_start=6, line_end=6
    )
    assert result.lines == {"first": 6, "last": 6}
    assert "-alpha\n+omega\n" in (result.diff or "")
    assert "@@ -3,4 +3,4 @@" in (result.diff or "")


def test_write_over_a_note_returns_its_diff(doc_mgr: DocumentManager) -> None:
    result = doc_mgr.write("note.md", "# Note\n\nonly this\n")
    assert result.created is False
    assert result.diff is not None and "+only this\n" in result.diff
    assert result.lines == {"first": 3, "last": 3}


def test_write_of_a_new_note_returns_its_range(doc_mgr: DocumentManager) -> None:
    result = doc_mgr.write("fresh.md", "one\ntwo\n")
    assert result.created is True
    assert result.diff is None
    assert result.lines == {"first": 1, "last": 2}


def test_write_of_the_same_text_returns_an_empty_diff(
    doc_mgr: DocumentManager,
) -> None:
    text = (doc_mgr._source_dir / "note.md").read_text(encoding="utf-8")
    result = doc_mgr.write("note.md", text)
    assert (result.diff, result.lines) == ("", None)


# --- Through git: the commit id agrees with git show -------------------------


def _strategy(pair: GitRepoPair) -> GitWriteStrategy:
    return GitWriteStrategy(
        enable_pull=True,
        enable_push=True,
        push_on_write=True,
        repo_path=pair.local_path,
        commit_name="vault-bot",
        commit_email="vault-bot@example.com",
    )


def _hunks(diff: str) -> str:
    """The hunks, without the function context git adds after a header's ``@@``."""
    body = diff[diff.index("@@") :].split("\n")
    return "\n".join(
        line[: line.index("@@", 2) + 2] if line.startswith("@@") else line
        for line in body
    )


def test_commit_and_diff_agree_with_git_show(git_repo_pair: GitRepoPair) -> None:
    strategy = _strategy(git_repo_pair)
    col = Collection(
        source_dir=git_repo_pair.local_path,
        read_only=False,
        on_write=strategy,
        git_strategy=strategy,
    )
    try:
        body = "".join(f"line {i}\n" for i in range(1, 21))
        written = col.write("note.md", body)
        assert (
            written.commit
            == _run_git(git_repo_pair.remote_path, "rev-parse", "main").strip()
        )

        edited = col.edit("note.md", old_text="line 10", new_text="line ten")
        assert edited.pushed is True
        remote_head = _run_git(git_repo_pair.remote_path, "rev-parse", "main")
        assert edited.commit == remote_head.strip()
        shown = _run_git(
            git_repo_pair.local_path, "show", "--format=", edited.commit or ""
        )
        assert _hunks(shown) == _hunks(edited.diff or "")

        # Nothing changed: no commit is made, so none is reported.
        same = col.write("note.md", (git_repo_pair.local_path / "note.md").read_text())
        assert (same.diff, same.commit) == ("", None)
    finally:
        col.close()


# --- Through the tool: the result's shape ------------------------------------


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result.data, dict):
        return result.data
    return json.loads(result.content[0].text)


async def test_tool_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from markdown_vault_mcp.server import make_server
    from tests.test_git_sync import _CLEAR_VARS

    for var in _CLEAR_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(tmp_path))
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_READ_ONLY", "false")
    async with Client(make_server()) as client:
        created = _payload(
            await client.call_tool("write", {"path": "t.md", "content": "a\nb\n"})
        )
        edited = _payload(
            await client.call_tool(
                "edit", {"path": "t.md", "old_text": "b", "new_text": "c"}
            )
        )
        attached = _payload(
            await client.call_tool(
                "write", {"path": "x.png", "content_base64": "iVBORw0K"}
            )
        )
    # No git here, so no commit; the keys appear only when set.
    assert created == {
        "path": "t.md",
        "created": True,
        "pushed": None,
        "push_error": None,
        "lines": {"first": 1, "last": 2},
    }
    assert edited["diff"].endswith("@@ -1,2 +1,2 @@\n a\n-b\n+c\n")
    assert edited["lines"] == {"first": 2, "last": 2}
    assert "commit" not in edited and "diff_truncated" not in edited
    # An attachment's result keeps its earlier shape.
    assert set(attached) == {"path", "created", "pushed", "push_error"}
