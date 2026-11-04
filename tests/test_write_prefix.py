"""Tests for the write prefix (AGO-846): writes confined to one folder.

``MARKDOWN_VAULT_MCP_WRITE_PREFIX`` confines every mutating operation
(write, edit, delete, both sides of rename, fetch, uploads) to one
vault-relative folder.  Reads, search and listing stay vault-wide.  A
refused path names the path that was likely meant when
``<prefix>/<first segment>`` exists as a folder.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from markdown_vault_mcp import _server_deps, uploads
from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.config import load_config
from markdown_vault_mcp.server import make_server
from markdown_vault_mcp.utils import check_write_prefix
from tests.test_server import _CLEAR_VARS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_LINE = "Writes go only under the server's write folder"
_SUGGEST = (
    "Writes go only under docs/; 'design/x.md' is outside it. "
    "Did you mean docs/design/x.md?"
)


@pytest.fixture
def docs_vault(vault_path: Path) -> Path:
    """The fixture vault plus ``docs/design/`` holding one note."""
    (vault_path / "docs" / "design").mkdir(parents=True)
    (vault_path / "docs" / "design" / "inside.md").write_text(
        "# Inside\n\nBody.\n", encoding="utf-8"
    )
    return vault_path


@pytest.fixture
def _env_docs_prefix(docs_vault: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Writable server env with ``WRITE_PREFIX=docs``."""
    for var in _CLEAR_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(docs_vault))
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_READ_ONLY", "false")
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_WRITE_PREFIX", "docs")
    return docs_vault


@pytest.fixture
def col(docs_vault: Path) -> Iterator[Collection]:
    """A writable Collection confined to ``docs``."""
    c = Collection(
        source_dir=docs_vault,
        read_only=False,
        write_prefix="docs",
        attachment_extensions=["pdf"],
    )
    c.build_index()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


class TestCheckWritePrefix:
    """``check_write_prefix`` decides inside/outside and builds the message."""

    @pytest.mark.parametrize("prefix", [None, ""])
    def test_no_prefix_allows_anything(
        self, tmp_path: Path, prefix: str | None
    ) -> None:
        check_write_prefix("anywhere/x.md", prefix, tmp_path)

    @pytest.mark.parametrize(
        "path", ["docs/x.md", "docs/design/x.md", "/docs/x.md", "./docs/x.md"]
    )
    def test_inside_allowed(self, tmp_path: Path, path: str) -> None:
        check_write_prefix(path, "docs", tmp_path)

    def test_outside_suggests_when_folder_exists(self, tmp_path: Path) -> None:
        (tmp_path / "docs" / "design").mkdir(parents=True)
        with pytest.raises(ValueError) as exc:
            check_write_prefix("design/x.md", "docs", tmp_path)
        assert str(exc.value) == _SUGGEST

    def test_outside_without_folder_says_where_writes_go(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        with pytest.raises(ValueError) as exc:
            check_write_prefix("design/x.md", "docs", tmp_path)
        assert (
            str(exc.value) == "Writes go only under docs/; 'design/x.md' is outside it."
        )

    def test_root_file_has_no_suggestion(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        with pytest.raises(ValueError, match="Writes go only under docs/") as exc:
            check_write_prefix("x.md", "docs", tmp_path)
        assert "Did you mean" not in str(exc.value)

    @pytest.mark.parametrize("path", ["docs", "docs/", "docs/../src/x.md", "../x.md"])
    def test_prefix_itself_and_escapes_refused(self, tmp_path: Path, path: str) -> None:
        with pytest.raises(ValueError, match="Writes go only under docs/"):
            check_write_prefix(path, "docs", tmp_path)

    def test_prefix_is_a_folder_not_a_string_prefix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            check_write_prefix("docsx/x.md", "docs", tmp_path)

    def test_multi_segment_prefix(self, tmp_path: Path) -> None:
        check_write_prefix("content/notes/x.md", "/content/notes/", tmp_path)
        with pytest.raises(ValueError, match="under content/notes/"):
            check_write_prefix("content/x.md", "content/notes", tmp_path)


class TestConfig:
    """``MARKDOWN_VAULT_MCP_WRITE_PREFIX`` reaches the Collection."""

    def test_unset_means_no_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(tmp_path))
        monkeypatch.delenv("MARKDOWN_VAULT_MCP_WRITE_PREFIX", raising=False)
        config = load_config()
        assert config.write_prefix is None
        assert config.to_collection_kwargs()["write_prefix"] is None

    def test_set_is_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(tmp_path))
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_WRITE_PREFIX", "/docs/")
        config = load_config()
        assert config.write_prefix == "docs"
        assert config.to_collection_kwargs()["write_prefix"] == "docs"


# ---------------------------------------------------------------------------
# Collection facade: every mutating method
# ---------------------------------------------------------------------------


class TestCollectionGuard:
    """Each mutating Collection method refuses outside and works inside."""

    def test_write(self, col: Collection, docs_vault: Path) -> None:
        with pytest.raises(
            ValueError, match=re.escape("Did you mean docs/design/x.md")
        ):
            col.write("design/x.md", "# X\n")
        assert not (docs_vault / "design").exists()
        assert col.write("docs/design/x.md", "# X\n").created is True

    def test_write_attachment(self, col: Collection, docs_vault: Path) -> None:
        with pytest.raises(ValueError, match="Writes go only under docs/"):
            col.write_attachment("a.pdf", b"%PDF")
        assert not (docs_vault / "a.pdf").exists()
        assert col.write_attachment("docs/a.pdf", b"%PDF").created is True

    def test_edit(self, col: Collection, docs_vault: Path) -> None:
        before = (docs_vault / "simple.md").read_text(encoding="utf-8")
        with pytest.raises(ValueError, match="Writes go only under docs/"):
            col.edit("simple.md", old_text="Simple", new_text="Changed")
        assert (docs_vault / "simple.md").read_text(encoding="utf-8") == before
        col.edit("docs/design/inside.md", old_text="Body.", new_text="Edited.")
        assert "Edited." in (docs_vault / "docs/design/inside.md").read_text(
            encoding="utf-8"
        )

    def test_delete(self, col: Collection, docs_vault: Path) -> None:
        with pytest.raises(ValueError, match="Writes go only under docs/"):
            col.delete("simple.md")
        assert (docs_vault / "simple.md").exists()
        col.delete("docs/design/inside.md")
        assert not (docs_vault / "docs/design/inside.md").exists()

    def test_rename_refuses_outside_source(
        self, col: Collection, docs_vault: Path
    ) -> None:
        with pytest.raises(ValueError, match=re.escape("'simple.md' is outside it")):
            col.rename("simple.md", "docs/simple.md")
        assert (docs_vault / "simple.md").exists()

    def test_rename_refuses_outside_target(
        self, col: Collection, docs_vault: Path
    ) -> None:
        with pytest.raises(
            ValueError, match=re.escape("Did you mean docs/design/out.md")
        ):
            col.rename("docs/design/inside.md", "design/out.md")
        assert (docs_vault / "docs/design/inside.md").exists()

    def test_rename_inside_allowed(self, col: Collection, docs_vault: Path) -> None:
        col.rename("docs/design/inside.md", "docs/design/moved.md")
        assert (docs_vault / "docs/design/moved.md").exists()

    def test_reads_stay_vault_wide(self, col: Collection) -> None:
        assert col.read("simple.md") is not None
        paths = {n.path for n in col.list()}
        assert "simple.md" in paths
        assert "docs/design/inside.md" in paths


# ---------------------------------------------------------------------------
# MCP tools end to end
# ---------------------------------------------------------------------------


async def _tool_error(client: Client, name: str, args: dict) -> str:
    with pytest.raises(ToolError) as exc:
        await client.call_tool(name, args)
    return str(exc.value)


@pytest.mark.usefixtures("_env_docs_prefix")
class TestMcpTools:
    """The mutating tools refuse outside ``docs/``; reads do not."""

    async def test_write_refused_with_suggestion(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client, "write", {"path": "design/x.md", "content": "# X\n\nA line.\n"}
            )
        assert _SUGGEST in msg
        assert not (docs_vault / "design").exists()

    async def test_write_allowed_inside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            result = await client.call_tool(
                "write", {"path": "docs/design/x.md", "content": "# X\n\nA line.\n"}
            )
        assert result.data["created"] is True
        assert (docs_vault / "docs/design/x.md").exists()

    async def test_edit_refused_outside(self) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client,
                "edit",
                {"path": "simple.md", "old_text": "Simple", "new_text": "X"},
            )
        assert "Writes go only under docs/; 'simple.md' is outside it." in msg

    async def test_edit_allowed_inside(self) -> None:
        async with Client(make_server()) as client:
            result = await client.call_tool(
                "edit",
                {
                    "path": "docs/design/inside.md",
                    "old_text": "Body.",
                    "new_text": "Edited.",
                },
            )
        assert result.data["replacements"] == 1

    async def test_delete_refused_outside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(client, "delete", {"path": "simple.md"})
        assert "Writes go only under docs/" in msg
        assert (docs_vault / "simple.md").exists()

    async def test_delete_allowed_inside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            await client.call_tool("delete", {"path": "docs/design/inside.md"})
        assert not (docs_vault / "docs/design/inside.md").exists()

    async def test_rename_refused_when_source_outside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client,
                "rename",
                {"old_path": "simple.md", "new_path": "docs/simple.md"},
            )
        assert "'simple.md' is outside it" in msg
        assert (docs_vault / "simple.md").exists()

    async def test_rename_refused_when_target_outside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client,
                "rename",
                {"old_path": "docs/design/inside.md", "new_path": "design/x.md"},
            )
        assert _SUGGEST in msg
        assert (docs_vault / "docs/design/inside.md").exists()

    async def test_rename_allowed_inside(self, docs_vault: Path) -> None:
        async with Client(make_server()) as client:
            await client.call_tool(
                "rename",
                {
                    "old_path": "docs/design/inside.md",
                    "new_path": "docs/design/moved.md",
                },
            )
        assert (docs_vault / "docs/design/moved.md").exists()

    async def test_fetch_refused_before_download(self, docs_vault: Path) -> None:
        from unittest.mock import patch

        import httpx

        with patch.object(httpx, "AsyncClient") as client_cls:
            async with Client(make_server()) as client:
                msg = await _tool_error(
                    client,
                    "fetch",
                    {"url": "https://example.com/n.md", "path": "design/x.md"},
                )
        assert _SUGGEST in msg
        client_cls.assert_not_called()
        assert not (docs_vault / "design").exists()

    async def test_reads_stay_vault_wide(self) -> None:
        async with Client(make_server()) as client:
            read = await client.call_tool("read", {"path": "simple.md"})
            listed = await client.call_tool("list_documents", {})
        assert "Simple" in str(read.data)
        assert "simple.md" in str(listed.content)

    @pytest.mark.parametrize("name", ["write", "edit", "delete", "rename", "fetch"])
    async def test_description_says_so(self, name: str) -> None:
        async with Client(make_server()) as client:
            tools = {t.name: t for t in await client.list_tools()}
        assert _LINE in (tools[name].description or "")


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


class TestUploadTarget:
    """``create_upload_link``'s pre-link validator applies the same guard."""

    def test_upload_target_outside_refused(self, col: Collection) -> None:
        saved = _server_deps._collection_singleton
        _server_deps.set_collection_singleton(col)
        try:
            with pytest.raises(ValueError, match="Writes go only under docs/"):
                uploads._validate_upload_target("note.md", None)
        finally:
            _server_deps._collection_singleton = saved
