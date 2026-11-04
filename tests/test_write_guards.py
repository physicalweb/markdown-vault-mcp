"""Tests for the write guards (AGO-853): malformed writes refused at the act.

Three refusals, each naming its rule and the permitted next act (AGO-854):
a placeholder body, backslash-escaped quotes added outside code, and an
edit anchor that matches zero or several times.  Every pass case beside
them is a write a note-taker legitimately makes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.exceptions import EditConflictError
from markdown_vault_mcp.server import make_server
from markdown_vault_mcp.utils.write_guards import (
    check_added_escapes,
    check_placeholder_body,
    count_stray_escapes,
    placeholder_reason,
)
from tests.test_server import _CLEAR_VARS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_FENCED_JSON = '```json\n{"said": "she wrote \\"no\\""}\n```\n'


# ---------------------------------------------------------------------------
# Placeholder bodies
# ---------------------------------------------------------------------------


class TestPlaceholderReason:
    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            ("", "is empty"),
            ("  \n\n\t\n", "is empty"),
            ("# Reading notes\n", "has only headings"),
            ("# A\n\n## B\n### C\n", "has only headings"),
            ("TODO", "is only the placeholder 'TODO'"),
            ("# Notes\n\nTODO.\n", "is only the placeholder 'TODO.'"),
            ("placeholder", "is only the placeholder 'placeholder'"),
            ("TBD", "is only the placeholder 'TBD'"),
            ("...", "is only the placeholder '...'"),
            ("…", "is only the placeholder '…'"),
            ("Lorem ipsum dolor sit amet.", "is only placeholder text ('lorem ipsum')"),
        ],
    )
    def test_refused(self, body: str, reason: str) -> None:
        assert placeholder_reason(body) == reason

    @pytest.mark.parametrize(
        "body",
        [
            "ok",
            "Met the river at three.",
            "# Body\n\nBody.\n",
            "TODO: compare the two readings of the second letter.",
            "# Ledger\n\n- next: reread section 2\n",
            "The word todo appears in this sentence, which is fine.",
            "#hashtag-only line is not a heading",
            "```\ncode only\n```\n",
        ],
    )
    def test_short_real_notes_pass(self, body: str) -> None:
        assert placeholder_reason(body) is None

    def test_message_names_rule_and_next_act(self) -> None:
        with pytest.raises(ValueError) as exc:
            check_placeholder_body("docs/x.md", "# X\n")
        msg = str(exc.value)
        assert "'docs/x.md' has only headings" in msg
        assert "rule: a note is written with its content, AGO-853" in msg
        assert "Nothing was written. Next: write it again" in msg


# ---------------------------------------------------------------------------
# Stray escapes
# ---------------------------------------------------------------------------


class TestCountStrayEscapes:
    @pytest.mark.parametrize(
        ("text", "count"),
        [
            ('She said \\"no\\".', 2),
            ('She said "no".', 0),
            ('Inline `{"a": "\\"b\\""}` stays.', 0),
            ('Double ``x ` \\" y`` span, then \\" prose.', 1),
            (_FENCED_JSON + 'after \\"x\\"', 2),
            ('~~~\n\\"in tilde fence\\"\n~~~\n', 0),
            ('````\n```\n\\"still fenced\\"\n````\n', 0),
            ('---\ntitle: "a \\"b\\""\n---\nbody', 0),
            ('a \\\\"b', 0),  # an escaped backslash, then a plain quote
            ('a \\\\\\"b', 1),
        ],
    )
    def test_counts(self, text: str, count: int) -> None:
        assert count_stray_escapes(text) == count

    def test_unclosed_fence_hides_the_rest(self) -> None:
        assert count_stray_escapes('```\n\\"x\\"\n') == 0

    def test_only_added_escapes_refused(self) -> None:
        old = 'Kept \\"as is\\".\n'
        check_added_escapes("n.md", old, old + "More plain text.\n")
        with pytest.raises(ValueError) as exc:
            check_added_escapes("n.md", old, old + 'And \\"new\\".\n')
        msg = str(exc.value)
        assert "adds 2 backslash-escaped quotation marks" in msg
        assert "rule: prose carries plain quotes, AGO-853" in msg
        assert "pass allow_escapes=true" in msg

    def test_singular(self) -> None:
        with pytest.raises(
            ValueError, match="adds 1 backslash-escaped quotation mark "
        ):
            check_added_escapes("n.md", "", 'one \\" here')


# ---------------------------------------------------------------------------
# Collection: guards are opt-in below the tool layer
# ---------------------------------------------------------------------------


@pytest.fixture
def col(vault_path: Path) -> Iterator[Collection]:
    c = Collection(source_dir=vault_path, read_only=False)
    c.build_index()
    yield c
    c.close()


class TestCollection:
    def test_default_allows_escapes(self, col: Collection, vault_path: Path) -> None:
        col.write("lib.md", 'Library \\"caller\\".\n')
        assert '\\"' in (vault_path / "lib.md").read_text()

    def test_write_refused_leaves_no_file(
        self, col: Collection, vault_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="adds 2 backslash"):
            col.write("x.md", 'Ki said \\"yes\\".\n', allow_escapes=False)
        assert not (vault_path / "x.md").exists()

    def test_overwrite_keeping_existing_escapes_passes(self, col: Collection) -> None:
        col.write("keep.md", 'Old \\"quote\\".\n')
        col.write("keep.md", 'Old \\"quote\\".\nNew line.\n', allow_escapes=False)

    def test_edit_inside_fence_passes(self, col: Collection) -> None:
        col.write("j.md", "# J\n\n" + _FENCED_JSON)
        col.edit(
            "j.md",
            old_text='she wrote \\"no\\"',
            new_text='she wrote \\"never\\"',
            allow_escapes=False,
        )

    def test_edit_adding_prose_escape_refused(
        self, col: Collection, vault_path: Path
    ) -> None:
        col.write("e.md", "# E\n\nA plain line.\n")
        with pytest.raises(ValueError, match="adds 2 backslash"):
            col.edit(
                "e.md",
                old_text="A plain line.",
                new_text='A \\"quoted\\" line.',
                allow_escapes=False,
            )
        assert (vault_path / "e.md").read_text() == "# E\n\nA plain line.\n"

    def test_edit_line_range_refused(self, col: Collection) -> None:
        col.write("l.md", "one\ntwo\nthree\n")
        with pytest.raises(ValueError, match="adds 1 backslash"):
            col.edit(
                "l.md",
                new_text='t\\"wo',
                line_start=2,
                line_end=2,
                allow_escapes=False,
            )


class TestAnchorRefusals:
    def test_twice(self, col: Collection) -> None:
        col.write("a.md", "same line\nsame line\n")
        with pytest.raises(EditConflictError) as exc:
            col.edit("a.md", old_text="same line", new_text="x")
        msg = str(exc.value)
        assert "old_text appears 2 times in a.md; must appear exactly once" in msg
        assert "rule: an edit's old_text matches exactly once, AGO-853" in msg
        assert "Next: widen old_text" in msg

    def test_twice_after_normalization(self, col: Collection) -> None:
        col.write("n.md", "a \u2014 b\na \u2013 b\n")
        with pytest.raises(EditConflictError) as exc:
            col.edit("n.md", old_text="a - b", new_text="x")
        assert "2 times in n.md after normalization" in str(exc.value)
        assert "Next: widen old_text" in str(exc.value)

    def test_none(self, col: Collection) -> None:
        col.write("z.md", "first\nsecond\n")
        with pytest.raises(EditConflictError) as exc:
            col.edit("z.md", old_text="third", new_text="x")
        msg = str(exc.value)
        assert msg.startswith("old_text not found in z.md (0 matches;")
        assert "Next: 'read' the section first" in msg
        assert "backslash-escaped" not in msg

    def test_none_with_escaped_anchor_hints_plain_quotes(self, col: Collection) -> None:
        col.write("q.md", 'She said "no".\n')
        with pytest.raises(EditConflictError) as exc:
            col.edit("q.md", old_text='She said \\"no\\".', new_text="x")
        assert "send it with plain quotes" in str(exc.value)

    def test_once_passes(self, col: Collection) -> None:
        col.write("o.md", "only once\n")
        assert col.edit("o.md", old_text="only once", new_text="done").match_type == (
            "exact"
        )


# ---------------------------------------------------------------------------
# MCP tools: guards on by default
# ---------------------------------------------------------------------------


@pytest.fixture
def _env(vault_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in _CLEAR_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(vault_path))
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_READ_ONLY", "false")
    return vault_path


async def _tool_error(client: Client, name: str, args: dict) -> str:
    with pytest.raises(ToolError) as exc:
        await client.call_tool(name, args)
    return str(exc.value)


@pytest.mark.usefixtures("_env")
class TestTools:
    async def test_write_placeholder_refused(self, vault_path: Path) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client, "write", {"path": "stub.md", "content": "# Stub\n\nTODO\n"}
            )
        assert "is only the placeholder 'TODO'" in msg
        assert not (vault_path / "stub.md").exists()

    async def test_write_empty_with_frontmatter_refused(self) -> None:
        async with Client(make_server()) as client:
            msg = await _tool_error(
                client,
                "write",
                {"path": "fm.md", "content": "", "frontmatter": {"title": "T"}},
            )
        assert "'fm.md' is empty" in msg

    async def test_write_one_line_note_passes(self) -> None:
        async with Client(make_server()) as client:
            result = await client.call_tool(
                "write", {"path": "short.md", "content": "Asked about the label.\n"}
            )
        assert result.data["created"] is True

    async def test_write_escapes_refused_then_opt_out(self, vault_path: Path) -> None:
        args = {"path": "esc.md", "content": 'He wrote \\"hau\\".\n'}
        async with Client(make_server()) as client:
            msg = await _tool_error(client, "write", args)
            assert "adds 2 backslash-escaped quotation marks" in msg
            assert not (vault_path / "esc.md").exists()
            await client.call_tool("write", {**args, "allow_escapes": True})
        assert (vault_path / "esc.md").exists()

    async def test_write_json_in_fence_passes(self) -> None:
        async with Client(make_server()) as client:
            result = await client.call_tool(
                "write",
                {"path": "json.md", "content": "Payload:\n\n" + _FENCED_JSON},
            )
        assert result.data["created"] is True

    async def test_edit_escapes_refused(self) -> None:
        async with Client(make_server()) as client:
            await client.call_tool(
                "write", {"path": "ed.md", "content": "A plain line.\n"}
            )
            msg = await _tool_error(
                client,
                "edit",
                {"path": "ed.md", "old_text": "plain", "new_text": '\\"plain\\"'},
            )
        assert "adds 2 backslash-escaped quotation marks" in msg

    async def test_edit_anchor_twice_names_count_and_next_act(self) -> None:
        async with Client(make_server()) as client:
            await client.call_tool("write", {"path": "tw.md", "content": "dup\ndup\n"})
            msg = await _tool_error(
                client, "edit", {"path": "tw.md", "old_text": "dup", "new_text": "x"}
            )
        assert "appears 2 times" in msg
        assert "Next: widen old_text" in msg
