"""rename moves a whole folder, keeping its tree and fixing links (AGO-641, agora).

Modelled on the MCP reference filesystem server's ``move_file``, which moves
files and directories alike: ``rename`` with an existing folder as
``old_path`` moves the folder in one act, re-indexes every document under it,
and with ``update_links=True`` rewrites links to the moved documents.  Under
git it is one commit, authored as the caller, pushed by the same write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from markdown_vault_mcp._author_context import reset, set_author
from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.exceptions import DocumentExistsError
from markdown_vault_mcp.git import GitWriteStrategy
from tests.fixtures.git import _run_git

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.git import GitRepoPair


def _seed(root: Path) -> None:
    """A folder ``notes`` with a nested tree, an attachment and links in/out."""
    (root / "notes" / "deep" / "deeper").mkdir(parents=True)
    (root / "notes" / "a.md").write_text(
        "# A\n\nSee [B](deep/b.md) and [[notes/deep/deeper/c]].\n"
    )
    (root / "notes" / "deep" / "b.md").write_text("# B\n\nBack to [A](../a.md).\n")
    (root / "notes" / "deep" / "deeper" / "c.md").write_text("# C\n\nQuokka.\n")
    (root / "notes" / "deep" / "pic.png").write_bytes(b"\x89PNG fake")
    (root / "index.md").write_text(
        "# Index\n\n"
        "- [A](notes/a.md)\n"
        "- [B](notes/deep/b.md#b)\n"
        "- [[notes/deep/deeper/c|the leaf]]\n"
        "- [Other](other.md)\n"
    )
    (root / "sub").mkdir()
    (root / "sub" / "rel.md").write_text(
        "# Rel\n\nSee [C](../notes/deep/deeper/c.md).\n"
    )
    (root / "other.md").write_text("# Other\n\nNo links here.\n")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root)
    return root


def _col(root: Path) -> Collection:
    col = Collection(source_dir=root, read_only=False)
    col.build_index()
    return col


class TestFolderMove:
    def test_moves_nested_tree(self, vault: Path) -> None:
        col = _col(vault)
        result = col.rename("notes", "archive/2026/notes")

        assert (result.old_path, result.new_path) == ("notes", "archive/2026/notes")
        assert not (vault / "notes").exists()
        dest = vault / "archive" / "2026" / "notes"
        assert (dest / "a.md").is_file()
        assert (dest / "deep" / "b.md").is_file()
        assert (dest / "deep" / "deeper" / "c.md").read_text() == "# C\n\nQuokka.\n"
        assert (dest / "deep" / "pic.png").read_bytes() == b"\x89PNG fake"

    def test_index_follows_the_move(self, vault: Path) -> None:
        col = _col(vault)
        col.rename("notes", "archive/notes")

        paths = {n.path for n in col.list()}
        assert "notes/a.md" not in paths
        assert {
            "archive/notes/a.md",
            "archive/notes/deep/b.md",
            "archive/notes/deep/deeper/c.md",
        } <= paths
        assert col.read("archive/notes/deep/deeper/c.md") is not None
        hits = {h.path for h in col.search("Quokka", mode="keyword")}
        assert hits == {"archive/notes/deep/deeper/c.md"}

    def test_update_links_rewrites_links_from_other_files(self, vault: Path) -> None:
        col = _col(vault)
        result = col.rename("notes", "archive/notes", update_links=True)

        index = (vault / "index.md").read_text()
        assert "[A](archive/notes/a.md)" in index
        assert "[B](archive/notes/deep/b.md#b)" in index
        assert "[[archive/notes/deep/deeper/c|the leaf]]" in index
        assert "[Other](other.md)" in index  # unrelated link untouched
        # A relative link from another folder is recomputed, still relative.
        rel = (vault / "sub" / "rel.md").read_text()
        assert "[C](../archive/notes/deep/deeper/c.md)" in rel
        # Sources rewritten: index.md, sub/rel.md and the moved a.md (whose
        # vault-path wikilink to c moved with it).  The moved b.md links to a
        # only relatively, so its text is unchanged and it is not counted.
        assert result.updated_links == 3
        assert (vault / "other.md").read_text() == "# Other\n\nNo links here.\n"
        assert col.get_backlinks("archive/notes/a.md")
        assert not col.get_broken_links()

    def test_links_inside_the_folder_keep_resolving(self, vault: Path) -> None:
        col = _col(vault)
        col.rename("notes", "archive/notes", update_links=True)

        dest = vault / "archive" / "notes"
        a = (dest / "a.md").read_text()
        assert "[B](deep/b.md)" in a  # relative link unchanged
        assert "[[archive/notes/deep/deeper/c]]" in a  # vault-path link moved
        assert "[A](../a.md)" in (dest / "deep" / "b.md").read_text()
        out = {o.target_path for o in col.get_outlinks("archive/notes/a.md")}
        assert {"archive/notes/deep/b.md", "archive/notes/deep/deeper/c.md"} <= out

    def test_without_update_links_sources_are_not_modified(self, vault: Path) -> None:
        col = _col(vault)
        before = (vault / "index.md").read_text()
        result = col.rename("notes", "archive/notes")
        assert result.updated_links == 0
        assert (vault / "index.md").read_text() == before

    def test_refuses_a_move_into_its_own_subtree(self, vault: Path) -> None:
        col = _col(vault)
        with pytest.raises(ValueError, match="own subtree"):
            col.rename("notes", "notes/deep/notes")
        with pytest.raises(ValueError, match="own subtree"):
            col.rename("notes", "notes")
        assert (vault / "notes" / "a.md").is_file()

    def test_refuses_an_existing_target(self, vault: Path) -> None:
        col = _col(vault)
        (vault / "taken").mkdir()
        with pytest.raises(DocumentExistsError):
            col.rename("notes", "taken")
        with pytest.raises(DocumentExistsError):
            col.rename("notes", "other.md")
        assert (vault / "notes" / "a.md").is_file()

    def test_refuses_the_root_traversal_and_if_match(self, vault: Path) -> None:
        col = _col(vault)
        with pytest.raises(ValueError, match="vault root"):
            col.rename(".", "elsewhere")
        with pytest.raises(ValueError, match="traversal"):
            col.rename("notes", "../outside")
        with pytest.raises(ValueError, match="traversal"):
            col.rename("..", "inside")
        with pytest.raises(ValueError, match="if_match"):
            col.rename("notes", "moved", if_match="abc")
        assert (vault / "notes" / "a.md").is_file()

    def test_single_file_rename_is_unchanged(self, vault: Path) -> None:
        col = _col(vault)
        result = col.rename("notes/a.md", "a2.md", update_links=True)
        assert result.updated_links == 2  # index.md and notes/deep/b.md
        assert "[A](a2.md)" in (vault / "index.md").read_text()
        assert "[A](../../a2.md)" in (vault / "notes/deep/b.md").read_text()


class TestFolderMoveUnderGit:
    def test_one_commit_authored_as_the_caller_and_pushed(
        self, git_repo_pair: GitRepoPair
    ) -> None:
        root = git_repo_pair.local_path
        _seed(root)
        _run_git(root, "add", "-A")
        _run_git(
            root,
            "-c",
            "user.name=seed",
            "-c",
            "user.email=seed@example.com",
            "commit",
            "-m",
            "seed",
        )
        _run_git(root, "push", "origin", "HEAD:main")
        head_before = _run_git(root, "rev-parse", "HEAD").strip()

        strategy = GitWriteStrategy(
            enable_pull=True,
            enable_push=True,
            push_on_write=True,
            repo_path=root,
            commit_name="vault-bot",
            commit_email="vault-bot@example.com",
        )
        col = Collection(
            source_dir=root, read_only=False, on_write=strategy, git_strategy=strategy
        )
        col.build_index()
        token = set_author("Arnon", "arnon@example.com")
        try:
            col.rename("notes", "archive/notes", update_links=True)
        finally:
            reset(token)
            col.close()

        commits = _run_git(
            root, "log", "--format=%an|%cn|%s", f"{head_before}..HEAD"
        ).splitlines()
        assert commits == ["Arnon|vault-bot|rename: archive/notes"]
        changed = set(
            _run_git(root, "show", "--name-only", "--format=", "HEAD").split()
        )
        assert {
            "notes/a.md",
            "archive/notes/a.md",
            "archive/notes/deep/deeper/c.md",
            "archive/notes/deep/pic.png",
            "index.md",
            "sub/rel.md",
        } <= changed
        assert (
            _run_git(root, "status", "--porcelain", "--untracked-files=no").strip()
            == ""
        )
        remote_head = _run_git(git_repo_pair.remote_path, "rev-parse", "main").strip()
        assert remote_head == _run_git(root, "rev-parse", "HEAD").strip()
