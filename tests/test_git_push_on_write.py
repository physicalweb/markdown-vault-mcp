"""Push-on-write: a write that reports success is on the remote (AGO-802, agora).

The debounced push left a committed write local-only for ~10 s after the tool
returned; a task killed in that gap lost a write the caller was told had
succeeded.  With ``push_on_write`` the strategy pushes inline (one fetch +
rebase retry on a rejected push), ``Collection.write`` / ``edit`` wait for
their own commit and push, and the result says whether it reached the remote.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastmcp import Client

from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.git import GitWriteStrategy, PushOutcome
from tests.fixtures.git import _run_git
from tests.test_git_force_methods import _seed_remote_commit
from tests.test_git_sync import _CLEAR_VARS

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.fixtures.git import GitRepoPair


def _strategy(pair: GitRepoPair, *, push_on_write: bool = True) -> GitWriteStrategy:
    return GitWriteStrategy(
        enable_pull=True,
        enable_push=True,
        push_delay_s=30.0,  # a debounce that would still be pending if used
        push_on_write=push_on_write,
        repo_path=pair.local_path,
        commit_name="vault-bot",
        commit_email="vault-bot@example.com",
    )


def _remote_files(pair: GitRepoPair) -> set[str]:
    out = _run_git(pair.remote_path, "ls-tree", "--name-only", "main")
    return set(out.split())


def _write(pair: GitRepoPair, strategy: GitWriteStrategy, name: str, body: str) -> Any:
    path = pair.local_path / name
    path.write_text(body)
    return strategy(path, body, "write")


class TestStrategyPushOnWrite:
    def test_pushed_before_return(self, git_repo_pair: GitRepoPair) -> None:
        strategy = _strategy(git_repo_pair)
        try:
            outcome = _write(git_repo_pair, strategy, "a.md", "a\n")
            assert outcome == PushOutcome(pushed=True)
            assert "a.md" in _remote_files(git_repo_pair)
            assert strategy._timer is None  # no debounce armed
            assert strategy._push_pending is False
        finally:
            strategy.close()

    def test_off_keeps_the_debounce(self, git_repo_pair: GitRepoPair) -> None:
        strategy = _strategy(git_repo_pair, push_on_write=False)
        try:
            assert _write(git_repo_pair, strategy, "a.md", "a\n") is None
            assert "a.md" not in _remote_files(git_repo_pair)
            assert strategy._timer is not None
        finally:
            strategy.close()
        assert "a.md" in _remote_files(git_repo_pair)  # close() flushed it

    def test_foreign_commit_rejection_rebases_and_lands(
        self, git_repo_pair: GitRepoPair
    ) -> None:
        strategy = _strategy(git_repo_pair)
        try:
            _seed_remote_commit(
                git_repo_pair, clone_name="foreign", file_name="theirs.md", body="t\n"
            )
            outcome = _write(git_repo_pair, strategy, "ours.md", "o\n")
            assert outcome == PushOutcome(pushed=True)
            assert {"theirs.md", "ours.md"} <= _remote_files(git_repo_pair)
            committer = _run_git(
                git_repo_pair.local_path, "log", "-1", "--format=%cn"
            ).strip()
            assert committer == "vault-bot"
        finally:
            strategy.close()

    def test_conflicting_foreign_commit_is_reported_and_aborted(
        self, git_repo_pair: GitRepoPair
    ) -> None:
        strategy = _strategy(git_repo_pair)
        try:
            _seed_remote_commit(
                git_repo_pair, clone_name="foreign", file_name="same.md", body="t\n"
            )
            outcome = _write(git_repo_pair, strategy, "same.md", "ours\n")
            assert isinstance(outcome, PushOutcome)
            assert outcome.pushed is False
            assert outcome.error
            assert strategy._push_pending is True  # retried later
            assert not strategy._rebase_in_progress(git_repo_pair.local_path, None)
            # The local commit is kept, not lost.
            assert (git_repo_pair.local_path / "same.md").read_text() == "ours\n"
        finally:
            strategy._closed = True

    def test_unreachable_remote_is_reported(self, git_repo_pair: GitRepoPair) -> None:
        strategy = _strategy(git_repo_pair)
        _write(git_repo_pair, strategy, "first.md", "1\n")  # init + one push
        _run_git(
            git_repo_pair.local_path,
            "remote",
            "set-url",
            "origin",
            str(git_repo_pair.remote_path.parent / "gone.git"),
        )
        outcome = _write(git_repo_pair, strategy, "second.md", "2\n")
        assert isinstance(outcome, PushOutcome)
        assert outcome.pushed is False
        assert outcome.error
        assert strategy._push_pending is True
        strategy._closed = True


def _collection(pair: GitRepoPair, strategy: GitWriteStrategy) -> Collection:
    return Collection(
        source_dir=pair.local_path,
        read_only=False,
        on_write=strategy,
        git_strategy=strategy,
    )


class TestCollectionReportsThePush:
    def test_write_and_edit_report_pushed(self, git_repo_pair: GitRepoPair) -> None:
        col = _collection(git_repo_pair, _strategy(git_repo_pair))
        try:
            result = col.write("note.md", "hello\n")
            assert (result.created, result.pushed, result.push_error) == (
                True,
                True,
                None,
            )
            assert "note.md" in _remote_files(git_repo_pair)

            edited = col.edit("note.md", old_text="hello", new_text="hi")
            assert (edited.pushed, edited.push_error) == (True, None)
            remote_body = _run_git(git_repo_pair.remote_path, "show", "main:note.md")
            assert "hi" in remote_body
        finally:
            col.close()

    def test_failed_push_is_on_the_result(self, git_repo_pair: GitRepoPair) -> None:
        strategy = _strategy(git_repo_pair)
        col = _collection(git_repo_pair, strategy)
        try:
            col.write("first.md", "1\n")
            _run_git(
                git_repo_pair.local_path,
                "remote",
                "set-url",
                "origin",
                str(git_repo_pair.remote_path.parent / "gone.git"),
            )
            result = col.write("second.md", "2\n")
            assert result.pushed is False
            assert result.push_error
            # Saved and committed locally all the same.
            log = _run_git(git_repo_pair.local_path, "log", "--format=%s", "-1")
            assert "second.md" in log
        finally:
            strategy._closed = True
            col.close()

    def test_off_leaves_the_result_unset(self, git_repo_pair: GitRepoPair) -> None:
        col = _collection(git_repo_pair, _strategy(git_repo_pair, push_on_write=False))
        try:
            result = col.write("note.md", "hello\n")
            assert (result.pushed, result.push_error) == (None, None)
        finally:
            col.close()

    def test_close_pushes_before_the_embedding_flush(
        self, git_repo_pair: GitRepoPair
    ) -> None:
        strategy = _strategy(git_repo_pair, push_on_write=False)
        col = _collection(git_repo_pair, strategy)
        col.write("note.md", "hello\n")
        order: list[str] = []

        def flush_embeddings() -> None:
            order.append("embeddings")
            assert "note.md" in _remote_files(git_repo_pair)

        col._index_mgr.flush_dirty_embeddings = flush_embeddings  # type: ignore[method-assign]
        col.close()
        assert order == ["embeddings"]


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result.data, dict):
        return result.data
    return json.loads(result.content[0].text)


class TestWriteToolReportsThePush:
    async def test_write_tool_returns_pushed(
        self, git_repo_pair: GitRepoPair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markdown_vault_mcp.server import make_server

        for var in _CLEAR_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(
            "MARKDOWN_VAULT_MCP_SOURCE_DIR", str(git_repo_pair.local_path)
        )
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_READ_ONLY", "false")
        monkeypatch.setenv(
            "MARKDOWN_VAULT_MCP_GIT_REPO_URL", str(git_repo_pair.remote_path)
        )
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_GIT_PULL_INTERVAL_S", "0")
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_GIT_PUSH_ON_WRITE", "true")

        server = make_server()
        async with Client(server) as client:
            result = await client.call_tool(
                "write", {"path": "tool.md", "content": "via the tool\n"}
            )
            payload = _payload(result)
            # On the remote before the session (and its shutdown flush) ends.
            assert "tool.md" in _remote_files(git_repo_pair)

        assert payload["created"] is True
        assert payload["pushed"] is True
        assert payload["push_error"] is None


class TestConfig:
    def test_env_flag_reaches_the_strategy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markdown_vault_mcp.config import load_config

        monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(tmp_path))
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_GIT_TOKEN", "ghp_test")
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_GIT_PUSH_ON_WRITE", "true")
        config = load_config()
        assert config.git_push_on_write is True
        strategy = config.to_collection_kwargs()["on_write"]
        assert strategy.pushes_on_write is True

    def test_default_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from markdown_vault_mcp.config import load_config

        monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(tmp_path))
        monkeypatch.delenv("MARKDOWN_VAULT_MCP_GIT_PUSH_ON_WRITE", raising=False)
        assert load_config().git_push_on_write is False
