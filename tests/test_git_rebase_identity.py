"""A pull that must replay a local commit works with no git identity configured.

AGO-756 (agora): the Docker image has no ``user.name`` / ``user.email`` (the runtime user's
GECOS is empty), and the strategy passes its identity only to ``git commit`` (``-c user.*``).
``git rebase`` replays commits as a new committer, so it died with "empty ident name", and
``force_pull`` / ``sync_once`` reported ``conflict_resolution_failed`` for a divergence with no
overlapping file: the clone stayed wedged until its commits were made vacuous upstream.
The ``git_repo_pair`` fixture configures an identity in the clone, which is why
``test_force_pull_rebased_when_divergent_on_different_files`` never saw it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.fixtures.git import _run_git
from tests.test_git_force_methods import _seed_remote_commit

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.git import GitRepoPair


@pytest.fixture
def no_git_identity(
    git_repo_pair: GitRepoPair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> GitRepoPair:
    """Strip every identity source, as in the container: no config, no env, no guessing."""
    _run_git(git_repo_pair.local_path, "config", "--unset", "user.name")
    _run_git(git_repo_pair.local_path, "config", "--unset", "user.email")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # The host's passwd entry would let git guess an identity; the container's cannot.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    for var in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    return git_repo_pair


def _local_write(pair: GitRepoPair, strategy: object) -> None:
    """A write through the strategy: committed with its ``-c user.*`` identity, not pushed."""
    path = pair.local_path / "local_only.md"
    path.write_text("local\n")
    strategy(path, "local\n", "write")  # type: ignore[operator]


def test_force_pull_rebases_with_no_identity(no_git_identity: GitRepoPair) -> None:
    from markdown_vault_mcp.git import PULL_REASON_REBASED, GitWriteStrategy

    pair = no_git_identity
    strategy = GitWriteStrategy(
        enable_pull=True,
        enable_push=False,
        repo_path=pair.local_path,
        commit_name="vault-bot",
        commit_email="vault-bot@example.com",
    )
    _local_write(pair, strategy)
    _seed_remote_commit(
        pair, clone_name="foreign", file_name="remote_only.md", body="remote\n"
    )

    result = strategy.force_pull()

    assert result.applied is True, result
    assert result.reason == PULL_REASON_REBASED
    assert (pair.local_path / "remote_only.md").exists()
    assert (pair.local_path / "local_only.md").exists()
    committer = _run_git(pair.local_path, "log", "-1", "--format=%cn <%ce>").strip()
    assert committer == "vault-bot <vault-bot@example.com>"


def test_sync_once_rebases_with_no_identity(no_git_identity: GitRepoPair) -> None:
    from markdown_vault_mcp.git import GitWriteStrategy

    pair = no_git_identity
    strategy = GitWriteStrategy(
        enable_pull=True,
        enable_push=False,
        repo_path=pair.local_path,
        commit_name="vault-bot",
        commit_email="vault-bot@example.com",
    )
    _local_write(pair, strategy)
    _seed_remote_commit(
        pair, clone_name="foreign", file_name="remote_only.md", body="remote\n"
    )

    assert strategy.sync_once(pair.local_path) is True
    assert (pair.local_path / "remote_only.md").exists()
    ahead = _run_git(pair.local_path, "rev-list", "--count", "@{upstream}..HEAD")
    assert ahead.strip() == "1"
