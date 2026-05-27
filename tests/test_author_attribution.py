"""Tests for per-request git author attribution.

Fork-only feature (not upstream). Verifies that:
1. ``_stage_and_commit`` uses ``--author`` when a contextvar is set
2. ``_stage_and_commit`` falls back to the configured identity when no
   contextvar is set (backward-compatible default)
3. ``_author_context`` set/get/reset round-trip works
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from markdown_vault_mcp import _author_context
from markdown_vault_mcp.git import _stage_and_commit


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A fresh git repo with one initial commit + one tracked file."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "init-user"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "init@example.com"],
        check=True,
    )
    initial = tmp_path / "seed.md"
    initial.write_text("# seed\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "seed.md"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"],
        check=True,
    )
    return tmp_path


def _last_commit_field(git_root: Path, field: str) -> str:
    """Read a field of the last commit (``aN``/``aE``/``cN``/``cE``)."""
    result = subprocess.run(
        ["git", "-C", str(git_root), "log", "-1", f"--format=%{field}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_stage_and_commit_uses_configured_identity_when_no_author_set(
    git_repo: Path,
) -> None:
    """Backward compat: with no contextvar set, both author + committer
    are the configured identity."""
    (git_repo / "note.md").write_text("content\n", encoding="utf-8")
    _stage_and_commit(
        git_root=git_repo,
        path=git_repo / "note.md",
        operation="write",
        commit_name="vault-bot",
        commit_email="bot@example.com",
    )
    assert _last_commit_field(git_repo, "aN") == "vault-bot"
    assert _last_commit_field(git_repo, "aE") == "bot@example.com"
    assert _last_commit_field(git_repo, "cN") == "vault-bot"
    assert _last_commit_field(git_repo, "cE") == "bot@example.com"


def test_stage_and_commit_uses_author_override_when_contextvar_set(
    git_repo: Path,
) -> None:
    """When the contextvar is set, author reflects the human; committer
    stays as the configured bot identity."""
    (git_repo / "handoff.md").write_text("# from lin\n", encoding="utf-8")
    token = _author_context.set_author("Lín", "lin@example.com")
    try:
        _stage_and_commit(
            git_root=git_repo,
            path=git_repo / "handoff.md",
            operation="write",
            commit_name="vault-bot",
            commit_email="bot@example.com",
        )
    finally:
        _author_context.reset(token)
    assert _last_commit_field(git_repo, "aN") == "Lín"
    assert _last_commit_field(git_repo, "aE") == "lin@example.com"
    assert _last_commit_field(git_repo, "cN") == "vault-bot"
    assert _last_commit_field(git_repo, "cE") == "bot@example.com"


def test_author_context_get_returns_none_by_default() -> None:
    """Fresh contextvar reads as None (no author set)."""
    assert _author_context.get_author() is None


def test_author_context_reset_restores_prior_value() -> None:
    """Set then reset returns to prior value."""
    assert _author_context.get_author() is None
    token = _author_context.set_author("Ki", "ki@example.com")
    try:
        assert _author_context.get_author() == ("Ki", "ki@example.com")
    finally:
        _author_context.reset(token)
    assert _author_context.get_author() is None
