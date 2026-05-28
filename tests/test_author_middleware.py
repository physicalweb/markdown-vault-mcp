"""Tests for ``GitAuthorAttributionMiddleware``.

Covers the two attribution sources:
1. ``participant_id`` arg from the tool call (honor-system; primary)
2. Auth0 OIDC subject from the validated access token (fallback)

Plus mapping-file loading edge cases (entries with auth0_subject vs
participant_id, malformed entries, default fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from markdown_vault_mcp import _author_context
from markdown_vault_mcp._author_middleware import (
    GitAuthorAttributionMiddleware,
    _Identity,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------------------


def _write_mapping(tmp_path: Path, content: str) -> Path:
    """Helper: write a YAML mapping file under ``tmp_path``."""
    p = tmp_path / "mapping.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_accepts_auth0_subject_entries(tmp_path: Path) -> None:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - auth0_subject: "google-oauth2|abc"
    git_name: "Arnon Zangvil"
    git_email: "zangvil@gmail.com"
""",
    )
    mw = GitAuthorAttributionMiddleware(mapping)
    assert mw._by_subject == {
        "google-oauth2|abc": _Identity("Arnon Zangvil", "zangvil@gmail.com")
    }
    assert mw._by_participant == {}


def test_load_accepts_participant_id_entries(tmp_path: Path) -> None:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
  - participant_id: "lin"
    git_name: "Lín"
    git_email: "lin@personas.agora.local"
""",
    )
    mw = GitAuthorAttributionMiddleware(mapping)
    assert mw._by_participant == {
        "ki": _Identity("Ki", "ki@personas.agora.local"),
        "lin": _Identity("Lín", "lin@personas.agora.local"),
    }
    assert mw._by_subject == {}


def test_load_mixed_entries(tmp_path: Path) -> None:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - auth0_subject: "google-oauth2|abc"
    git_name: "Arnon Zangvil"
    git_email: "zangvil@gmail.com"
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
default:
  git_name: "vault-bot"
  git_email: "bot@example.com"
""",
    )
    mw = GitAuthorAttributionMiddleware(mapping)
    assert "google-oauth2|abc" in mw._by_subject
    assert "ki" in mw._by_participant
    assert mw._default == _Identity("vault-bot", "bot@example.com")


def test_load_skips_malformed_entries(tmp_path: Path) -> None:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - git_name: "no-key"
    git_email: "no@example.com"
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
""",
    )
    mw = GitAuthorAttributionMiddleware(mapping)
    assert mw._by_participant == {"ki": _Identity("Ki", "ki@personas.agora.local")}
    assert mw._by_subject == {}


def test_load_no_mapping_path_is_noop() -> None:
    mw = GitAuthorAttributionMiddleware(None)
    assert mw._by_subject == {}
    assert mw._by_participant == {}
    assert mw._default is None


# ---------------------------------------------------------------------------
# Resolution priority — participant_id beats Auth0 subject beats default
# ---------------------------------------------------------------------------


def _make_context(args: dict[str, Any] | None) -> Any:
    """Build a minimal middleware-context stub exposing ``.message.arguments``."""
    msg = MagicMock()
    msg.arguments = args
    ctx = MagicMock()
    ctx.message = msg
    return ctx


@pytest.fixture
def mw(tmp_path: Path) -> GitAuthorAttributionMiddleware:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - auth0_subject: "google-oauth2|arnon-sub"
    git_name: "Arnon Zangvil"
    git_email: "zangvil@gmail.com"
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
default:
  git_name: "vault-bot"
  git_email: "bot@example.com"
""",
    )
    return GitAuthorAttributionMiddleware(mapping)


def test_participant_id_wins_over_auth0_subject(
    mw: GitAuthorAttributionMiddleware,
) -> None:
    """When both arg + token resolve, participant_id wins (honor system)."""
    ctx = _make_context({"participant_id": "ki", "path": "x.md"})
    token = MagicMock(claims={"sub": "google-oauth2|arnon-sub"})
    with patch(
        "markdown_vault_mcp._author_middleware.get_access_token",
        return_value=token,
    ):
        # Drive on_call_tool to observe which contextvar gets set
        observed: dict[str, tuple[str, str] | None] = {}

        async def call_next(_context: Any) -> str:
            observed["author"] = _author_context.get_author()
            return "ok"

        import asyncio

        asyncio.run(mw.on_call_tool(ctx, call_next))
        assert observed["author"] == ("Ki", "ki@personas.agora.local")


def test_auth0_subject_used_when_no_participant_id(
    mw: GitAuthorAttributionMiddleware,
) -> None:
    ctx = _make_context({"path": "x.md"})
    token = MagicMock(claims={"sub": "google-oauth2|arnon-sub"})
    with patch(
        "markdown_vault_mcp._author_middleware.get_access_token",
        return_value=token,
    ):
        observed: dict[str, tuple[str, str] | None] = {}

        async def call_next(_context: Any) -> str:
            observed["author"] = _author_context.get_author()
            return "ok"

        import asyncio

        asyncio.run(mw.on_call_tool(ctx, call_next))
        assert observed["author"] == ("Arnon Zangvil", "zangvil@gmail.com")


def test_default_used_when_neither_resolves(
    mw: GitAuthorAttributionMiddleware,
) -> None:
    ctx = _make_context({"path": "x.md"})  # no participant_id
    with patch(
        "markdown_vault_mcp._author_middleware.get_access_token",
        return_value=None,  # no token either
    ):
        observed: dict[str, tuple[str, str] | None] = {}

        async def call_next(_context: Any) -> str:
            observed["author"] = _author_context.get_author()
            return "ok"

        import asyncio

        asyncio.run(mw.on_call_tool(ctx, call_next))
        assert observed["author"] == ("vault-bot", "bot@example.com")


def test_unknown_participant_id_falls_back_to_auth0(
    mw: GitAuthorAttributionMiddleware,
) -> None:
    """Honor-system: an unknown participant_id falls through to Auth0 lookup."""
    ctx = _make_context({"participant_id": "unknown-persona", "path": "x.md"})
    token = MagicMock(claims={"sub": "google-oauth2|arnon-sub"})
    with patch(
        "markdown_vault_mcp._author_middleware.get_access_token",
        return_value=token,
    ):
        observed: dict[str, tuple[str, str] | None] = {}

        async def call_next(_context: Any) -> str:
            observed["author"] = _author_context.get_author()
            return "ok"

        import asyncio

        asyncio.run(mw.on_call_tool(ctx, call_next))
        assert observed["author"] == ("Arnon Zangvil", "zangvil@gmail.com")


def test_no_args_no_token_no_default_is_noop(tmp_path: Path) -> None:
    mapping = _write_mapping(
        tmp_path,
        """
mappings:
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
""",
    )
    mw = GitAuthorAttributionMiddleware(mapping)
    ctx = _make_context(None)
    with patch(
        "markdown_vault_mcp._author_middleware.get_access_token",
        return_value=None,
    ):
        observed: dict[str, tuple[str, str] | None] = {"author": ("sentinel", "x")}

        async def call_next(_context: Any) -> str:
            observed["author"] = _author_context.get_author()
            return "ok"

        import asyncio

        asyncio.run(mw.on_call_tool(ctx, call_next))
        # No identity set; observed stays at the prior contextvar value (None)
        assert observed["author"] is None
