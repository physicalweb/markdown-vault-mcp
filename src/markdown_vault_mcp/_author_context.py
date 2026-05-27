"""Per-request git author identity context.

When markdown-vault-mcp runs behind OIDC auth with multiple users, every
commit lands with the server's single configured identity by default.
This module provides a :mod:`contextvars`-based mechanism for the
request-handling layer (a FastMCP middleware) to set a per-request git
author identity that :mod:`markdown_vault_mcp.git`'s commit code reads.

Committer (used for the push credential + git's ``committer`` field)
stays as the configured server identity. Only the ``author`` field is
overridden — standard git distinction for "authored by X, committed by
Y" workflows (shared push credential, real human author).

When no author is set, ``_stage_and_commit`` falls back to the
configured identity for both committer and author (today's behavior;
fully backward-compatible).
"""

from __future__ import annotations

import contextvars

_current_author: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "markdown_vault_mcp_current_author", default=None
)


def set_author(name: str, email: str) -> contextvars.Token[tuple[str, str] | None]:
    """Set the per-request git author identity.

    Returns the contextvar token so the caller can reset it after the
    request (recommended pattern: ``token = set_author(...); try: ...; finally: reset(token)``).
    """
    return _current_author.set((name, email))


def get_author() -> tuple[str, str] | None:
    """Read the per-request git author identity, or ``None`` if unset."""
    return _current_author.get()


def reset(token: contextvars.Token[tuple[str, str] | None]) -> None:
    """Reset the per-request git author identity to its prior value."""
    _current_author.reset(token)
