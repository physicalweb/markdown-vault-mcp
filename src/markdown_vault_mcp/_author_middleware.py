"""FastMCP middleware that sets per-request git author identity.

Reads the validated OIDC access token from the request context, looks
up the authenticated subject in a YAML mapping file, and sets the
:mod:`markdown_vault_mcp._author_context` for the duration of the
request. :mod:`markdown_vault_mcp.git`'s ``_stage_and_commit`` reads
the context and uses the identity as the commit's ``author`` field
(committer stays as the server identity per the docstring).

YAML mapping format (file path configured via
``MARKDOWN_VAULT_MCP_GIT_AUTHOR_MAPPING``):

.. code-block:: yaml

    mappings:
      - auth0_subject: "auth0|abc123"
        git_name: "Lín"
        git_email: "lin@example.com"
      - auth0_subject: "google-oauth2|xyz789"
        git_name: "Ki"
        git_email: "ki@example.com"
    default:  # optional
      git_name: "vault-bot"
      git_email: "bot@example.com"

Tokens whose ``sub`` claim doesn't match any mapping fall through to
the configured server identity (no author override). To attribute
unmapped subjects to a default identity, add a ``default`` entry.

Environment integration:
- ``MARKDOWN_VAULT_MCP_GIT_AUTHOR_MAPPING`` — path to the YAML file.
  When unset, the middleware is a no-op (writes still use the
  configured server identity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from . import _author_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Identity:
    name: str
    email: str


class GitAuthorAttributionMiddleware(Middleware):
    """Set the per-request git author from the authenticated OIDC subject.

    Hooks ``on_call_tool`` (the write-tool surface) and ``on_message``
    (fallback for any other write path). Reads the validated access
    token, maps ``claims['sub']`` to a ``(name, email)`` pair via the
    YAML mapping, and sets the contextvar that ``_stage_and_commit``
    reads.

    When no token is present, no mapping is configured, or the subject
    isn't in the mapping (and no default is set), the middleware is a
    no-op — writes use the configured server identity.

    The mapping file is loaded once at construction time. Live-reload
    is intentionally not supported; a deploy roll restarts the
    container.
    """

    def __init__(self, mapping_path: str | Path | None) -> None:
        super().__init__()
        self._by_subject: dict[str, _Identity] = {}
        self._default: _Identity | None = None
        if mapping_path is None:
            logger.info(
                "GitAuthorAttributionMiddleware: no mapping path configured; "
                "all commits will use the configured server identity."
            )
            return
        self._load(Path(mapping_path))

    def _load(self, path: Path) -> None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning(
                "GitAuthorAttributionMiddleware: mapping file %s not found; "
                "all commits will use the configured server identity.",
                path,
            )
            return
        except Exception as e:
            logger.error(
                "GitAuthorAttributionMiddleware: failed to load %s: %s; "
                "all commits will use the configured server identity.",
                path,
                e,
            )
            return

        if not isinstance(raw, dict):
            logger.error(
                "GitAuthorAttributionMiddleware: %s root must be a mapping",
                path,
            )
            return

        mappings = raw.get("mappings", [])
        if not isinstance(mappings, list):
            logger.error(
                "GitAuthorAttributionMiddleware: %s 'mappings' must be a list",
                path,
            )
            return

        for entry in mappings:
            if not isinstance(entry, dict):
                continue
            sub = entry.get("auth0_subject")
            name = entry.get("git_name")
            email = entry.get("git_email")
            if not (isinstance(sub, str) and isinstance(name, str) and isinstance(email, str)):
                logger.warning(
                    "GitAuthorAttributionMiddleware: skipping malformed entry %r",
                    entry,
                )
                continue
            self._by_subject[sub] = _Identity(name=name, email=email)

        default = raw.get("default")
        if isinstance(default, dict):
            name = default.get("git_name")
            email = default.get("git_email")
            if isinstance(name, str) and isinstance(email, str):
                self._default = _Identity(name=name, email=email)

        logger.info(
            "GitAuthorAttributionMiddleware: loaded %d subject mapping(s)%s",
            len(self._by_subject),
            " + default" if self._default is not None else "",
        )

    def _resolve(self) -> _Identity | None:
        token = get_access_token()
        if token is None:
            return self._default
        sub = token.claims.get("sub") if isinstance(token.claims, dict) else None
        if isinstance(sub, str):
            identity = self._by_subject.get(sub)
            if identity is not None:
                return identity
        return self._default

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        identity = self._resolve()
        if identity is None:
            return await call_next(context)
        ctx_token = _author_context.set_author(identity.name, identity.email)
        try:
            return await call_next(context)
        finally:
            _author_context.reset(ctx_token)
