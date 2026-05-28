"""FastMCP middleware that sets per-request git author identity.

Two attribution sources, in priority order:

1. **Per-call ``participant_id`` arg** (honor system). When a write-tool
   call carries a ``participant_id`` argument, the middleware uses it
   to look up the git author in the YAML mapping. No auth check —
   matches the Agora substrate convention where the LLM is trusted to
   fill in its own participant id (see ``docs/plans/076`` in the
   ``physicalweb/agora`` repo for the design context).
2. **Auth0 subject** (fallback). When ``participant_id`` is absent,
   reads the validated OIDC access token from the request context and
   looks up the authenticated subject. This is the original mechanism
   (Plan 022 in agora).

Either way, the resolved identity sets the
:mod:`markdown_vault_mcp._author_context` for the duration of the
request. :mod:`markdown_vault_mcp.git`'s ``_stage_and_commit`` reads
the context and uses the identity as the commit's ``author`` field
(committer stays as the server identity per the docstring).

YAML mapping format (file path configured via
``MARKDOWN_VAULT_MCP_GIT_AUTHOR_MAPPING``):

.. code-block:: yaml

    mappings:
      # Human entries — keyed by OIDC subject
      - auth0_subject: "google-oauth2|xyz789"
        git_name: "Arnon Zangvil"
        git_email: "zangvil@gmail.com"
      # Persona entries — keyed by participant_id, honor-system trust
      - participant_id: "ki"
        git_name: "Ki"
        git_email: "ki@personas.agora.local"
      - participant_id: "lin"
        git_name: "Lín"
        git_email: "lin@personas.agora.local"
    default:  # optional
      git_name: "vault-bot"
      git_email: "bot@example.com"

An entry must have *either* ``auth0_subject`` *or* ``participant_id``,
not both. Malformed entries are skipped with a warning.

Resolution order in :meth:`on_call_tool`:

1. Read ``participant_id`` from tool args (if a write-tool call carries it)
2. If present + maps to a known persona — use that identity
3. Else fall back to Auth0 subject lookup
4. Else fall back to the ``default`` entry (if configured)
5. Else no-op (writes use the configured server identity)

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
        self._by_participant: dict[str, _Identity] = {}
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
            pid = entry.get("participant_id")
            name = entry.get("git_name")
            email = entry.get("git_email")
            if not (isinstance(name, str) and isinstance(email, str)):
                logger.warning(
                    "GitAuthorAttributionMiddleware: skipping malformed entry "
                    "(missing/non-string git_name or git_email): %r",
                    entry,
                )
                continue
            identity = _Identity(name=name, email=email)
            has_sub = isinstance(sub, str)
            has_pid = isinstance(pid, str)
            if has_sub and has_pid:
                logger.warning(
                    "GitAuthorAttributionMiddleware: entry has both "
                    "auth0_subject and participant_id; using both: %r",
                    entry,
                )
            if has_sub:
                self._by_subject[sub] = identity
            if has_pid:
                self._by_participant[pid] = identity
            if not (has_sub or has_pid):
                logger.warning(
                    "GitAuthorAttributionMiddleware: skipping entry without "
                    "auth0_subject or participant_id: %r",
                    entry,
                )

        default = raw.get("default")
        if isinstance(default, dict):
            name = default.get("git_name")
            email = default.get("git_email")
            if isinstance(name, str) and isinstance(email, str):
                self._default = _Identity(name=name, email=email)

        logger.info(
            "GitAuthorAttributionMiddleware: loaded %d subject + %d participant "
            "mapping(s)%s",
            len(self._by_subject),
            len(self._by_participant),
            " + default" if self._default is not None else "",
        )

    def _resolve_by_participant(self, participant_id: str | None) -> _Identity | None:
        """Honor-system lookup by ``participant_id`` claim. No auth check."""
        if not isinstance(participant_id, str):
            return None
        return self._by_participant.get(participant_id)

    def _resolve_by_subject(self) -> _Identity | None:
        """Auth0-backed lookup by OIDC ``sub`` claim."""
        token = get_access_token()
        if token is None:
            return None
        sub = token.claims.get("sub") if isinstance(token.claims, dict) else None
        if isinstance(sub, str):
            return self._by_subject.get(sub)
        return None

    def _extract_participant_id(
        self, context: MiddlewareContext[Any]
    ) -> str | None:
        """Read ``participant_id`` from the tool-call args, if present.

        FastMCP's ``on_call_tool`` middleware context carries the
        ``CallToolRequestParams`` as ``context.message``. The tool args
        live on ``message.arguments`` as a dict. Missing-args or
        non-dict-args returns ``None`` (no claim made).
        """
        msg = getattr(context, "message", None)
        if msg is None:
            return None
        args = getattr(msg, "arguments", None)
        if not isinstance(args, dict):
            return None
        value = args.get("participant_id")
        return value if isinstance(value, str) else None

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        # Resolution order:
        # 1. participant_id from tool args (honor system)
        # 2. Auth0 subject from validated OIDC token
        # 3. default entry from mapping (if configured)
        participant_id = self._extract_participant_id(context)
        identity = (
            self._resolve_by_participant(participant_id)
            or self._resolve_by_subject()
            or self._default
        )
        if identity is None:
            return await call_next(context)
        ctx_token = _author_context.set_author(identity.name, identity.email)
        try:
            return await call_next(context)
        finally:
            _author_context.reset(ctx_token)
