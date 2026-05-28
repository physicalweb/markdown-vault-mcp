"""GitHub webhook receiver — pull-on-push for multi-author handoffs.

Fork-only feature (not upstream). Closes the read-side staleness paper-cut
where the local clone only refreshed on the periodic pull interval (default
600s) or when a user explicitly called ``git_sync``. With this module wired,
any push to the upstream repo triggers an immediate ``force_pull`` —
strictly faster than waiting for the periodic interval.

Endpoint shape::

    POST /github-webhook
    X-Hub-Signature-256: sha256=<hex digest>
    X-GitHub-Event: <event-type>
    <JSON payload>

Validation:
- HMAC-SHA256 of the raw payload against ``MARKDOWN_VAULT_MCP_GITHUB_WEBHOOK_SECRET``
- Constant-time comparison via :func:`hmac.compare_digest`
- Rejects: missing signature header, malformed prefix, signature mismatch

Action on valid ``push`` event:
- Calls :meth:`GitWriteStrategy.force_pull` (off the event loop via ``asyncio.to_thread``)
- Reindexes the collection if HEAD moved

Non-push events (ping, etc.) get acknowledged with 200 but no pull. Ping
events are the GitHub-side verification that the webhook is reachable.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, Response

from markdown_vault_mcp.git import GitWriteStrategy

if TYPE_CHECKING:
    from starlette.requests import Request

    from markdown_vault_mcp.collection import Collection

logger = logging.getLogger(__name__)

_SIGNATURE_HEADER = "X-Hub-Signature-256"
_EVENT_HEADER = "X-GitHub-Event"
_SIGNATURE_PREFIX = "sha256="


def verify_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify GitHub's ``X-Hub-Signature-256`` header against ``secret``.

    Returns True iff the header is well-formed and the HMAC-SHA256 of the
    payload matches the digest in the header. Uses :func:`hmac.compare_digest`
    for constant-time comparison.
    """
    if not signature_header:
        return False
    if not signature_header.startswith(_SIGNATURE_PREFIX):
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    received = signature_header[len(_SIGNATURE_PREFIX):]
    return hmac.compare_digest(expected, received)


async def _trigger_pull_and_reindex(collection: Collection) -> dict[str, Any]:
    """Run force_pull + (conditionally) reindex against the collection.

    Mirrors the in-tool ``_run_pull_leg`` flow in ``_server_tools.py``: pull,
    then reindex iff HEAD moved. Returns a small dict for logging / response.
    """
    strategy = collection._git_strategy
    if strategy is None or not isinstance(strategy, GitWriteStrategy):
        logger.warning(
            "github_webhook: no managed git strategy on collection; "
            "skipping pull (route should not be mounted in this mode)"
        )
        return {"pulled": False, "reason": "no_git_strategy"}

    pull_result = await asyncio.to_thread(strategy.force_pull, dry_run=False)
    head_moved = (
        pull_result.applied and pull_result.from_sha != pull_result.to_sha
    )
    payload: dict[str, Any] = {
        "pulled": True,
        "applied": pull_result.applied,
        "from_sha": pull_result.from_sha,
        "to_sha": pull_result.to_sha,
        "head_moved": head_moved,
    }
    if head_moved:
        try:
            await asyncio.to_thread(collection.reindex)
            payload["reindexed"] = True
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("github_webhook: reindex failed: %s", e)
            payload["reindexed"] = False
            payload["reindex_error"] = str(e)
    return payload


def make_webhook_handler(collection_getter, secret: str):
    """Build the Starlette webhook handler bound to ``collection_getter`` + ``secret``.

    The ``collection_getter`` is a zero-arg callable that returns the active
    ``Collection``. We can't capture the collection at module-import time
    because the lifespan builds it lazily — but we can capture a getter that
    closes over the resolved-at-call-time reference.
    """

    async def handler(request: Request) -> Response:
        # Read raw bytes for HMAC verification — must use raw bytes, not
        # decoded text, because the signature is over the wire bytes.
        payload = await request.body()
        signature = request.headers.get(_SIGNATURE_HEADER)
        if not verify_signature(payload, signature, secret):
            logger.warning(
                "github_webhook: signature verification failed "
                "(header_present=%s, payload_bytes=%d)",
                signature is not None,
                len(payload),
            )
            return JSONResponse({"error": "invalid signature"}, status_code=401)

        event = request.headers.get(_EVENT_HEADER, "unknown")

        # Ping events are GitHub's "I can reach you" probe — ack with 200,
        # don't pull (there's nothing to pull yet).
        if event == "ping":
            logger.info("github_webhook: ping received; route is live")
            return JSONResponse({"ok": True, "event": "ping"})

        # Push events trigger the pull. Other events (issues, PRs, etc.)
        # are not configured by our GitHub-side setup, but if they arrive
        # we just 200-ack without pulling.
        if event != "push":
            logger.info(
                "github_webhook: non-push event %r — acknowledged, no pull",
                event,
            )
            return JSONResponse({"ok": True, "event": event, "pulled": False})

        # Parse the payload for logging — failure to parse doesn't block
        # the pull, since the pull doesn't depend on payload content.
        try:
            payload_data = json.loads(payload)
            pusher = payload_data.get("pusher", {}).get("name", "unknown")
            ref = payload_data.get("ref", "unknown")
            head_commit = payload_data.get("head_commit", {})
            commit_id = head_commit.get("id", "unknown")[:7] if head_commit else "unknown"
            logger.info(
                "github_webhook: push received from %s on %s @ %s",
                pusher,
                ref,
                commit_id,
            )
        except Exception:
            logger.warning("github_webhook: failed to parse payload for logging")

        try:
            collection = collection_getter()
        except RuntimeError:
            logger.warning(
                "github_webhook: collection not yet built; deferring pull"
            )
            return JSONResponse(
                {"ok": True, "event": "push", "pulled": False, "reason": "collection_not_ready"}
            )

        result = await _trigger_pull_and_reindex(collection)
        logger.info("github_webhook: pull result %s", result)
        return JSONResponse({"ok": True, "event": "push", **result})

    return handler
