"""Tests for the GitHub webhook receiver — HMAC verification + handler logic.

The handler's pull mechanism (``_trigger_pull_and_reindex``) is exercised
in integration via the live ECS environment; here we cover the request-side
logic in isolation:

- HMAC signature verification (valid, invalid, missing, malformed)
- Event dispatch (ping vs push vs other)
- Collection-not-ready degradation
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from markdown_vault_mcp._github_webhook import (
    make_webhook_handler,
    verify_signature,
)

SECRET = "test-secret"


def _signature_for(payload: bytes, secret: str = SECRET) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


def test_verify_signature_accepts_valid_hmac() -> None:
    payload = b'{"event":"test"}'
    sig = _signature_for(payload)
    assert verify_signature(payload, sig, SECRET) is True


def test_verify_signature_rejects_wrong_secret() -> None:
    payload = b'{"event":"test"}'
    sig = _signature_for(payload, secret="wrong-secret")
    assert verify_signature(payload, sig, SECRET) is False


def test_verify_signature_rejects_tampered_payload() -> None:
    payload = b'{"event":"test"}'
    sig = _signature_for(payload)
    tampered = b'{"event":"tampered"}'
    assert verify_signature(tampered, sig, SECRET) is False


def test_verify_signature_rejects_missing_header() -> None:
    assert verify_signature(b"x", None, SECRET) is False
    assert verify_signature(b"x", "", SECRET) is False


def test_verify_signature_rejects_malformed_prefix() -> None:
    payload = b'{"event":"test"}'
    raw_digest = hmac.new(
        SECRET.encode(),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    # Missing "sha256=" prefix
    assert verify_signature(payload, raw_digest, SECRET) is False
    # Wrong prefix
    assert verify_signature(payload, f"sha1={raw_digest}", SECRET) is False


# ---------------------------------------------------------------------------
# Handler — event dispatch
# ---------------------------------------------------------------------------


def _mock_request(
    body: bytes,
    *,
    signature: str | None,
    event: str,
) -> Request:
    """Build a Starlette Request stub with body + headers."""
    req = MagicMock(spec=Request)
    req.body = AsyncMock(return_value=body)
    headers: dict[str, str] = {"X-GitHub-Event": event}
    if signature is not None:
        headers["X-Hub-Signature-256"] = signature
    req.headers = headers
    return req


@pytest.mark.asyncio
async def test_handler_rejects_invalid_signature() -> None:
    handler = make_webhook_handler(collection_getter=lambda: None, secret=SECRET)
    payload = b'{"event":"push"}'
    req = _mock_request(payload, signature="sha256=invalid", event="push")
    response = await handler(req)
    assert response.status_code == 401
    body = json.loads(response.body)
    assert body == {"error": "invalid signature"}


@pytest.mark.asyncio
async def test_handler_handles_ping_event() -> None:
    handler = make_webhook_handler(collection_getter=lambda: None, secret=SECRET)
    payload = b'{"zen":"go fast"}'
    req = _mock_request(payload, signature=_signature_for(payload), event="ping")
    response = await handler(req)
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body == {"ok": True, "event": "ping"}


@pytest.mark.asyncio
async def test_handler_ignores_non_push_events() -> None:
    handler = make_webhook_handler(collection_getter=lambda: None, secret=SECRET)
    payload = b'{"action":"opened"}'
    req = _mock_request(
        payload, signature=_signature_for(payload), event="pull_request"
    )
    response = await handler(req)
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["pulled"] is False
    assert body["event"] == "pull_request"


@pytest.mark.asyncio
async def test_handler_handles_collection_not_ready() -> None:
    def _raise_not_ready() -> Any:
        raise RuntimeError("Collection not initialised")

    handler = make_webhook_handler(
        collection_getter=_raise_not_ready, secret=SECRET
    )
    payload = b'{"pusher":{"name":"x"},"ref":"refs/heads/main","head_commit":{"id":"deadbeef"}}'
    req = _mock_request(payload, signature=_signature_for(payload), event="push")
    response = await handler(req)
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["pulled"] is False
    assert body["reason"] == "collection_not_ready"


@pytest.mark.asyncio
async def test_handler_triggers_pull_on_push_event() -> None:
    # Mock collection with a GitWriteStrategy
    from markdown_vault_mcp.git import GitWriteStrategy, PullResult

    collection = MagicMock()
    strategy = MagicMock(spec=GitWriteStrategy)
    pull_result = PullResult(
        applied=True,
        fast_forward=True,
        commits_pulled=1,
        from_sha="aaaa",
        to_sha="bbbb",
        reason=None,
        conflict_files=(),
    )
    strategy.force_pull = MagicMock(return_value=pull_result)
    collection._git_strategy = strategy
    collection.reindex = MagicMock()

    handler = make_webhook_handler(
        collection_getter=lambda: collection, secret=SECRET
    )
    payload = b'{"pusher":{"name":"arnon"},"ref":"refs/heads/main","head_commit":{"id":"bbbb1234"}}'
    req = _mock_request(payload, signature=_signature_for(payload), event="push")
    response = await handler(req)
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["pulled"] is True
    assert body["applied"] is True
    assert body["head_moved"] is True
    strategy.force_pull.assert_called_once_with(dry_run=False)
    collection.reindex.assert_called_once()
