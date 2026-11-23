"""The HTTP server's event store, over real handshakes (AGO-1017).

The MCP SDK keys event streams by JSON-RPC request id across all sessions, so
every session's ``initialize`` (id 0) lands in one stream list.  Once that list
holds more than ``max_events_per_stream`` (100) events, each ``initialize``
trims it, and two concurrent trims race to delete the same oldest file
(``FileNotFoundError`` in ``FileTreeStore``), failing the handshake.

``serve`` therefore builds no event store unless
``MARKDOWN_VAULT_MCP_EVENT_STORE_URL`` asks for one.  These tests run the app
``serve`` builds under uvicorn and send it more than 100 handshakes.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import uvicorn
from fastmcp import FastMCP

from markdown_vault_mcp._cli_impl import _build_parser, _cmd_serve

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest

# More than max_events_per_stream (100), so a store would have to trim.
HANDSHAKES = 120
PROTOCOL = "2025-11-25"  # the version whose initialize also stores a priming event
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _tiny_server(**_: Any) -> FastMCP:
    server: FastMCP = FastMCP("event-store-test")

    @server.tool
    def echo(text: str) -> str:
        """Return the text."""
        return text

    return server


def _build_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str | None) -> Any:
    """Run ``_cmd_serve`` up to uvicorn and return the ASGI app it built."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(vault))
    if url is None:
        monkeypatch.delenv("MARKDOWN_VAULT_MCP_EVENT_STORE_URL", raising=False)
    else:
        monkeypatch.setenv("MARKDOWN_VAULT_MCP_EVENT_STORE_URL", url)
    captured: dict[str, Any] = {}
    with (
        patch(
            "fastmcp_pvl_core._factory._DEFAULT_EVENT_STORE_DIR",
            str(tmp_path / "default-events"),
        ),
        patch("markdown_vault_mcp.server.make_server", side_effect=_tiny_server),
        patch("uvicorn.run", side_effect=lambda app, **_: captured.update(app=app)),
    ):
        _cmd_serve(_build_parser().parse_args(["serve", "--transport", "http"]))
    return captured["app"]


@contextmanager
def _serving(app: Any) -> Iterator[str]:
    """Serve *app* on a free local port in a thread; yield its MCP URL."""
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            lifespan="on",
            log_level="warning",
            ws="none",
            timeout_graceful_shutdown=1,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _sse(body: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Split an SSE body into its event ids and its JSON messages."""
    ids = [ln[3:].strip() for ln in body.splitlines() if ln.startswith("id:")]
    msgs = [
        json.loads(ln[5:])
        for ln in body.splitlines()
        if ln.startswith("data:") and ln[5:].strip()
    ]
    return ids, msgs


def _initialize(url: str) -> tuple[str | None, list[str], list[dict[str, Any]]]:
    """One handshake.  Returns (session id, SSE event ids, messages)."""
    body = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
    }
    try:
        r = httpx.post(url, json=body, headers=HEADERS, timeout=2)
    except httpx.TimeoutException:
        return None, [], []
    ids, msgs = _sse(r.text)
    return r.headers.get("mcp-session-id"), ids, msgs


def _answered(msgs: list[dict[str, Any]]) -> bool:
    return any(m.get("id") == 0 and "result" in m for m in msgs)


def _handshakes(
    url: str, workers: int
) -> list[tuple[str | None, list[str], list[dict[str, Any]]]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda _: _initialize(url), range(HANDSHAKES)))


def test_unset_url_serves_handshakes_with_no_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset: >100 concurrent handshakes all answer, none primed, nothing on disk."""
    app = _build_app(monkeypatch, tmp_path, None)
    with _serving(app) as url:
        results = _handshakes(url, workers=16)

    assert all(_answered(msgs) for _, _, msgs in results)
    # No event store: no priming event, no event ids on any stream.
    assert all(ids == [] for _, ids, _ in results)
    assert not (tmp_path / "default-events").exists()


def test_unset_url_tool_call_and_last_event_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset: a tool call answers; a Last-Event-ID is never replayed.

    With no store the server issues no event id, so a client (the TS SDK sends
    ``last-event-id`` only after it has received an ``id:``) has nothing to
    resume from.  A token minted by an earlier task arrives with that task's
    session id, which this server answers 404 before any replay.  A client that
    forges one on a live session gets mcp 1.26's no-store ``_replay_events``,
    which returns without a response (uvicorn answers 500); the session is
    unharmed.
    """
    app = _build_app(monkeypatch, tmp_path, None)
    with _serving(app) as url:
        sid, _, msgs = _initialize(url)
        assert sid and _answered(msgs)
        session = {**HEADERS, "mcp-session-id": sid, "mcp-protocol-version": PROTOCOL}
        r = httpx.post(
            url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=session,
            timeout=5,
        )
        assert r.status_code == 202
        r = httpx.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hi"}},
            },
            headers=session,
            timeout=5,
        )
        ids, msgs = _sse(r.text)
        assert ids == []
        assert msgs[-1]["result"]["content"][0]["text"] == "hi"

        def resume(session_id: str) -> httpx.Response:
            return httpx.get(
                url,
                headers={
                    "Accept": "text/event-stream",
                    "mcp-session-id": session_id,
                    "mcp-protocol-version": PROTOCOL,
                    "Last-Event-ID": "an-id-this-server-never-issued",
                },
                timeout=5,
            )

        assert resume("a-session-of-an-earlier-task").status_code == 404
        assert resume(sid).status_code == 500
        r = httpx.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "still"}},
            },
            headers=session,
            timeout=5,
        )
        assert _sse(r.text)[1][-1]["result"]["content"][0]["text"] == "still"


def test_set_url_keeps_the_store_and_reaches_its_trim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Set: the file store is built as before, and >100 handshakes make it trim.

    The handshakes run one at a time, so the stream list grows by at least one
    id per handshake and must pass 100.  The trim is where the fault lives; that
    a handshake *fails* there depends on two stores interleaving, so it is
    printed (run with ``-s``), not asserted.
    """
    events = tmp_path / "events"
    app = _build_app(monkeypatch, tmp_path, f"file://{events}")
    with _serving(app) as url:
        results = _handshakes(url, workers=1)

    answered = sum(_answered(msgs) for _, _, msgs in results)
    print(f"store on: {answered}/{HANDSHAKES} handshakes answered")
    # The store is in use: answered handshakes carry a priming event id.
    assert any(ids for _, ids, _ in results)
    # Two events stored per handshake; the trim deleted the oldest.
    stored = list((events / "fastmcp_events").glob("*.json"))
    assert 0 < len(stored) < 2 * HANDSHAKES
