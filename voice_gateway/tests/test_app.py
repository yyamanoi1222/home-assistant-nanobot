"""Tests for the aiohttp application."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from aiohttp.test_utils import TestClient, TestServer
from nanobot_voice_gateway.app import MANAGER_KEY, create_app
from nanobot_voice_gateway.config import GatewayConfig

AUTH = {"Authorization": "Bearer secret"}


class FakeSessions:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def export(self, session_key: str) -> Any:
        self.calls.append(("export", session_key))
        return {"history": []}

    def clear(self, session_key: str) -> None:
        self.calls.append(("clear", session_key))

    async def restore(self, snapshot: Any, session_key: str) -> None:
        self.calls.append(("restore", session_key, snapshot))


class FakeRun:
    def __init__(self, events: list[Any], *, block: bool = False) -> None:
        self.events = events
        self.block = block
        self.cancelled = asyncio.Event()
        self.result = SimpleNamespace(
            content="こんにちは。", tools_used=[], usage={}, stop_reason="completed"
        )

    async def stream_events(self):
        for event in self.events:
            yield event
        if self.block:
            await self.cancelled.wait()

    async def wait(self) -> Any:
        return self.result

    async def cancel(self) -> None:
        self.cancelled.set()

    async def aclose(self) -> None:
        return None


class FakeBot:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.sessions = FakeSessions()

    async def run_streamed(self, text: str, **kwargs: Any) -> FakeRun:
        return self.run


def _config() -> GatewayConfig:
    return GatewayConfig(api_key="secret", turn_ttl=0, heartbeat_interval=0.5)


async def test_health_is_public_and_capabilities_requires_auth() -> None:
    app = create_app(FakeBot(FakeRun([])), _config())
    async with TestClient(TestServer(app)) as client:
        health = await client.get("/health")
        assert health.status == 200

        unauthorized = await client.get("/v1/voice/capabilities")
        assert unauthorized.status == 401

        capabilities = await client.get("/v1/voice/capabilities", headers=AUTH)
        assert capabilities.status == 200
        body = await capabilities.json()
        assert body["cancel"] is True
        assert "segment" in body["events"]


async def test_turn_streams_segments_and_completion() -> None:
    run = FakeRun(
        [
            SimpleNamespace(type="text.delta", delta="こんにちは。"),
            SimpleNamespace(type="text.completed", resuming=False),
        ]
    )
    app = create_app(FakeBot(run), _config())
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/voice/turns",
            headers=AUTH,
            json={
                "turn_id": "t1",
                "session_key": "s1",
                "text": "hello",
                "language": "ja",
            },
        )
        assert response.status == 200
        text = await response.text()

    assert "event: turn.started" in text
    assert "event: segment" in text
    assert "event: turn.completed" in text
    assert '"final":true' in text


async def test_cancel_endpoint_reports_turn_state() -> None:
    run = FakeRun(
        [SimpleNamespace(type="text.delta", delta="途中の文章です。")],
        block=True,
    )
    app = create_app(FakeBot(run), _config())
    manager = app[MANAGER_KEY]

    runner = await manager.register(
        {
            "turn_id": "t-cancel",
            "session_key": "s1",
            "text": "hello",
        }
    )
    task = asyncio.create_task(runner.run())
    for _ in range(100):
        if runner._run_stream is not None:
            break
        await asyncio.sleep(0.01)

    async with TestClient(TestServer(app)) as client:
        accepted = await client.post("/v1/voice/turns/t-cancel/cancel", headers=AUTH)
        assert accepted.status == 202

        await asyncio.wait_for(task, timeout=2)

        already_done = await client.post(
            "/v1/voice/turns/t-cancel/cancel", headers=AUTH
        )
        assert already_done.status == 204

        unknown = await client.post("/v1/voice/turns/missing/cancel", headers=AUTH)
        assert unknown.status == 404


async def test_duplicate_turn_id_is_rejected() -> None:
    app = create_app(FakeBot(FakeRun([])), _config())
    manager = app[MANAGER_KEY]
    payload = {"turn_id": "dup", "session_key": "s1", "text": "hello"}

    await manager.register(payload)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/voice/turns", headers=AUTH, json=payload)
        assert response.status == 409
