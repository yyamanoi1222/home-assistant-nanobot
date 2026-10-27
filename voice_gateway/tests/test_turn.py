"""Tests for the turn state machine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from nanobot_voice_gateway.config import GatewayConfig
from nanobot_voice_gateway.turn import TurnRequest, TurnRunner, TurnState


class FakeSessions:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.snapshot = {"history": ["old"]}

    async def export(self, session_key: str) -> Any:
        self.calls.append(("export", session_key))
        return self.snapshot

    def clear(self, session_key: str) -> None:
        self.calls.append(("clear", session_key))

    async def restore(self, snapshot: Any, session_key: str) -> None:
        self.calls.append(("restore", session_key, snapshot))


class FakeRun:
    def __init__(
        self,
        events: list[Any],
        result: Any | None = None,
        *,
        block: bool = False,
        fail_after: Exception | None = None,
    ) -> None:
        self.events = events
        self.result = result or SimpleNamespace(
            content="", tools_used=[], usage={}, stop_reason="completed"
        )
        self.block = block
        self.fail_after = fail_after
        self.cancelled = asyncio.Event()
        self.cancel_called = False
        self.aclose_called = False

    async def stream_events(self):
        for event in self.events:
            yield event
        if self.fail_after is not None:
            raise self.fail_after
        if self.block:
            await self.cancelled.wait()

    async def wait(self) -> Any:
        return self.result

    async def cancel(self) -> None:
        self.cancel_called = True
        self.cancelled.set()

    async def aclose(self) -> None:
        self.aclose_called = True


class FakeBot:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.sessions = FakeSessions()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run_streamed(self, text: str, **kwargs: Any) -> FakeRun:
        self.calls.append((text, kwargs))
        return self.run


def _request(**overrides: Any) -> TurnRequest:
    values = {
        "turn_id": "t1",
        "session_key": "s1",
        "text": "hello",
    }
    values.update(overrides)
    return TurnRequest(**values)


async def _drain(runner: TurnRunner) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        item = await asyncio.wait_for(runner.queue.get(), timeout=1)
        if item is None:
            break
        events.append(item)
    return events


async def test_completed_turn_emits_final_segment() -> None:
    run = FakeRun(
        [
            SimpleNamespace(type="text.delta", delta="こんにちは。"),
            SimpleNamespace(type="text.completed", resuming=False),
            SimpleNamespace(
                type="run.completed",
                result=SimpleNamespace(
                    content="こんにちは。",
                    tools_used=[],
                    usage={},
                    stop_reason="completed",
                ),
            ),
        ]
    )
    bot = FakeBot(run)
    runner = TurnRunner(bot=bot, config=GatewayConfig(), request=_request())

    await runner.run()
    events = await _drain(runner)

    assert [event["type"] for event in events] == [
        "turn.started",
        "segment",
        "turn.completed",
    ]
    assert events[1]["text"] == "こんにちは。"
    assert events[1]["final"] is True
    assert runner.state is TurnState.COMPLETED
    assert ("export", "s1") in bot.sessions.calls
    assert not any(call[0] == "clear" for call in bot.sessions.calls)


async def test_cancel_stops_run_and_rolls_back() -> None:
    run = FakeRun(
        [SimpleNamespace(type="text.delta", delta="途中の文章です。")],
        block=True,
    )
    bot = FakeBot(run)
    runner = TurnRunner(bot=bot, config=GatewayConfig(), request=_request())

    task = asyncio.create_task(runner.run())
    for _ in range(100):
        if runner._run_stream is not None:
            break
        await asyncio.sleep(0.01)

    runner.request_cancel("barge_in")
    await asyncio.wait_for(task, timeout=2)

    events = await _drain(runner)
    types = [event["type"] for event in events]

    assert "turn.cancelled" in types
    assert "turn.completed" not in types
    assert run.cancel_called
    assert ("clear", "s1") in bot.sessions.calls
    assert any(call[0] == "restore" for call in bot.sessions.calls)

    clear_index = bot.sessions.calls.index(("clear", "s1"))
    restore_index = next(
        index for index, call in enumerate(bot.sessions.calls) if call[0] == "restore"
    )
    assert clear_index < restore_index


async def test_cancel_before_start_does_not_call_sdk() -> None:
    run = FakeRun([])
    bot = FakeBot(run)
    runner = TurnRunner(bot=bot, config=GatewayConfig(), request=_request())

    runner.request_cancel("barge_in")
    await runner.run()
    events = await _drain(runner)

    assert [event["type"] for event in events] == ["turn.cancelled"]
    assert bot.calls == []
    assert not any(call[0] in {"clear", "restore"} for call in bot.sessions.calls)


async def test_failure_rolls_back_partial_turn() -> None:
    run = FakeRun(
        [SimpleNamespace(type="text.delta", delta="失敗前です。")],
        fail_after=RuntimeError("boom"),
    )
    bot = FakeBot(run)
    runner = TurnRunner(bot=bot, config=GatewayConfig(), request=_request())

    await runner.run()
    events = await _drain(runner)

    assert [event["type"] for event in events] == ["turn.started", "error"]
    assert events[1]["code"] == "internal"
    assert runner.state is TurnState.FAILED
    assert ("clear", "s1") in bot.sessions.calls
    assert any(call[0] == "restore" for call in bot.sessions.calls)


async def test_ephemeral_turn_skips_snapshot() -> None:
    run = FakeRun(
        [
            SimpleNamespace(type="text.delta", delta="こんにちは。"),
            SimpleNamespace(type="text.completed", resuming=False),
        ]
    )
    bot = FakeBot(run)
    runner = TurnRunner(
        bot=bot,
        config=GatewayConfig(),
        request=_request(ephemeral=True),
    )

    await runner.run()
    events = await _drain(runner)

    assert events[-1]["type"] == "turn.completed"
    assert not any(call[0] == "export" for call in bot.sessions.calls)
