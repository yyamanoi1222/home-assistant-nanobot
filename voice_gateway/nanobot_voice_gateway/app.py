"""aiohttp application for the voice gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from aiohttp import web

from .config import GatewayConfig
from .protocol import EVENT_TYPES, PING_FRAME, PROTOCOL_VERSION, sse
from .turn import TurnRequest, TurnRunner

LOGGER = logging.getLogger(__name__)


class InvalidTurn(ValueError):
    """Raised when a turn payload is malformed."""


class DuplicateTurn(ValueError):
    """Raised when a turn_id is already known."""


class TurnManager:
    """Track active turns and their session locks."""

    def __init__(self, bot: Any, config: GatewayConfig) -> None:
        self.bot = bot
        self.config = config
        self._turns: dict[str, TurnRunner] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_turns)
        self._lock = asyncio.Lock()

    async def register(self, payload: Any) -> TurnRunner:
        request = self._parse(payload)
        async with self._lock:
            if request.turn_id in self._turns:
                raise DuplicateTurn(request.turn_id)
            lock = self._locks.setdefault(request.session_key, asyncio.Lock())
            runner = TurnRunner(
                bot=self.bot,
                config=self.config,
                request=request,
                session_lock=lock,
                semaphore=self._semaphore,
            )
            self._turns[request.turn_id] = runner
        return runner

    async def cancel(self, turn_id: str) -> int:
        async with self._lock:
            runner = self._turns.get(turn_id)
            if runner is None:
                return 404
            if runner.is_terminal:
                return 204
            runner.request_cancel("barge_in")
            return 202

    async def abandon(self, turn_id: str) -> None:
        async with self._lock:
            self._turns.pop(turn_id, None)

    def track(self, turn_id: str, task: asyncio.Task[None]) -> None:
        task.add_done_callback(lambda done: self._on_done(turn_id, done))

    def _on_done(self, turn_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            LOGGER.warning("Turn task %s was cancelled", turn_id)
        else:
            error = task.exception()
            if error is not None:
                LOGGER.error("Turn task %s failed", turn_id, exc_info=error)
        if self.config.turn_ttl <= 0:
            self._turns.pop(turn_id, None)
            return
        try:
            asyncio.create_task(self._remove_later(turn_id))
        except RuntimeError:
            LOGGER.debug("Event loop closed before turn cleanup")

    async def _remove_later(self, turn_id: str) -> None:
        await asyncio.sleep(self.config.turn_ttl)
        await self.abandon(turn_id)

    @staticmethod
    def _parse(payload: Any) -> TurnRequest:
        if not isinstance(payload, dict):
            raise InvalidTurn("body must be a JSON object")

        turn_id = payload.get("turn_id")
        session_key = payload.get("session_key")
        text = payload.get("text")

        if not isinstance(turn_id, str) or not turn_id:
            raise InvalidTurn("turn_id must be a non-empty string")
        if not isinstance(session_key, str) or not session_key:
            raise InvalidTurn("session_key must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise InvalidTurn("text must be a non-empty string")

        context = payload.get("context") or {}
        if not isinstance(context, dict):
            raise InvalidTurn("context must be an object")

        language = payload.get("language", "ja")
        channel = payload.get("channel", "assist")
        model = payload.get("model")

        return TurnRequest(
            turn_id=turn_id,
            session_key=session_key,
            text=text.strip(),
            language=str(language),
            channel=str(channel),
            model=str(model) if model else None,
            context=context,
            ephemeral=bool(payload.get("ephemeral", False)),
        )


MANAGER_KEY: web.AppKey[TurnManager] = web.AppKey("manager", TurnManager)
CONFIG_KEY: web.AppKey[GatewayConfig] = web.AppKey("config", GatewayConfig)


def create_app(bot: Any, config: GatewayConfig) -> web.Application:
    """Build the aiohttp application."""

    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        if (
            config.api_key
            and request.path != "/health"
            and request.headers.get("Authorization") != f"Bearer {config.api_key}"
        ):
            raise web.HTTPUnauthorized(text="invalid api key")
        return await handler(request)

    app = web.Application(middlewares=[auth_middleware])
    manager = TurnManager(bot, config)
    app[MANAGER_KEY] = manager
    app[CONFIG_KEY] = config

    app.router.add_get("/health", _handle_health)
    app.router.add_get("/v1/voice/capabilities", _handle_capabilities)
    app.router.add_post("/v1/voice/turns", _handle_turn)
    app.router.add_post("/v1/voice/turns/{turn_id}/cancel", _handle_cancel)
    return app


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _handle_capabilities(request: web.Request) -> web.Response:
    config: GatewayConfig = request.app[CONFIG_KEY]
    return web.json_response(
        {
            "v": PROTOCOL_VERSION,
            "events": list(EVENT_TYPES),
            "cancel": True,
            "rollback": True,
            "max_concurrent_turns": config.max_concurrent_turns,
        }
    )


async def _handle_turn(request: web.Request) -> web.StreamResponse:
    manager: TurnManager = request.app[MANAGER_KEY]
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="invalid JSON body") from None

    try:
        runner = await manager.register(payload)
    except InvalidTurn as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    except DuplicateTurn as err:
        raise web.HTTPConflict(text=f"turn_id already in use: {err}") from err

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

    try:
        await response.prepare(request)
    except Exception:
        await manager.abandon(runner.request.turn_id)
        raise

    task = asyncio.create_task(
        runner.run(), name=f"nanobot-turn:{runner.request.turn_id}"
    )
    manager.track(runner.request.turn_id, task)

    heartbeat = max(1.0, request.app[CONFIG_KEY].heartbeat_interval)

    try:
        while True:
            try:
                item = await asyncio.wait_for(runner.queue.get(), timeout=heartbeat)
            except TimeoutError:
                await response.write(PING_FRAME)
                continue

            if item is None:
                break

            await response.write(sse(item["type"], item))
    except asyncio.CancelledError:
        runner.request_cancel("disconnect")
        raise
    except (ConnectionError, RuntimeError):
        runner.request_cancel("disconnect")
    finally:
        with contextlib.suppress(Exception):
            await response.write_eof()

    return response


async def _handle_cancel(request: web.Request) -> web.Response:
    manager: TurnManager = request.app[MANAGER_KEY]
    status = await manager.cancel(request.match_info["turn_id"])
    if status == 404:
        raise web.HTTPNotFound()
    if status == 204:
        return web.Response(status=204)
    return web.Response(status=202)
