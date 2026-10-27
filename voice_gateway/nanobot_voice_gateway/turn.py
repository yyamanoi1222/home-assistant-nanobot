"""Turn lifecycle: SDK stream to protocol events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import GatewayConfig
from .protocol import make_event
from .segmenter import Segmenter, sanitize_for_speech
from .sessions import export_session, rollback_session

LOGGER = logging.getLogger(__name__)


class TurnState(str, Enum):
    """Lifecycle states for one voice turn."""

    ACCEPTED = "accepted"
    SNAPSHOTTING = "snapshotting"
    RUNNING = "running"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {TurnState.COMPLETED, TurnState.CANCELLED, TurnState.FAILED}
)


@dataclass(slots=True)
class TurnRequest:
    """One voice turn request."""

    turn_id: str
    session_key: str
    text: str
    language: str = "ja"
    channel: str = "assist"
    model: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    ephemeral: bool = False


class TurnRunner:
    """Owns one SDK run and translates it into protocol events.

    The runner is intentionally independent from the HTTP connection. A
    disconnected client still cancels the SDK run and rolls back the session.
    """

    def __init__(
        self,
        *,
        bot: Any,
        config: GatewayConfig,
        request: TurnRequest,
        session_lock: asyncio.Lock | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.bot = bot
        self.config = config
        self.request = request
        self.state = TurnState.ACCEPTED
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        self._session_lock = session_lock or asyncio.Lock()
        self._semaphore = semaphore or asyncio.Semaphore(config.max_concurrent_turns)

        self._seq = 0
        self._closed = False
        self._cancel_requested = False
        self._cancel_reason = "barge_in"
        self._cancel_task: asyncio.Task[None] | None = None
        self._run_stream: Any | None = None
        self._started = False
        self._result: Any | None = None
        self._run_error: str | None = None
        self._snapshot: Any = None
        self._snapshot_ok = False

        self._segmenter = Segmenter()
        self._held: str | None = None
        self._segment_index = 0
        self._emitted_any = False
        self._final_emitted = False

    @property
    def is_terminal(self) -> bool:
        """Return True once a terminal event has been emitted."""
        return self.state in TERMINAL_STATES or self._closed

    def request_cancel(self, reason: str = "barge_in") -> None:
        """Request cancellation without blocking the caller."""
        if self.is_terminal:
            return
        if not self._cancel_requested:
            self._cancel_requested = True
            self._cancel_reason = reason
            LOGGER.info("Cancelling turn %s (%s)", self.request.turn_id, reason)
        if self._run_stream is not None and (
            self._cancel_task is None or self._cancel_task.done()
        ):
            try:
                self._cancel_task = asyncio.create_task(self._cancel_stream())
            except RuntimeError:
                LOGGER.debug("No running loop; cancellation deferred")

    async def _cancel_stream(self) -> None:
        run = self._run_stream
        if run is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await run.cancel()

    async def run(self) -> None:
        """Run the turn to a terminal state."""
        if self.is_terminal:
            return
        try:
            async with self._semaphore, self._session_lock:
                await self._run_locked()
        except asyncio.CancelledError:
            self._cancel_requested = True
            await self._finish_cancelled(
                self._snapshot, self._snapshot_ok, partial=self._emitted_any
            )
        except Exception as err:
            LOGGER.exception("Unhandled turn error for %s", self.request.turn_id)
            await self._finish_failed(err, self._snapshot, self._snapshot_ok)
        finally:
            if not self._closed:
                self._closed = True
                self.queue.put_nowait(None)

    async def _run_locked(self) -> None:
        snapshot: Any = None
        snapshot_ok = False

        self.state = TurnState.SNAPSHOTTING
        if not self.request.ephemeral and self.config.snapshot_enabled:
            try:
                snapshot = await export_session(self.bot, self.request.session_key)
                snapshot_ok = True
                self._snapshot = snapshot
                self._snapshot_ok = True
            except Exception:
                LOGGER.warning(
                    "Could not snapshot session %s; rollback disabled",
                    self.request.session_key,
                    exc_info=True,
                )

        if self._cancel_requested:
            await self._finish_cancelled(snapshot, snapshot_ok, partial=False)
            return

        self.state = TurnState.RUNNING
        self._emit(
            "turn.started",
            model=self.request.model or self.config.model_preset,
            channel=self.request.channel,
        )

        try:
            self._started = True
            run = await self.bot.run_streamed(
                self.request.text,
                session_key=self.request.session_key,
                channel=self.request.channel,
                chat_id=self.request.context.get("device_id", "direct"),
                attributes=self.request.context,
                model=self.request.model,
                ephemeral=self.request.ephemeral,
            )
            self._run_stream = run

            if self._cancel_requested:
                await self._stop_stream()
            else:
                try:
                    async for event in run.stream_events():
                        if self._cancel_requested:
                            break
                        self._handle_event(event)
                except asyncio.CancelledError:
                    self._cancel_requested = True

                if not self._cancel_requested:
                    try:
                        self._result = await run.wait()
                    except asyncio.CancelledError:
                        self._cancel_requested = True
        except asyncio.CancelledError:
            self._cancel_requested = True
        except Exception as err:  # noqa: BLE001
            await self._finish_failed(err, snapshot, snapshot_ok)
            return

        if self._cancel_requested:
            await self._finish_cancelled(
                snapshot, snapshot_ok, partial=self._emitted_any
            )
            return

        if self._run_error:
            await self._finish_failed(
                RuntimeError(self._run_error), snapshot, snapshot_ok
            )
            return

        await self._finish_completed()

    async def _stop_stream(self) -> None:
        run = self._run_stream
        if run is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await run.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(run.wait(), timeout=self.config.cancel_timeout)
        with contextlib.suppress(Exception, asyncio.CancelledError):
            aclose = getattr(run, "aclose", None)
            if aclose is not None:
                await aclose()

    async def _finish_completed(self) -> None:
        if self._closed:
            return
        self.state = TurnState.FINALIZING
        self._finalize_segments()
        self.state = TurnState.COMPLETED
        result = self._result
        self._emit_terminal(
            "turn.completed",
            content=getattr(result, "content", "") or "",
            tools_used=list(getattr(result, "tools_used", []) or []),
            usage=dict(getattr(result, "usage", {}) or {}),
            stop_reason=getattr(result, "stop_reason", None),
        )

    async def _finish_cancelled(
        self,
        snapshot: Any,
        snapshot_ok: bool,
        *,
        partial: bool,
    ) -> None:
        if self._closed:
            return

        self.state = TurnState.CANCELLING
        if self._run_stream is not None:
            await self._stop_stream()

        rolled_back = True
        if self._started and snapshot_ok:
            self.state = TurnState.ROLLING_BACK
            try:
                await rollback_session(self.bot, self.request.session_key, snapshot)
            except Exception:
                rolled_back = False
                LOGGER.exception(
                    "Rollback failed for session %s", self.request.session_key
                )
        elif self._started and self.config.snapshot_enabled:
            rolled_back = False
            LOGGER.warning(
                "Turn %s cancelled without a usable snapshot",
                self.request.turn_id,
            )

        self.state = TurnState.CANCELLED
        self._emit_terminal(
            "turn.cancelled",
            reason=self._cancel_reason,
            partial=partial,
            rolled_back=rolled_back,
        )

    async def _finish_failed(
        self,
        err: Exception,
        snapshot: Any,
        snapshot_ok: bool,
    ) -> None:
        if self._closed:
            return

        if self._run_stream is not None:
            await self._stop_stream()

        if self._started and snapshot_ok:
            try:
                await rollback_session(self.bot, self.request.session_key, snapshot)
            except Exception:
                LOGGER.exception(
                    "Rollback failed for session %s", self.request.session_key
                )

        self.state = TurnState.FAILED
        self._emit_terminal(
            "error",
            code=_error_code(err),
            message=str(err),
            retryable=_is_retryable(err),
        )

    def _handle_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")

        if event_type == "text.delta":
            delta = getattr(event, "delta", "") or ""
            for segment in self._segmenter.feed(delta):
                self._push_segment(segment, final=False)
            return

        if event_type == "text.completed":
            segments = self._segmenter.flush()
            final = not bool(getattr(event, "resuming", False))
            if segments:
                for segment in segments[:-1]:
                    self._push_segment(segment, final=False)
                self._push_segment(segments[-1], final=final)
            elif final:
                self._flush_held(final=True)
            return

        if event_type == "tool.started":
            self._emit(
                "tool",
                name=getattr(event, "name", None),
                state="started",
                tool_call_id=getattr(event, "tool_call_id", None),
            )
            return

        if event_type == "tool.completed":
            self._emit(
                "tool",
                name=getattr(event, "name", None),
                state="completed",
                tool_call_id=getattr(event, "tool_call_id", None),
            )
            return

        if event_type == "tool.failed":
            self._emit(
                "tool",
                name=getattr(event, "name", None),
                state="failed",
                tool_call_id=getattr(event, "tool_call_id", None),
                error=getattr(event, "error", None),
            )
            return

        if event_type == "run.completed":
            result = getattr(event, "result", None)
            if result is not None:
                self._result = result
            return

        if event_type == "run.failed":
            self._run_error = getattr(event, "error", None) or "run failed"

    def _push_segment(self, text: str, *, final: bool) -> None:
        if self._final_emitted or not text:
            return

        if self._held is not None:
            self._emit_segment(self._held, final=False)
            self._held = None

        if final:
            self._emit_segment(text, final=True)
        else:
            self._held = text

    def _flush_held(self, *, final: bool) -> None:
        if self._held is None:
            return
        text = self._held
        self._held = None
        self._emit_segment(text, final=final)

    def _finalize_segments(self) -> None:
        if self._final_emitted:
            self._held = None
            return

        parts: list[str] = []
        if self._held is not None:
            parts.append(self._held)
            self._held = None
        parts.extend(self._segmenter.flush())

        if parts:
            for segment in parts[:-1]:
                self._emit_segment(segment, final=False)
            self._emit_segment(parts[-1], final=True)

        if not self._final_emitted:
            content = getattr(self._result, "content", "") if self._result else ""
            fallback = sanitize_for_speech(content)
            if fallback:
                self._emit_segment(fallback, final=True)

    def _emit_segment(self, text: str, *, final: bool) -> None:
        clean = sanitize_for_speech(text)
        if not clean or self._final_emitted:
            return

        self._emit(
            "segment",
            index=self._segment_index,
            text=clean,
            final=final,
            lang=self.request.language,
        )
        self._segment_index += 1
        self._emitted_any = True
        if final:
            self._final_emitted = True

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._closed:
            return
        self._seq += 1
        event = make_event(
            event_type,
            turn_id=self.request.turn_id,
            session_key=self.request.session_key,
            seq=self._seq,
            **payload,
        )
        self.queue.put_nowait(event)

    def _emit_terminal(self, event_type: str, **payload: Any) -> None:
        if self._closed:
            return
        self._emit(event_type, **payload)
        self._closed = True
        self.queue.put_nowait(None)


def _error_code(err: Exception) -> str:
    name = type(err).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connection" in name:
        return "model_error"
    return "internal"


def _is_retryable(err: Exception) -> bool:
    name = type(err).__name__.lower()
    return "timeout" in name or "connection" in name
