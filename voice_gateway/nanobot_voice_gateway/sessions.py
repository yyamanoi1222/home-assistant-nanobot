"""Session snapshot and rollback helpers."""

from __future__ import annotations

import inspect
from typing import Any


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def export_session(bot: Any, session_key: str) -> Any:
    """Export a trusted session snapshot, raising if the SDK cannot export."""
    sessions = getattr(bot, "sessions", None)
    if sessions is None:
        return None
    return await _maybe_await(sessions.export(session_key))


async def rollback_session(bot: Any, session_key: str, snapshot: Any) -> None:
    """Restore a session snapshot, clearing any partial turn first."""
    sessions = getattr(bot, "sessions", None)
    if sessions is None:
        return

    await _maybe_await(sessions.clear(session_key))
    if snapshot is not None:
        await _maybe_await(sessions.restore(snapshot, session_key))
