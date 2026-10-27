"""SSE protocol helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

PROTOCOL_VERSION = 1

EVENT_TYPES = (
    "turn.started",
    "status",
    "segment",
    "tool",
    "turn.completed",
    "turn.cancelled",
    "error",
)

TERMINAL_EVENT_TYPES = frozenset({"turn.completed", "turn.cancelled", "error"})


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_event(
    event_type: str,
    *,
    turn_id: str,
    session_key: str,
    seq: int,
    **payload: Any,
) -> dict[str, Any]:
    """Build one protocol event."""
    return {
        "v": PROTOCOL_VERSION,
        "type": event_type,
        "turn_id": turn_id,
        "session_key": session_key,
        "seq": seq,
        "ts": utcnow_iso(),
        **payload,
    }


def sse(event: str, data: dict[str, Any]) -> bytes:
    """Encode one SSE frame."""
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode()


PING_FRAME = b": ping\n\n"
