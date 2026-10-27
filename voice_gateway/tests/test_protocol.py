"""Tests for the SSE protocol helpers."""

from nanobot_voice_gateway.protocol import (
    PROTOCOL_VERSION,
    make_event,
    sse,
)


def test_make_event_envelope() -> None:
    event = make_event("segment", turn_id="t1", session_key="s1", seq=2, text="hi")
    assert event["v"] == PROTOCOL_VERSION
    assert event["type"] == "segment"
    assert event["turn_id"] == "t1"
    assert event["session_key"] == "s1"
    assert event["seq"] == 2
    assert event["text"] == "hi"
    assert event["ts"].endswith("Z")


def test_sse_frame() -> None:
    assert sse("segment", {"a": 1}) == b'event: segment\ndata: {"a":1}\n\n'
