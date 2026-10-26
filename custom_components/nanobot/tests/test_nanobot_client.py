"""Tests for the nanobot API client."""

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from custom_components.nanobot.nanobot_client import (
    NanobotAPIError,
    NanobotClient,
    NanobotConnectionError,
)


class _SseContent:
    """Async iterator over SSE lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration as err:
            raise StopAsyncIteration from err


async def test_validate_connection_success() -> None:
    """Test successful health check."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.get.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    await client.validate_connection()

    session.get.assert_called_once()


async def test_validate_connection_failure() -> None:
    """Test health check failure raises connection error."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = False
    response.status = 500
    response.text = AsyncMock(return_value="Internal Server Error")
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.get.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotAPIError):
        await client.validate_connection()


async def test_send_message_success() -> None:
    """Test sending a message and parsing the response."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.json = AsyncMock(
        return_value={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello, world!"}}],
        }
    )
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    result = await client.send_message("Hi", session_id="test-session")

    assert result == "Hello, world!"
    session.post.assert_called_once()
    call_kwargs = session.post.call_args.kwargs
    assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert call_kwargs["json"]["session_id"] == "test-session"
    assert call_kwargs["json"]["stream"] is False


async def test_send_message_no_session_id() -> None:
    """Test sending a message without a session ID."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.json = AsyncMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        }
    )
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    result = await client.send_message("Hello")

    assert result == "OK"
    call_kwargs = session.post.call_args.kwargs
    assert "session_id" not in call_kwargs["json"]
    assert "model" not in call_kwargs["json"]


async def test_send_message_with_model() -> None:
    """Test sending a request with an explicit model."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.json = AsyncMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        }
    )
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    result = await client.send_message("Hello", model="MiniMax-M2.7")

    assert result == "OK"
    call_kwargs = session.post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "MiniMax-M2.7"


async def test_send_message_api_key_header() -> None:
    """Test that API key is sent as Bearer token."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.json = AsyncMock(return_value={"choices": [{"message": {"content": ""}}]})
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(
        session=session, api_url="http://127.0.0.1:8900", api_key="secret"
    )
    await client.send_message("Hi")

    call_kwargs = session.post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer secret"


async def test_send_message_connection_error() -> None:
    """Test connection error handling."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.side_effect = aiohttp.ClientError("Connection refused")

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotConnectionError):
        await client.send_message("Hi")


async def test_send_message_invalid_response() -> None:
    """Test handling of invalid API response."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.json = AsyncMock(return_value={"invalid": "response"})
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotAPIError):
        await client.send_message("Hi")


def _streaming_response(lines: list[bytes]) -> MagicMock:
    """Build a mocked streaming aiohttp response."""
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = True
    response.status = 200
    response.content = _SseContent(lines)
    return response


async def _collect_stream(client: NanobotClient, text: str, **kwargs) -> list[str]:
    return [chunk async for chunk in client.send_message_stream(text, **kwargs)]


async def test_send_message_stream_success() -> None:
    """Test streaming assistant content from SSE events."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = _streaming_response(
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":", world!"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
    )
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    chunks = await _collect_stream(client, "Hi", session_id="test-session")

    assert chunks == ["Hello", ", world!"]
    call_kwargs = session.post.call_args.kwargs
    assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert call_kwargs["json"]["session_id"] == "test-session"
    assert call_kwargs["json"]["stream"] is True
    assert call_kwargs["headers"]["Accept"] == "text/event-stream"


async def test_send_message_stream_with_model() -> None:
    """Test the model is forwarded on streaming requests."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = _streaming_response([b"data: [DONE]\n"])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    chunks = await _collect_stream(client, "Hi", model="MiniMax-M2.7")

    assert chunks == []
    assert session.post.call_args.kwargs["json"]["model"] == "MiniMax-M2.7"


async def test_send_message_stream_api_error() -> None:
    """Test a non-2xx streaming response raises an API error."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock(spec=aiohttp.ClientResponse)
    response.ok = False
    response.status = 500
    response.text = AsyncMock(return_value='{"error": {"message": "boom"}}')
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotAPIError, match="boom"):
        await _collect_stream(client, "Hi")


async def test_send_message_stream_malformed_event() -> None:
    """Test malformed SSE JSON raises an API error."""
    session = MagicMock(spec=aiohttp.ClientSession)
    response = _streaming_response([b"data: {not-json}\n"])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = cm

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotAPIError):
        await _collect_stream(client, "Hi")


async def test_send_message_stream_connection_error() -> None:
    """Test connection errors while streaming."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.side_effect = aiohttp.ClientError("Connection refused")

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(NanobotConnectionError):
        await _collect_stream(client, "Hi")


async def test_send_message_stream_empty_text() -> None:
    """Test that empty text is rejected before making a request."""
    session = MagicMock(spec=aiohttp.ClientSession)

    client = NanobotClient(session=session, api_url="http://127.0.0.1:8900")
    with pytest.raises(ValueError):
        await _collect_stream(client, "   ")
