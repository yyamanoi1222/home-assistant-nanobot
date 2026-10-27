"""Async client for the nanobot voice gateway and chat API."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .const import LOGGER

_SSE_DONE = object()


class NanobotClientError(HomeAssistantError):
    """Base error for nanobot client failures."""


class NanobotConnectionError(NanobotClientError):
    """Error raised when nanobot is unreachable."""


class NanobotAPIError(NanobotClientError):
    """Error raised when nanobot returns a non-2xx response."""

    def __init__(self, message: str, status: int | None = None) -> None:
        """Initialize the error with an optional HTTP status."""
        super().__init__(message)
        self.status = status


class NanobotVoiceUnavailable(NanobotAPIError):
    """Error raised when the voice gateway endpoints are not available."""


class NanobotClient:
    """Thin client for the nanobot voice gateway and chat completions API."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize the client.

        Args:
            session: aiohttp session to use for requests.
            api_url: Base URL of the nanobot voice gateway, e.g. http://127.0.0.1:8900.
            api_key: Optional Bearer token for authenticated endpoints.
            timeout: Request timeout in seconds.
        """
        self._session = session
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        # Server-sent event responses can stay silent for a long time while
        # nanobot runs a tool. A total or read timeout would close the
        # connection mid-turn, so only cap the connection attempt here.
        self._stream_timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=timeout,
            sock_read=None,
        )

    @property
    def api_url(self) -> str:
        """Return the configured API URL."""
        return self._api_url

    def _headers(self) -> Mapping[str, str]:
        """Build request headers."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def validate_connection(self) -> None:
        """Validate connectivity by calling the /health endpoint.

        Raises:
            NanobotConnectionError: If nanobot cannot be reached.
            NanobotAPIError: If nanobot returns an error response.
        """
        url = f"{self._api_url}/health"
        try:
            async with self._session.get(
                url, headers=self._headers(), timeout=self._timeout
            ) as response:
                if not response.ok:
                    text = await response.text()
                    raise NanobotAPIError(
                        f"Health check failed ({response.status}): {text}",
                        status=response.status,
                    )
        except aiohttp.ClientError as err:
            raise NanobotConnectionError(
                f"Unable to reach nanobot at {self._api_url}: {err}"
            ) from err
        except TimeoutError as err:
            raise NanobotConnectionError(
                f"Timeout while connecting to nanobot at {self._api_url}"
            ) from err

    async def send_message(
        self,
        text: str,
        session_id: str | None = None,
        model: str | None = None,
    ) -> str:
        """Send a user message to nanobot and return the assistant reply.

        nanobot's OpenAI-compatible endpoint requires exactly one user message
        per request and manages conversation history internally via ``session_id``.
        We therefore always send a single message and map Home Assistant's
        ``conversation_id`` directly to nanobot's ``session_id``.

        Args:
            text: The user utterance to send.
            session_id: Optional session identifier for conversation continuity.
            model: Optional model name. When provided it must match a model
                returned by ``/v1/models``. Leave unset to use nanobot's
                configured default model.

        Returns:
            The assistant response text.

        Raises:
            NanobotConnectionError: If the request fails due to network issues.
            NanobotAPIError: If the API returns an error or an unexpected body.
            ValueError: If ``text`` is empty or contains only whitespace.
        """
        stripped_text = (text or "").strip()
        if not stripped_text:
            raise ValueError("text must not be empty")

        url = f"{self._api_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": stripped_text}],
            "stream": False,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if model is not None:
            payload["model"] = model

        LOGGER.debug(
            "Sending request to nanobot (%s): %s",
            url,
            {**payload, "messages": "<redacted>"} if payload else payload,
        )

        try:
            async with self._session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            ) as response:
                body = await response.json(content_type=None)
                if not response.ok:
                    error_text = (
                        body.get("error", {}).get("message", await response.text())
                        if isinstance(body, dict)
                        else await response.text()
                    )
                    raise NanobotAPIError(
                        f"nanobot API error ({response.status}): {error_text}",
                        status=response.status,
                    )
        except aiohttp.ClientError as err:
            raise NanobotConnectionError(
                f"Error communicating with nanobot: {err}"
            ) from err
        except TimeoutError as err:
            raise NanobotConnectionError("Request to nanobot timed out") from err

        return self._extract_response_text(body)

    async def send_message_stream(
        self,
        text: str,
        session_id: str | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Send a user message and yield assistant reply tokens as they arrive.

        nanobot's streaming endpoint emits OpenAI-compatible server-sent events
        with ``choices[0].delta.content`` fragments. The stream ends with a
        ``data: [DONE]`` sentinel.

        Args:
            text: The user utterance to send.
            session_id: Optional session identifier for conversation continuity.
            model: Optional model name.

        Yields:
            Non-empty assistant content fragments in arrival order.

        Raises:
            NanobotConnectionError: If the request fails due to network issues.
            NanobotAPIError: If the API returns an error or an invalid event.
            ValueError: If ``text`` is empty or contains only whitespace.
        """
        stripped_text = (text or "").strip()
        if not stripped_text:
            raise ValueError("text must not be empty")

        url = f"{self._api_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": stripped_text}],
            "stream": True,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if model is not None:
            payload["model"] = model

        LOGGER.debug(
            "Streaming request to nanobot (%s): %s",
            url,
            {**payload, "messages": "<redacted>"},
        )

        headers = {**self._headers(), "Accept": "text/event-stream"}
        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._stream_timeout,
            ) as response:
                if not response.ok:
                    error_text = await self._extract_error_text(response)
                    raise NanobotAPIError(
                        f"nanobot API error ({response.status}): {error_text}",
                        status=response.status,
                    )

                async for raw_line in response.content:
                    content = self._parse_sse_line(raw_line)
                    if content is None:
                        continue
                    if content is _SSE_DONE:
                        break
                    yield content
        except aiohttp.ClientError as err:
            raise NanobotConnectionError(
                f"Error communicating with nanobot: {err}"
            ) from err
        except TimeoutError as err:
            raise NanobotConnectionError("Request to nanobot timed out") from err

    async def stream_voice_turn(
        self,
        *,
        text: str,
        session_id: str,
        model: str | None = None,
        language: str = "ja",
        context: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run one voice turn against the nanobot voice gateway.

        Yields the structured events documented by the gateway:
        ``turn.started``, ``segment``, ``tool``, ``turn.completed``,
        ``turn.cancelled`` and ``error``.
        """
        stripped_text = (text or "").strip()
        if not stripped_text:
            raise ValueError("text must not be empty")

        turn_id = turn_id or uuid4().hex
        url = f"{self._api_url}/v1/voice/turns"
        payload: dict[str, Any] = {
            "turn_id": turn_id,
            "session_key": session_id,
            "text": stripped_text,
            "language": language,
            "context": context or {},
        }
        if model is not None:
            payload["model"] = model

        LOGGER.debug(
            "Voice turn request to nanobot (%s): %s",
            url,
            {**payload, "text": "<redacted>"},
        )

        headers = {**self._headers(), "Accept": "text/event-stream"}
        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._stream_timeout,
            ) as response:
                if not response.ok:
                    error_text = await self._extract_error_text(response)
                    if response.status in (404, 405):
                        raise NanobotVoiceUnavailable(
                            "nanobot voice gateway is not available",
                            status=response.status,
                        )
                    raise NanobotAPIError(
                        f"nanobot voice API error ({response.status}): {error_text}",
                        status=response.status,
                    )

                event_name: str | None = None
                data_lines: list[str] = []

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            yield self._decode_voice_event(event_name, data_lines)
                        event_name = None
                        data_lines = []
                        continue

                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())

                if data_lines:
                    yield self._decode_voice_event(event_name, data_lines)
        except aiohttp.ClientError as err:
            raise NanobotConnectionError(
                f"Error communicating with nanobot: {err}"
            ) from err
        except TimeoutError as err:
            raise NanobotConnectionError("Request to nanobot timed out") from err

    async def cancel_voice_turn(self, turn_id: str) -> None:
        """Ask the gateway to cancel a running turn (best effort)."""
        url = f"{self._api_url}/v1/voice/turns/{quote(turn_id, safe='')}/cancel"
        try:
            async with self._session.post(
                url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status not in (202, 204, 404):
                    LOGGER.debug(
                        "Cancel for turn %s returned %s",
                        turn_id,
                        response.status,
                    )
        except (aiohttp.ClientError, TimeoutError):
            LOGGER.debug("Could not cancel turn %s", turn_id, exc_info=True)

    @staticmethod
    def _decode_voice_event(
        event_name: str | None, data_lines: list[str]
    ) -> dict[str, Any]:
        """Decode one SSE event into a voice protocol event."""
        data = "\n".join(data_lines)
        try:
            event = json.loads(data)
        except ValueError as err:
            raise NanobotAPIError(
                "Invalid voice event from nanobot: malformed JSON"
            ) from err

        if not isinstance(event, dict):
            raise NanobotAPIError(
                "Invalid voice event from nanobot: expected JSON object"
            )

        if event_name and not event.get("type"):
            event["type"] = event_name
        if not event.get("type"):
            raise NanobotAPIError(
                "Invalid voice event from nanobot: missing event type"
            )
        return event

    @staticmethod
    async def _extract_error_text(response: aiohttp.ClientResponse) -> str:
        """Return a readable error message for a non-2xx streaming response."""
        text = await response.text()
        try:
            body = json.loads(text)
        except ValueError:
            return text
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
        return text

    @staticmethod
    def _parse_sse_line(raw_line: bytes) -> str | None | object:
        """Extract an assistant content fragment from one SSE line."""
        line = raw_line.strip()
        if not line or not line.startswith(b"data:"):
            return None

        data = line[len(b"data:") :].strip()
        if data == b"[DONE]":
            return _SSE_DONE

        try:
            chunk = json.loads(data)
        except ValueError as err:
            raise NanobotAPIError(
                "Invalid streaming response from nanobot: malformed JSON"
            ) from err

        if not isinstance(chunk, dict):
            raise NanobotAPIError(
                "Invalid streaming response from nanobot: expected JSON object"
            )

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first_choice = choices[0]
        delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
        if not isinstance(delta, dict):
            return None

        content = delta.get("content")
        if not content:
            return None

        return str(content)

    @staticmethod
    def _extract_response_text(body: Any) -> str:
        """Extract assistant content from an OpenAI-style chat completion."""
        if not isinstance(body, dict):
            raise NanobotAPIError("Invalid response from nanobot: expected JSON object")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise NanobotAPIError("Invalid response from nanobot: missing choices")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise NanobotAPIError("Invalid response from nanobot: missing message")

        content = message.get("content")
        if content is None:
            raise NanobotAPIError("Invalid response from nanobot: missing content")

        return str(content)
