"""Async client for the nanobot OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .const import LOGGER


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


class NanobotClient:
    """Thin client for nanobot's OpenAI-compatible chat completions API."""

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
            api_url: Base URL of the nanobot API, e.g. http://127.0.0.1:8900.
            api_key: Optional Bearer token for authenticated endpoints.
            timeout: Request timeout in seconds.
        """
        self._session = session
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)

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
        self, text: str, session_id: str | None = None
    ) -> str:
        """Send a user message to nanobot and return the assistant reply.

        nanobot's OpenAI-compatible endpoint requires exactly one user message
        per request and manages conversation history internally via ``session_id``.
        We therefore always send a single message and map Home Assistant's
        ``conversation_id`` directly to nanobot's ``session_id``.

        Args:
            text: The user utterance to send.
            session_id: Optional session identifier for conversation continuity.

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
                    error_text = body.get("error", {}).get("message", await response.text()) if isinstance(body, dict) else await response.text()
                    raise NanobotAPIError(
                        f"nanobot API error ({response.status}): {error_text}",
                        status=response.status,
                    )
        except aiohttp.ClientError as err:
            raise NanobotConnectionError(f"Error communicating with nanobot: {err}") from err
        except TimeoutError as err:
            raise NanobotConnectionError("Request to nanobot timed out") from err

        return self._extract_response_text(body)

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
