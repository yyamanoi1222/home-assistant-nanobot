"""Base conversation entity for nanobot."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncGenerator
from dataclasses import fields
from typing import Any, cast
from uuid import uuid4

from homeassistant.components.conversation import (
    AssistantContent,
    AssistantContentDeltaDict,
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.intent import IntentResponse

from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_FILLER_TEXT,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_VOICE_PROMPT,
    DEFAULT_FILLER_TEXT,
    DEFAULT_VOICE_PROMPT,
    DOMAIN,
    LOGGER,
)
from .nanobot_client import (
    NanobotAPIError,
    NanobotClient,
    NanobotClientError,
    NanobotConnectionError,
)

_CONVERSATION_RESULT_FIELDS = {f.name for f in fields(ConversationResult)}
_ECHO_MIN_LENGTH = 8
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_echo(text: str) -> str:
    """Normalize text so spoken echo can be compared with a reply."""
    return _NON_WORD_RE.sub("", text.lower())


def _looks_like_echo(text: str, last_response: str | None) -> bool:
    """Return True when user input is part of the previous reply."""
    if not last_response:
        return False

    normalized_text = _normalize_for_echo(text)
    if len(normalized_text) < _ECHO_MIN_LENGTH:
        return False

    return normalized_text in _normalize_for_echo(last_response)


def _make_result(
    response: IntentResponse,
    conversation_id: str | None,
    continue_conversation: bool = False,
) -> ConversationResult:
    if "continue_conversation" in _CONVERSATION_RESULT_FIELDS:
        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )
    return ConversationResult(response=response, conversation_id=conversation_id)


def _make_speech_result(
    language: str,
    speech: str,
    conversation_id: str | None,
    continue_conversation: bool = False,
) -> ConversationResult:
    """Build a result that speaks a complete message."""
    intent_response = IntentResponse(language=language)
    intent_response.async_set_speech(speech)
    return _make_result(intent_response, conversation_id, continue_conversation)


def _turn_assistant_text(chat_log: ChatLog, start_index: int) -> str | None:
    """Return all assistant text added to the chat log during this turn."""
    texts = [
        content.content
        for content in chat_log.content[start_index:]
        if isinstance(content, AssistantContent) and content.content
    ]
    if not texts:
        return None
    return " ".join(texts)


class NanobotBaseConversationEntity(ConversationEntity):
    """Base nanobot conversation entity.

    Subclasses apply platform-specific registration behavior (e.g. agent
    registration in the conversation platform).
    """

    _attr_has_entity_name = True
    _attr_name = "nanobot"
    _attr_supported_features = 0
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry_id: str, config: dict) -> None:
        """Initialize the nanobot conversation entity.

        Args:
            hass: Home Assistant instance.
            entry_id: Config entry identifier.
            config: Entry data containing api_url, api_key, model, and timeout.
        """
        self.hass = hass
        self._entry_id = entry_id
        self._api_url = config[CONF_API_URL]
        self._api_key = config.get(CONF_API_KEY)
        self._model = config.get(CONF_MODEL)
        self._timeout = config.get(CONF_TIMEOUT, 120)
        self._voice_prompt = config.get(CONF_VOICE_PROMPT, DEFAULT_VOICE_PROMPT)
        self._filler_text = config.get(CONF_FILLER_TEXT, DEFAULT_FILLER_TEXT)
        self._prompted_sessions: set[str] = set()
        self._last_response_text: str | None = None

        self._attr_unique_id = entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": "nanobot",
            "manufacturer": "yyamanoi1222",
            "model": "nanobot",
            "entry_type": "service",
        }

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process a user message via nanobot.

        Home Assistant's Assist pipeline (especially voice) supplies a new
        ``conversation_id`` on every turn. To preserve context across turns we
        stabilize the nanobot ``session_id`` by falling back to the
        ``device_id`` when no meaningful conversation ID is present, or by
        using the caller-supplied ``conversation_id`` as-is.

        nanobot's API requires exactly one user message per request; we
        always send only the latest user utterance and rely on session_id
        for continuity. The configured voice prompt is prepended to the first
        request of each session. Reply tokens are streamed into the chat log
        so the Assist pipeline can start TTS before the full answer is
        generated. Successful replies keep the conversation open so Assist
        listens for a follow-up. If speech-to-text echoes part of the previous
        reply back, the input is dropped instead of being sent to nanobot
        again.
        """
        text = (user_input.text or "").strip()
        if not text:
            return _make_speech_result(
                user_input.language, "I didn't catch that.", None, False
            )

        session_id = self._resolve_session_id(user_input)

        if _looks_like_echo(text, self._last_response_text):
            LOGGER.debug("Ignoring possible echo of the previous reply: %s", text)
            return _make_speech_result(user_input.language, "", session_id, False)

        session = async_get_clientsession(self.hass)
        client = NanobotClient(
            session=session,
            api_url=self._api_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        start_index = len(chat_log.content)
        try:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._async_stream_reply(client, text, session_id, user_input),
            ):
                pass
        except NanobotClientError as err:
            LOGGER.error("nanobot conversation error: %s", err)
            return _make_speech_result(
                user_input.language,
                "Sorry, I could not reach nanobot.",
                session_id,
                False,
            )

        response_text = _turn_assistant_text(chat_log, start_index)
        if not response_text:
            LOGGER.error("nanobot returned an empty reply")
            return _make_speech_result(
                user_input.language,
                "Sorry, I could not reach nanobot.",
                session_id,
                False,
            )

        self._last_response_text = response_text
        return _make_speech_result(user_input.language, response_text, session_id, True)

    def _resolve_session_id(self, user_input: ConversationInput) -> str:
        """Return the nanobot session ID for a conversation turn."""
        return (
            user_input.conversation_id
            or user_input.device_id
            or f"{DOMAIN}-{self._entry_id}"
        )

    @staticmethod
    async def _async_pump_stream(
        stream: AsyncGenerator[str, None],
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> None:
        """Move nanobot tokens into a queue so consumers can apply timeouts."""
        try:
            async for token in stream:
                await queue.put(("token", token))
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            await queue.put(("error", err))
        else:
            await queue.put(("done", None))

    async def _async_stream_reply(
        self,
        client: NanobotClient,
        text: str,
        session_id: str,
        user_input: ConversationInput,
    ) -> AsyncGenerator[AssistantContentDeltaDict, None]:
        """Translate nanobot voice events into chat log deltas.

        The response start is bounded by the configured timeout. Once nanobot
        starts streaming, no additional timeout is applied: long tool calls and
        pauses must not cut the reply off.
        """
        voice_prompt = (self._voice_prompt or "").strip()
        inject_prompt = bool(voice_prompt) and session_id not in self._prompted_sessions
        message = f"{voice_prompt}\n\n{text}" if inject_prompt else text

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stream = self._async_voice_token_stream(client, message, session_id, user_input)
        producer = asyncio.create_task(self._async_pump_stream(stream, queue))

        first = True
        received_token = False

        try:
            while True:
                if first:
                    try:
                        kind, value = await asyncio.wait_for(
                            queue.get(), timeout=self._timeout
                        )
                    except TimeoutError as err:
                        raise NanobotConnectionError(
                            "Timeout waiting for nanobot to start responding"
                        ) from err
                else:
                    kind, value = await queue.get()

                if kind == "error":
                    raise value
                if kind == "done":
                    break

                token = cast(str, value)
                received_token = True

                if inject_prompt:
                    self._prompted_sessions.add(session_id)
                    inject_prompt = False

                if first:
                    yield {"role": "assistant", "content": token}
                    first = False
                else:
                    yield {"content": token}
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            with contextlib.suppress(Exception):
                await stream.aclose()

        if not received_token:
            raise NanobotAPIError("nanobot returned an empty reply")

    async def _async_voice_token_stream(
        self,
        client: NanobotClient,
        text: str,
        session_id: str,
        user_input: ConversationInput,
    ) -> AsyncGenerator[str, None]:
        """Yield speakable text from the voice gateway event stream."""
        turn_id = uuid4().hex
        completed = False
        spoken_segment = False
        filler_spoken = False

        try:
            async for event in client.stream_voice_turn(
                text=text,
                session_id=session_id,
                model=self._model,
                language=user_input.language or "ja",
                context={
                    "device_id": user_input.device_id,
                    "conversation_id": user_input.conversation_id,
                },
                turn_id=turn_id,
            ):
                event_type = event.get("type")

                if event_type == "segment":
                    segment = event.get("text") or ""
                    if segment:
                        spoken_segment = True
                        yield segment
                    continue

                if event_type == "tool":
                    if (
                        event.get("state") == "started"
                        and not spoken_segment
                        and not filler_spoken
                        and self._filler_text
                    ):
                        filler_spoken = True
                        yield self._filler_text
                    continue

                if event_type == "turn.completed":
                    completed = True
                    return

                if event_type == "turn.cancelled":
                    completed = True
                    return

                if event_type == "error":
                    message = event.get("message") or "nanobot voice turn failed"
                    raise NanobotAPIError(str(message))
        finally:
            if not completed:
                with contextlib.suppress(Exception):
                    await client.cancel_voice_turn(turn_id)
