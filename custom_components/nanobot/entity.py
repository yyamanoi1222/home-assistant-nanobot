"""Base conversation entity for nanobot."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from dataclasses import fields

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

from .const import CONF_API_KEY, CONF_API_URL, CONF_MODEL, CONF_TIMEOUT, DOMAIN, LOGGER
from .nanobot_client import NanobotClient, NanobotClientError

_CONVERSATION_RESULT_FIELDS = {f.name for f in fields(ConversationResult)}
_ECHO_MIN_LENGTH = 8
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_echo(text: str) -> str:
    """Normalize text so spoken echo can be compared with a reply."""
    return _NON_WORD_RE.sub("", text.lower())


def _looks_like_echo(text: str, last_response: str | None) -> bool:
    """Return True when user input exactly echoes the previous reply."""
    if not last_response:
        return False

    normalized_text = _normalize_for_echo(text)
    if len(normalized_text) < _ECHO_MIN_LENGTH:
        return False

    return normalized_text == _normalize_for_echo(last_response)


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


def _last_assistant_content(chat_log: ChatLog) -> str | None:
    """Return the text of the last assistant message in the chat log."""
    if not chat_log.content:
        return None

    content = chat_log.content[-1]
    if isinstance(content, AssistantContent) and content.content:
        return content.content
    return None


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
        for continuity. Reply tokens are streamed into the chat log so the
        Assist pipeline can start TTS before the full answer is generated.
        Successful replies keep the conversation open so Assist listens for a
        follow-up. If speech-to-text echoes the previous reply back, the input
        is dropped instead of being sent to nanobot again.
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

        try:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._async_stream_reply(client, text, session_id),
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

        response_text = _last_assistant_content(chat_log)
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

    async def _async_stream_reply(
        self,
        client: NanobotClient,
        text: str,
        session_id: str,
    ) -> AsyncGenerator[AssistantContentDeltaDict, None]:
        """Translate nanobot SSE fragments into chat log deltas."""
        first = True
        async for token in client.send_message_stream(
            text=text,
            session_id=session_id,
            model=self._model,
        ):
            if first:
                yield {"role": "assistant", "content": token}
                first = False
            else:
                yield {"content": token}
