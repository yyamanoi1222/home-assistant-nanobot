"""Base conversation entity for nanobot."""

from __future__ import annotations

from dataclasses import fields

from homeassistant.components.conversation import (
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


def _make_result(
    response: IntentResponse,
    conversation_id: str | None,
    continue_conversation: bool,
) -> ConversationResult:
    if "continue_conversation" in _CONVERSATION_RESULT_FIELDS:
        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )
    return ConversationResult(response=response, conversation_id=conversation_id)


class NanobotBaseConversationEntity(ConversationEntity):
    """Base nanobot conversation entity.

    Subclasses apply platform-specific registration behavior (e.g. agent
    registration in the conversation platform).
    """

    _attr_has_entity_name = True
    _attr_name = "nanobot"
    _attr_supported_features = 0
    _attr_supports_streaming = False

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

        self._attr_unique_id = entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": "nanobot",
            "manufacturer": "yyamanoi1222",
            "model": "nanobot",
            "entry_type": "service",
        }

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a user message via nanobot.

        Home Assistant's Assist pipeline (especially voice) supplies a new
        ``conversation_id`` on every turn. To preserve context across turns we
        stabilize the nanobot ``session_id`` by falling back to the
        ``device_id`` when no meaningful conversation ID is present, or by
        using the caller-supplied ``conversation_id`` as-is.

        nanobot's API requires exactly one user message per request; we
        always send only the latest user utterance and rely on session_id
        for continuity.
        """
        text = (user_input.text or "").strip()
        if not text:
            intent_response = IntentResponse(language=user_input.language)
            intent_response.async_set_speech("I didn't catch that.")
            return _make_result(intent_response, None, False)

        # Stabilize session_id for Assist pipelines that rotate conversation_id.
        session_id = user_input.conversation_id
        if not session_id:
            session_id = user_input.device_id
        if not session_id:
            session_id = f"{DOMAIN}-{self._entry_id}"

        session = async_get_clientsession(self.hass)
        client = NanobotClient(
            session=session,
            api_url=self._api_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        try:
            response_text = await client.send_message(
                text=text,
                session_id=session_id,
                model=self._model,
            )
        except NanobotClientError as err:
            LOGGER.error("nanobot conversation error: %s", err)
            intent_response = IntentResponse(language=user_input.language)
            intent_response.async_set_speech("Sorry, I could not reach nanobot.")
            return _make_result(intent_response, session_id, False)

        intent_response = IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return _make_result(intent_response, session_id, True)
