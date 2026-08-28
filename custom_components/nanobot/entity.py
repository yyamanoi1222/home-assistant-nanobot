"""Base conversation entity for nanobot."""

from __future__ import annotations

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.intent import IntentResponse

from .const import CONF_API_KEY, CONF_API_URL, CONF_TIMEOUT, DOMAIN, LOGGER
from .nanobot_client import NanobotClient, NanobotClientError


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
            config: Entry data containing api_url, api_key, and timeout.
        """
        self.hass = hass
        self._entry_id = entry_id
        self._api_url = config[CONF_API_URL]
        self._api_key = config.get(CONF_API_KEY)
        self._timeout = config.get(CONF_TIMEOUT, 120)

        self._attr_unique_id = entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": "nanobot",
            "manufacturer": "HKUDS",
            "model": "nanobot",
            "entry_type": "service",
        }

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a user message via nanobot.

        We map Home Assistant's conversation_id directly to nanobot's
        session_id so that nanobot persists the conversation context.
        """
        session = async_get_clientsession(self.hass)
        client = NanobotClient(
            session=session,
            api_url=self._api_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        try:
            response_text = await client.send_message(
                text=user_input.text,
                session_id=user_input.conversation_id,
            )
        except NanobotClientError as err:
            LOGGER.error("nanobot conversation error: %s", err)
            response_text = "Sorry, I could not reach nanobot."

        intent_response = IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return ConversationResult(response=intent_response)
