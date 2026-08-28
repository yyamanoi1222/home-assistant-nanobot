"""Conversation platform setup for nanobot."""

from __future__ import annotations

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .entity import NanobotBaseConversationEntity
from .typing import NanobotRuntimeData


class NanobotConversationEntity(NanobotBaseConversationEntity):
    """nanobot conversation agent registered as a platform entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        config: dict,
    ) -> None:
        """Initialize the agent and keep a reference to its config entry."""
        super().__init__(hass, config_entry.entry_id, config)
        self._config_entry = config_entry

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    async def async_added_to_hass(self) -> None:
        """Register agent when entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(
            self.hass,
            self._config_entry,
            self,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unregister agent when entity is removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self._config_entry)
        await super().async_will_remove_from_hass()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up nanobot conversation entity."""
    runtime_data: NanobotRuntimeData = config_entry.runtime_data
    async_add_entities(
        [NanobotConversationEntity(hass, config_entry, runtime_data.config)],
    )
