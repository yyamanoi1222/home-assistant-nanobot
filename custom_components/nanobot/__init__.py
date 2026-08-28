"""The nanobot integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN, LOGGER
from .typing import NanobotRuntimeData

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


type NanobotConfigEntry = ConfigEntry[NanobotRuntimeData]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the nanobot component."""
    LOGGER.debug("Setting up %s integration", DOMAIN)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: NanobotConfigEntry) -> bool:
    """Set up nanobot from a config entry."""
    entry.runtime_data = NanobotRuntimeData(
        config=dict(entry.data), title=entry.title
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: NanobotConfigEntry) -> bool:
    """Unload a nanobot config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_listener(hass: HomeAssistant, entry: NanobotConfigEntry) -> None:
    """Handle options update."""
    entry.runtime_data = NanobotRuntimeData(
        config=dict(entry.data), title=entry.title
    )
    await hass.config_entries.async_reload(entry.entry_id)
