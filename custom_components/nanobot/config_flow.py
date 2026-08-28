"""Config flow for nanobot integration."""

from __future__ import annotations

from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import (
    CONF_API_URL,
    CONF_MODEL,
    CONF_TIMEOUT,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT,
    DOMAIN,
    LOGGER,
)
from .nanobot_client import (
    NanobotAPIError,
    NanobotClient,
    NanobotClientError,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_URL, default=DEFAULT_API_URL): str,
        vol.Optional(CONF_API_KEY): str,
        vol.Optional(CONF_MODEL): str,
        vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
            NumberSelector(
                NumberSelectorConfig(
                    min=1, max=600, step=1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Coerce(int),
        ),
    }
)


async def _validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input by calling nanobot's /health endpoint."""
    session = async_get_clientsession(hass)
    client = NanobotClient(
        session=session,
        api_url=data[CONF_API_URL],
        api_key=data.get(CONF_API_KEY),
        timeout=data[CONF_TIMEOUT],
    )
    await client.validate_connection()


class NanobotConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for nanobot."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(
                {CONF_API_URL: user_input[CONF_API_URL]}
            )
            try:
                await _validate_input(self.hass, user_input)
            except aiohttp.ClientResponseError as err:
                if err.status in (401, 403):
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "cannot_connect"
            except (NanobotClientError, aiohttp.ClientError, TimeoutError):
                errors["base"] = "cannot_connect"
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Unexpected exception during nanobot config flow")
                errors["base"] = "unknown"
            else:
                title = user_input[CONF_API_URL].rstrip("/").split("//", 1)[-1]
                return self.async_create_entry(title=title, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )
