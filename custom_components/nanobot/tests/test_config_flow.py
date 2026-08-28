"""Tests for the nanobot config flow."""

from unittest.mock import patch

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.nanobot.config_flow import (
    NanobotConfigFlow,
    STEP_USER_DATA_SCHEMA,
)
from custom_components.nanobot.const import (
    CONF_API_URL,
    CONF_MODEL,
    CONF_TIMEOUT,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT,
    DOMAIN,
)


def _make_flow(hass: HomeAssistant) -> NanobotConfigFlow:
    """Create a config flow instance without integration dependencies."""
    flow = NanobotConfigFlow()
    flow.hass = hass
    FlowResultType.FORM
    flow.context = {"source": config_entries.SOURCE_USER}
    flow.flow_id = "test-flow-id"
    return flow


async def test_user_flow_success(hass: HomeAssistant) -> None:
    """Test successful config flow."""
    flow = _make_flow(hass)

    with patch(
        "custom_components.nanobot.config_flow._validate_input",
        return_value=None,
    ):
        result = await flow.async_step_user(
            {
                CONF_API_URL: DEFAULT_API_URL,
                CONF_API_KEY: "",
                CONF_MODEL: "MiniMax-M2.7",
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
            }
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "127.0.0.1:8900"
    assert result["data"] == {
        CONF_API_URL: DEFAULT_API_URL,
        CONF_API_KEY: "",
        CONF_MODEL: "MiniMax-M2.7",
        CONF_TIMEOUT: DEFAULT_TIMEOUT,
    }


async def test_user_flow_cannot_connect(hass: HomeAssistant) -> None:
    """Test config flow handles connection error."""
    flow = _make_flow(hass)

    from custom_components.nanobot.nanobot_client import NanobotConnectionError

    with patch(
        "custom_components.nanobot.config_flow._validate_input",
        side_effect=NanobotConnectionError("Connection error"),
    ):
        result = await flow.async_step_user(
            {
                CONF_API_URL: "http://invalid:8900",
                CONF_API_KEY: "",
                CONF_TIMEOUT: DEFAULT_TIMEOUT,
            }
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_already_configured(hass: HomeAssistant) -> None:
    """Test duplicate entries are aborted."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="nanobot",
        data={
            CONF_API_URL: DEFAULT_API_URL,
            CONF_API_KEY: "",
            CONF_TIMEOUT: DEFAULT_TIMEOUT,
        },
    )
    entry.add_to_hass(hass)

    flow = _make_flow(hass)

    try:
        with patch(
            "custom_components.nanobot.config_flow._validate_input",
            return_value=None,
        ):
            result = await flow.async_step_user(
                {
                    CONF_API_URL: DEFAULT_API_URL,
                    CONF_API_KEY: "",
                    CONF_TIMEOUT: DEFAULT_TIMEOUT,
                }
            )
    except Exception as exc:
        from homeassistant.data_entry_flow import AbortFlow

        assert isinstance(exc, AbortFlow)
        assert exc.reason == "already_configured"
        return

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


def test_step_user_data_schema() -> None:
    """Test the data schema defaults."""
    schema = STEP_USER_DATA_SCHEMA
def test_step_user_data_schema_optional_model() -> None:
    """Test the data schema accepts model as optional."""
    schema = STEP_USER_DATA_SCHEMA
    assert schema(
        {
            CONF_API_URL: DEFAULT_API_URL,
            CONF_TIMEOUT: DEFAULT_TIMEOUT,
        }
    )
    assert schema(
        {
            CONF_API_URL: DEFAULT_API_URL,
            CONF_TIMEOUT: DEFAULT_TIMEOUT,
            CONF_MODEL: "MiniMax-M2.7",
        }
    )
