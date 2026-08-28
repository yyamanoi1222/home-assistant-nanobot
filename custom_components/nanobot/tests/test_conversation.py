"""Tests for the nanobot conversation entity."""

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.conversation import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.nanobot.const import CONF_API_URL, CONF_MODEL, CONF_TIMEOUT

TEST_CONFIG = {
    CONF_API_URL: "http://127.0.0.1:8900",
    CONF_API_KEY: "",
    CONF_TIMEOUT: 120,
}


def _make_input(text: str, conversation_id: str | None = None) -> ConversationInput:
    """Create a ConversationInput for tests."""
    return ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id=conversation_id,
        device_id=None,
        language="ja",
        agent_id="test-agent",
    )


async def test_conversation_entity_process(hass: HomeAssistant) -> None:
    """Test the conversation entity processes input correctly."""
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=TEST_CONFIG
    )

    user_input = _make_input("Hello", conversation_id="test-conversation")

    with patch(
        "custom_components.nanobot.entity.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.nanobot.entity.NanobotClient.send_message",
        new=AsyncMock(return_value="Hi there!"),
    ):
        result = await entity.async_process(user_input)

    assert isinstance(result, ConversationResult)
    assert result.response.speech["plain"]["speech"] == "Hi there!"


async def test_conversation_entity_error(hass: HomeAssistant) -> None:
    """Test the conversation entity returns an error message on failure."""
    from custom_components.nanobot.conversation import NanobotConversationEntity
    from custom_components.nanobot.nanobot_client import NanobotConnectionError

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=TEST_CONFIG
    )

    user_input = _make_input("Hello", conversation_id="test-conversation")

    with patch(
        "custom_components.nanobot.entity.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.nanobot.entity.NanobotClient.send_message",
        new=AsyncMock(side_effect=NanobotConnectionError("down")),
    ):
        result = await entity.async_process(user_input)

    assert isinstance(result, ConversationResult)
    assert "could not reach nanobot" in result.response.speech["plain"]["speech"]


async def test_conversation_entity_session_id_mapping(hass: HomeAssistant) -> None:
    """Test conversation_id is passed as session_id to nanobot client."""
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=TEST_CONFIG
    )

    user_input = _make_input("Hello", conversation_id="my-session-id")

    send_message = AsyncMock(return_value="OK")
    with patch(
        "custom_components.nanobot.entity.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.nanobot.entity.NanobotClient.send_message",
        new=send_message,
    ):
        await entity.async_process(user_input)

    send_message.assert_awaited_once()
    call_kwargs = send_message.call_args.kwargs
    assert call_kwargs["session_id"] == "my-session-id"


async def test_conversation_entity_model_mapping(hass: HomeAssistant) -> None:
    """Test model from config is passed to nanobot client."""
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    config_with_model = {
        **TEST_CONFIG,
        CONF_MODEL: "MiniMax-M2.7",
    }
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=config_with_model
    )

    user_input = _make_input("Hello", conversation_id="my-session-id")

    send_message = AsyncMock(return_value="OK")
    with patch(
        "custom_components.nanobot.entity.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.nanobot.entity.NanobotClient.send_message",
        new=send_message,
    ):
        await entity.async_process(user_input)

    send_message.assert_awaited_once()
    call_kwargs = send_message.call_args.kwargs
    assert call_kwargs["model"] == "MiniMax-M2.7"
