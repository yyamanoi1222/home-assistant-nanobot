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


def _make_input(
    text: str,
    conversation_id: str | None = None,
    device_id: str | None = None,
) -> ConversationInput:
    """Create a ConversationInput for tests."""
    from dataclasses import fields

    kwargs: dict = {
        "text": text,
        "context": MagicMock(),
        "conversation_id": conversation_id,
        "device_id": device_id,
        "language": "ja",
        "agent_id": "test-agent",
    }
    names = {f.name for f in fields(ConversationInput)}
    if "satellite_id" in names:
        kwargs["satellite_id"] = None
    if "extra_system_prompt" in names:
        kwargs["extra_system_prompt"] = None
    return ConversationInput(**kwargs)


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
    assert result.conversation_id == "test-conversation"


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


async def test_conversation_entity_session_id_falls_back_to_device_id(
    hass: HomeAssistant,
) -> None:
    """Test session_id falls back to device_id when conversation_id is absent."""
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=TEST_CONFIG
    )

    user_input = _make_input("Hello", conversation_id=None, device_id="kitchen-speaker")

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
    assert call_kwargs["session_id"] == "kitchen-speaker"


async def test_conversation_entity_session_id_falls_back_to_entry_id(
    hass: HomeAssistant,
) -> None:
    """Test session_id falls back to entry-scoped id when neither id is present."""
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    entity = NanobotConversationEntity(
        hass, config_entry=config_entry, config=TEST_CONFIG
    )

    user_input = _make_input("Hello", conversation_id=None, device_id=None)

    send_message = AsyncMock(return_value="OK")
    with patch(
        "custom_components.nanobot.entity.async_get_clientsession",
        return_value=MagicMock(),
    ), patch(
        "custom_components.nanobot.entity.NanobotClient.send_message",
        new=send_message,
    ):
        result = await entity.async_process(user_input)

    send_message.assert_awaited_once()
    call_kwargs = send_message.call_args.kwargs
    assert call_kwargs["session_id"] == "nanobot-test-entry"
    assert result.conversation_id == "nanobot-test-entry"


async def _process_with_reply(hass: HomeAssistant, reply: str):
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
        new=AsyncMock(return_value=reply),
    ):
        return await entity.async_process(user_input)


async def test_continue_conversation_always_on_success(hass: HomeAssistant) -> None:
    from dataclasses import fields

    from homeassistant.components.conversation import ConversationResult

    if "continue_conversation" not in {f.name for f in fields(ConversationResult)}:
        return
    for reply in ("How can I help?", "Done.", "こんにちは"):
        result = await _process_with_reply(hass, reply)
        assert result.continue_conversation is True
