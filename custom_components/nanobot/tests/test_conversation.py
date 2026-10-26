"""Tests for the nanobot conversation entity."""

from unittest.mock import MagicMock, patch

from homeassistant.components.conversation import (
    ChatLog,
    ConversationInput,
    ConversationResult,
    UserContent,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.nanobot.const import CONF_API_URL, CONF_MODEL, CONF_TIMEOUT
from custom_components.nanobot.nanobot_client import NanobotConnectionError

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


def _make_chat_log(
    hass: HomeAssistant,
    text: str,
    conversation_id: str = "test-conversation",
) -> ChatLog:
    """Create a chat log with the user message already added."""
    chat_log = ChatLog(hass, conversation_id)
    chat_log.async_add_user_content(UserContent(content=text))
    return chat_log


def _streaming_reply(text: str):
    """Return a send_message_stream replacement that yields ``text`` in chunks."""

    async def _stream(*args, **kwargs):
        for index in range(0, len(text), 5):
            yield text[index : index + 5]

    return _stream


class _RecordingStream:
    """Record send_message_stream calls and stream a fixed reply."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self):
        for index in range(0, len(self.text), 5):
            yield self.text[index : index + 5]


def _make_entity(hass: HomeAssistant, config: dict | None = None):
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    return NanobotConversationEntity(
        hass, config_entry=config_entry, config=config or TEST_CONFIG
    )


async def _process(entity, user_input: ConversationInput) -> ConversationResult:
    """Run one turn through the entity."""
    chat_log = _make_chat_log(
        entity.hass,
        user_input.text,
        user_input.conversation_id or "test-conversation",
    )
    return await entity._async_handle_message(user_input, chat_log)


async def test_conversation_entity_process(hass: HomeAssistant) -> None:
    """Test the conversation entity processes input correctly."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id="test-conversation")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=_streaming_reply("Hi there!"),
        ),
    ):
        result = await _process(entity, user_input)

    assert isinstance(result, ConversationResult)
    assert result.response.speech["plain"]["speech"] == "Hi there!"
    assert result.conversation_id == "test-conversation"


async def test_conversation_entity_error(hass: HomeAssistant) -> None:
    """Test the conversation entity returns an error message on failure."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id="test-conversation")

    async def _failing_stream(*args, **kwargs):
        raise NanobotConnectionError("down")
        yield  # pragma: no cover

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=_failing_stream,
        ),
    ):
        result = await _process(entity, user_input)

    assert isinstance(result, ConversationResult)
    assert "could not reach nanobot" in result.response.speech["plain"]["speech"]


async def test_conversation_entity_session_id_mapping(hass: HomeAssistant) -> None:
    """Test conversation_id is passed as session_id to nanobot client."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id="my-session-id")
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=stream,
        ),
    ):
        await _process(entity, user_input)

    assert len(stream.calls) == 1
    assert stream.calls[0]["session_id"] == "my-session-id"


async def test_conversation_entity_model_mapping(hass: HomeAssistant) -> None:
    """Test model from config is passed to nanobot client."""
    entity = _make_entity(hass, {**TEST_CONFIG, CONF_MODEL: "MiniMax-M2.7"})
    user_input = _make_input("Hello", conversation_id="my-session-id")
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=stream,
        ),
    ):
        await _process(entity, user_input)

    assert len(stream.calls) == 1
    assert stream.calls[0]["model"] == "MiniMax-M2.7"


async def test_conversation_entity_session_id_falls_back_to_device_id(
    hass: HomeAssistant,
) -> None:
    """Test session_id falls back to device_id when conversation_id is absent."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id=None, device_id="kitchen-speaker")
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=stream,
        ),
    ):
        await _process(entity, user_input)

    assert len(stream.calls) == 1
    assert stream.calls[0]["session_id"] == "kitchen-speaker"


async def test_conversation_entity_session_id_falls_back_to_entry_id(
    hass: HomeAssistant,
) -> None:
    """Test session_id falls back to entry-scoped id when neither id is present."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id=None, device_id=None)

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=_streaming_reply("OK"),
        ),
    ):
        result = await _process(entity, user_input)

    assert result.conversation_id == "nanobot-test-entry"


async def test_continue_conversation_always_on_success(hass: HomeAssistant) -> None:
    from dataclasses import fields

    if "continue_conversation" not in {f.name for f in fields(ConversationResult)}:
        return

    for reply in ("How can I help?", "Done.", "こんにちは", "以上です。"):
        entity = _make_entity(hass)
        with (
            patch(
                "custom_components.nanobot.entity.async_get_clientsession",
                return_value=MagicMock(),
            ),
            patch(
                "custom_components.nanobot.entity.NanobotClient.send_message_stream",
                new=_streaming_reply(reply),
            ),
        ):
            result = await _process(
                entity, _make_input("Hello", conversation_id="test-conversation")
            )
        assert result.continue_conversation is True


async def test_conversation_entity_ignores_echoed_reply(hass: HomeAssistant) -> None:
    """Test a reply echoed back by speech-to-text is not sent to nanobot."""
    entity = _make_entity(hass)
    stream = _RecordingStream(
        "明日の天気は晴れです。気温は25度になります。他に知りたいことはありますか？"
    )

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=stream,
        ),
    ):
        first = await _process(
            entity,
            _make_input("明日の天気は？", conversation_id="test-conversation"),
        )
        second = await _process(
            entity,
            _make_input(
                "明日の天気は晴れです、気温は25度になります。他に知りたいことはありますか",
                conversation_id="new-conversation-id",
            ),
        )

    assert len(stream.calls) == 1

    from dataclasses import fields

    if "continue_conversation" in {f.name for f in fields(ConversationResult)}:
        assert first.continue_conversation is True
        assert second.continue_conversation is False


async def test_conversation_entity_does_not_drop_short_reply(
    hass: HomeAssistant,
) -> None:
    """Test short follow-ups are not mistaken for echoes."""
    entity = _make_entity(hass)
    stream = _RecordingStream("明日の天気は晴れです。気温は25度になります。")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.send_message_stream",
            new=stream,
        ),
    ):
        await _process(
            entity,
            _make_input("明日の天気は？", conversation_id="test-conversation"),
        )
        await _process(entity, _make_input("はい", conversation_id="test-conversation"))

    assert len(stream.calls) == 2
