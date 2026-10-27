"""Tests for the nanobot conversation entity."""

import asyncio
from unittest.mock import MagicMock, patch

from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationInput,
    ConversationResult,
    ToolResultContent,
    UserContent,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.nanobot.const import (
    CONF_API_URL,
    CONF_FILLER_TEXT,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_VOICE_PROMPT,
    DEFAULT_VOICE_PROMPT,
)
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
    """Return a stream_voice_turn replacement that yields ``text`` in chunks."""

    async def _stream(*args, **kwargs):
        yield {"type": "turn.started"}
        for index in range(0, len(text), 5):
            yield {
                "type": "segment",
                "text": text[index : index + 5],
                "final": index + 5 >= len(text),
            }
        yield {"type": "turn.completed", "content": text}

    return _stream


class _RecordingStream:
    """Record stream_voice_turn calls and stream a fixed reply."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self):
        yield {"type": "turn.started"}
        for index in range(0, len(self.text), 5):
            yield {
                "type": "segment",
                "text": self.text[index : index + 5],
                "final": index + 5 >= len(self.text),
            }
        yield {"type": "turn.completed", "content": self.text}


class _FirstTurnCancelledStream:
    """Cancel the first turn and complete subsequent turns."""

    def __init__(self, text: str = "OK") -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._stream(cancel=len(self.calls) == 1)

    async def _stream(self, *, cancel: bool):
        yield {"type": "turn.started"}
        yield {"type": "segment", "text": self.text, "final": True}
        if cancel:
            yield {"type": "turn.cancelled"}
        else:
            yield {"type": "turn.completed", "content": self.text}


def _make_entity(hass: HomeAssistant, config: dict | None = None):
    from custom_components.nanobot.conversation import NanobotConversationEntity

    config_entry = MagicMock()
    config_entry.entry_id = "test-entry"
    return NanobotConversationEntity(
        hass, config_entry=config_entry, config=config or TEST_CONFIG
    )


async def _process(entity, user_input: ConversationInput) -> ConversationResult:
    """Run one turn through the entity."""
    result, _ = await _process_with_chat_log(entity, user_input)
    return result


async def _process_with_chat_log(
    entity, user_input: ConversationInput
) -> tuple[ConversationResult, ChatLog]:
    """Run one turn through the entity and return the chat log as well."""
    chat_log = _make_chat_log(
        entity.hass,
        user_input.text,
        user_input.conversation_id or "test-conversation",
    )
    result = await entity._async_handle_message(user_input, chat_log)
    return result, chat_log


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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_streaming_reply("Hi there!"),
        ),
    ):
        result = await _process(entity, user_input)

    assert isinstance(result, ConversationResult)
    assert result.response.speech["plain"]["speech"] == "Hi there!"
    assert result.conversation_id == "test-conversation"


async def test_tool_filler_is_spoken_once(hass: HomeAssistant) -> None:
    """Test a filler is spoken when a tool starts before any reply text."""
    entity = _make_entity(hass, {**TEST_CONFIG, CONF_FILLER_TEXT: "確認しますね。"})
    user_input = _make_input("天気は?", conversation_id="test-conversation")

    async def _tool_stream(*args, **kwargs):
        yield {"type": "tool", "name": "web_search", "state": "started"}
        yield {"type": "tool", "name": "web_search", "state": "completed"}
        yield {"type": "segment", "text": "晴れです。", "final": True}
        yield {"type": "turn.completed", "content": "晴れです。"}

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_tool_stream,
        ),
    ):
        result = await _process(entity, user_input)

    assert result.response.speech["plain"]["speech"] == "確認しますね。 晴れです。"


async def test_tool_events_are_forwarded_as_external_tool_calls(
    hass: HomeAssistant,
) -> None:
    """Test tool events become external tool call and result deltas."""
    entity = _make_entity(hass, {**TEST_CONFIG, CONF_FILLER_TEXT: "確認しますね。"})
    user_input = _make_input("天気は?", conversation_id="test-conversation")

    async def _tool_stream(*args, **kwargs):
        yield {
            "type": "tool",
            "name": "web_search",
            "state": "started",
            "tool_call_id": "call-1",
        }
        yield {
            "type": "tool",
            "name": "web_search",
            "state": "completed",
            "tool_call_id": "call-1",
        }
        yield {"type": "segment", "text": "晴れです。", "final": True}
        yield {"type": "turn.completed", "content": "晴れです。"}

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_tool_stream,
        ),
    ):
        _result, chat_log = await _process_with_chat_log(entity, user_input)

    assistant = [
        content for content in chat_log.content if isinstance(content, AssistantContent)
    ]
    assert assistant[0].content == "確認しますね。"
    assert assistant[0].tool_calls is not None
    assert assistant[0].tool_calls[0].tool_name == "web_search"
    assert assistant[0].tool_calls[0].external is True
    assert assistant[0].tool_calls[0].id == "call-1"

    tool_results = [
        content
        for content in chat_log.content
        if isinstance(content, ToolResultContent)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "call-1"
    assert tool_results[0].tool_name == "web_search"
    assert tool_results[0].result.error is False

    assert assistant[-1].content == "晴れです。"


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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
                "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(
            entity,
            _make_input("明日の天気は？", conversation_id="test-conversation"),
        )
        await _process(entity, _make_input("はい", conversation_id="test-conversation"))

    assert len(stream.calls) == 2


async def _send_and_get_message(hass: HomeAssistant, config: dict | None = None) -> str:
    """Run one turn and return the text sent to nanobot."""
    entity = _make_entity(hass, config)
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(
            entity, _make_input("Hello", conversation_id="test-conversation")
        )

    return stream.calls[0]["text"]


async def test_default_voice_prompt_is_prepended(hass: HomeAssistant) -> None:
    """Test the default voice prompt is prepended to the user message."""
    message = await _send_and_get_message(hass)

    assert message == f"{DEFAULT_VOICE_PROMPT}\n\nHello"


async def test_custom_voice_prompt_is_prepended(hass: HomeAssistant) -> None:
    """Test a configured voice prompt replaces the default."""
    message = await _send_and_get_message(
        hass, {**TEST_CONFIG, CONF_VOICE_PROMPT: "短く話してください。"}
    )

    assert message == "短く話してください。\n\nHello"


async def test_empty_voice_prompt_is_disabled(hass: HomeAssistant) -> None:
    """Test an empty voice prompt sends the user message unchanged."""
    message = await _send_and_get_message(hass, {**TEST_CONFIG, CONF_VOICE_PROMPT: ""})

    assert message == "Hello"


async def test_voice_prompt_only_on_first_turn(hass: HomeAssistant) -> None:
    """Test the voice prompt is only prepended to the first turn of a session."""
    entity = _make_entity(hass)
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(entity, _make_input("Hello", conversation_id="same"))
        await _process(entity, _make_input("Again", conversation_id="same"))

    assert stream.calls[0]["text"] == f"{DEFAULT_VOICE_PROMPT}\n\nHello"
    assert stream.calls[1]["text"] == "Again"


async def test_voice_prompt_reapplied_after_cancelled_first_turn(
    hass: HomeAssistant,
) -> None:
    """Test a cancelled first turn does not mark the prompt as sent."""
    entity = _make_entity(hass)
    stream = _FirstTurnCancelledStream()

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(entity, _make_input("Hello", conversation_id="same"))
        await _process(entity, _make_input("Again", conversation_id="same"))

    assert stream.calls[0]["text"] == f"{DEFAULT_VOICE_PROMPT}\n\nHello"
    assert stream.calls[1]["text"] == f"{DEFAULT_VOICE_PROMPT}\n\nAgain"


async def test_voice_prompt_reapplied_for_new_session(hass: HomeAssistant) -> None:
    """Test a new session receives the voice prompt again."""
    entity = _make_entity(hass)
    stream = _RecordingStream("OK")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(entity, _make_input("Hello", conversation_id="one"))
        await _process(entity, _make_input("Again", conversation_id="two"))

    assert stream.calls[0]["text"] == f"{DEFAULT_VOICE_PROMPT}\n\nHello"
    assert stream.calls[1]["text"] == f"{DEFAULT_VOICE_PROMPT}\n\nAgain"


async def test_conversation_entity_ignores_partial_echo(
    hass: HomeAssistant,
) -> None:
    """Test a partial echo of the previous reply is not sent to nanobot."""
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
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=stream,
        ),
    ):
        await _process(
            entity,
            _make_input("明日の天気は？", conversation_id="test-conversation"),
        )
        second = await _process(
            entity,
            _make_input("気温は25度になります", conversation_id="test-conversation"),
        )

    assert len(stream.calls) == 1
    assert second.continue_conversation is False


async def test_fast_reply_does_not_emit_tool_call(
    hass: HomeAssistant,
) -> None:
    """Test a fast reply is emitted without synthetic tool calls."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id="test-conversation")

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_streaming_reply("Hi there!"),
        ),
    ):
        result, chat_log = await _process_with_chat_log(entity, user_input)

    assistant = [
        content for content in chat_log.content if isinstance(content, AssistantContent)
    ]
    assert len(assistant) == 1
    assert assistant[0].tool_calls is None
    assert result.response.speech["plain"]["speech"] == "Hi there!"


async def test_slow_stream_is_not_cut_off(hass: HomeAssistant) -> None:
    """Test a pause after streamed text does not end the reply."""
    entity = _make_entity(hass)
    user_input = _make_input("天気は?", conversation_id="test-conversation")

    async def _slow_stream(*args, **kwargs):
        yield {"type": "turn.started"}
        yield {"type": "segment", "text": "Webで調べますね。"}
        await asyncio.sleep(0.05)
        yield {"type": "segment", "text": "晴れです。"}
        yield {
            "type": "turn.completed",
            "content": "Webで調べますね。晴れです。",
        }

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_slow_stream,
        ),
    ):
        result, chat_log = await _process_with_chat_log(entity, user_input)

    assistant = [
        content for content in chat_log.content if isinstance(content, AssistantContent)
    ]

    assert [content.content for content in assistant] == [
        "Webで調べますね。晴れです。",
    ]
    assert all(content.tool_calls is None for content in assistant)
    assert result.response.speech["plain"]["speech"] == "Webで調べますね。晴れです。"


async def test_first_token_timeout_returns_error(hass: HomeAssistant) -> None:
    """Test a stream that never starts responding is reported as an error."""
    entity = _make_entity(hass)
    entity._timeout = 0.01
    user_input = _make_input("Hello", conversation_id="test-conversation")

    async def _slow_stream(*args, **kwargs):
        await asyncio.sleep(0.05)
        yield {"type": "segment", "text": "too late"}
        yield {"type": "turn.completed", "content": "too late"}

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_slow_stream,
        ),
    ):
        result = await _process(entity, user_input)

    assert "could not reach nanobot" in result.response.speech["plain"]["speech"]


async def test_empty_stream_returns_error(hass: HomeAssistant) -> None:
    """Test an empty nanobot stream is reported as an error."""
    entity = _make_entity(hass)
    user_input = _make_input("Hello", conversation_id="test-conversation")

    async def _empty_stream(*args, **kwargs):
        if False:
            yield ""

    with (
        patch(
            "custom_components.nanobot.entity.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.nanobot.entity.NanobotClient.stream_voice_turn",
            new=_empty_stream,
        ),
    ):
        result = await _process(entity, user_input)

    assert "could not reach nanobot" in result.response.speech["plain"]["speech"]
