"""Base conversation entity for nanobot."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncGenerator
from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

from homeassistant.components.conversation import (
    AssistantContent,
    AssistantContentDeltaDict,
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
    ToolResultContent,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.intent import IntentResponse

from .const import (
    ACTIVITY_TOOL_NAME,
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_VOICE_PROMPT,
    DEFAULT_VOICE_PROMPT,
    DOMAIN,
    LOGGER,
    STREAM_IDLE_TIMEOUT,
)
from .nanobot_client import NanobotAPIError, NanobotClient, NanobotClientError

if TYPE_CHECKING:
    from homeassistant.components.conversation import ToolResultContentDeltaDict

_CONVERSATION_RESULT_FIELDS = {f.name for f in fields(ConversationResult)}
_TOOL_INPUT_FIELDS = {f.name for f in fields(llm.ToolInput)}
_TOOL_RESULT_FIELDS = {f.name for f in fields(ToolResultContent)}
_SUPPORTS_EXTERNAL_TOOL_CALLS = "external" in _TOOL_INPUT_FIELDS
_TOOL_RESULT_USES_RESULT = "result" in _TOOL_RESULT_FIELDS
_ECHO_MIN_LENGTH = 8
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_echo(text: str) -> str:
    """Normalize text so spoken echo can be compared with a reply."""
    return _NON_WORD_RE.sub("", text.lower())


def _looks_like_echo(text: str, last_response: str | None) -> bool:
    """Return True when user input is part of the previous reply."""
    if not last_response:
        return False

    normalized_text = _normalize_for_echo(text)
    if len(normalized_text) < _ECHO_MIN_LENGTH:
        return False

    return normalized_text in _normalize_for_echo(last_response)


def _make_result(
    response: IntentResponse,
    conversation_id: str | None,
    continue_conversation: bool = False,
) -> ConversationResult:
    if "continue_conversation" in _CONVERSATION_RESULT_FIELDS:
        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )
    return ConversationResult(response=response, conversation_id=conversation_id)


def _make_speech_result(
    language: str,
    speech: str,
    conversation_id: str | None,
    continue_conversation: bool = False,
) -> ConversationResult:
    """Build a result that speaks a complete message."""
    intent_response = IntentResponse(language=language)
    intent_response.async_set_speech(speech)
    return _make_result(intent_response, conversation_id, continue_conversation)


def _turn_assistant_text(chat_log: ChatLog, start_index: int) -> str | None:
    """Return all assistant text added to the chat log during this turn."""
    texts = [
        content.content
        for content in chat_log.content[start_index:]
        if isinstance(content, AssistantContent) and content.content
    ]
    if not texts:
        return None
    return " ".join(texts)


def _make_activity_tool_call() -> llm.ToolInput:
    """Build the synthetic tool call used to start TTS during a silent wait."""
    return llm.ToolInput(
        tool_name=ACTIVITY_TOOL_NAME,
        tool_args={},
        external=True,
    )


def _make_tool_result_delta(
    tool_call: llm.ToolInput,
) -> ToolResultContentDeltaDict:
    """Build a tool result delta that closes the synthetic tool call."""
    delta: dict[str, Any] = {
        "role": "tool_result",
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.tool_name,
    }
    if _TOOL_RESULT_USES_RESULT:
        delta["result"] = llm.ToolResult(data={"status": "completed"})
    else:
        delta["tool_result"] = {"status": "completed"}
    return delta


class NanobotBaseConversationEntity(ConversationEntity):
    """Base nanobot conversation entity.

    Subclasses apply platform-specific registration behavior (e.g. agent
    registration in the conversation platform).
    """

    _attr_has_entity_name = True
    _attr_name = "nanobot"
    _attr_supported_features = 0
    _attr_supports_streaming = True

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
        self._voice_prompt = config.get(CONF_VOICE_PROMPT, DEFAULT_VOICE_PROMPT)
        self._prompted_sessions: set[str] = set()
        self._last_response_text: str | None = None

        self._attr_unique_id = entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": "nanobot",
            "manufacturer": "yyamanoi1222",
            "model": "nanobot",
            "entry_type": "service",
        }

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process a user message via nanobot.

        Home Assistant's Assist pipeline (especially voice) supplies a new
        ``conversation_id`` on every turn. To preserve context across turns we
        stabilize the nanobot ``session_id`` by falling back to the
        ``device_id`` when no meaningful conversation ID is present, or by
        using the caller-supplied ``conversation_id`` as-is.

        nanobot's API requires exactly one user message per request; we
        always send only the latest user utterance and rely on session_id
        for continuity. The configured voice prompt is prepended to the first
        request of each session. Reply tokens are streamed into the chat log
        so the Assist pipeline can start TTS before the full answer is
        generated. When nanobot goes silent for a while, a synthetic external
        tool call is emitted so Home Assistant starts speaking the buffered
        text instead of waiting for the 60 character streaming threshold.
        Successful replies keep the conversation open so Assist listens for a
        follow-up. If speech-to-text echoes part of the previous reply back,
        the input is dropped instead of being sent to nanobot again.
        """
        text = (user_input.text or "").strip()
        if not text:
            return _make_speech_result(
                user_input.language, "I didn't catch that.", None, False
            )

        session_id = self._resolve_session_id(user_input)

        if _looks_like_echo(text, self._last_response_text):
            LOGGER.debug("Ignoring possible echo of the previous reply: %s", text)
            return _make_speech_result(user_input.language, "", session_id, False)

        session = async_get_clientsession(self.hass)
        client = NanobotClient(
            session=session,
            api_url=self._api_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        start_index = len(chat_log.content)
        try:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._async_stream_reply(client, text, session_id),
            ):
                pass
        except NanobotClientError as err:
            LOGGER.error("nanobot conversation error: %s", err)
            return _make_speech_result(
                user_input.language,
                "Sorry, I could not reach nanobot.",
                session_id,
                False,
            )

        response_text = _turn_assistant_text(chat_log, start_index)
        if not response_text:
            LOGGER.error("nanobot returned an empty reply")
            return _make_speech_result(
                user_input.language,
                "Sorry, I could not reach nanobot.",
                session_id,
                False,
            )

        self._last_response_text = response_text
        return _make_speech_result(user_input.language, response_text, session_id, True)

    def _resolve_session_id(self, user_input: ConversationInput) -> str:
        """Return the nanobot session ID for a conversation turn."""
        return (
            user_input.conversation_id
            or user_input.device_id
            or f"{DOMAIN}-{self._entry_id}"
        )

    @staticmethod
    async def _async_pump_stream(
        stream: AsyncGenerator[str, None],
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> None:
        """Move nanobot tokens into a queue so consumers can apply timeouts."""
        try:
            async for token in stream:
                await queue.put(("token", token))
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            await queue.put(("error", err))
        else:
            await queue.put(("done", None))

    async def _async_stream_reply(
        self,
        client: NanobotClient,
        text: str,
        session_id: str,
    ) -> AsyncGenerator[AssistantContentDeltaDict | ToolResultContentDeltaDict, None]:
        """Translate nanobot SSE fragments into chat log deltas.

        A silence longer than ``STREAM_IDLE_TIMEOUT`` after the first token is
        treated as a likely tool execution. We then emit a synthetic external
        tool call, which makes the Assist pipeline start TTS immediately so
        buffered text is spoken while nanobot is busy.
        """
        voice_prompt = (self._voice_prompt or "").strip()
        inject_prompt = bool(voice_prompt) and session_id not in self._prompted_sessions
        message = f"{voice_prompt}\n\n{text}" if inject_prompt else text

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stream = client.send_message_stream(
            text=message,
            session_id=session_id,
            model=self._model,
        )
        producer = asyncio.create_task(self._async_pump_stream(stream, queue))

        first = True
        received_token = False
        tool_call: llm.ToolInput | None = None
        timeout_disabled = False

        try:
            while True:
                if tool_call is not None or timeout_disabled or first:
                    kind, value = await queue.get()
                else:
                    try:
                        kind, value = await asyncio.wait_for(
                            queue.get(), timeout=STREAM_IDLE_TIMEOUT
                        )
                    except TimeoutError:
                        if _SUPPORTS_EXTERNAL_TOOL_CALLS:
                            tool_call = _make_activity_tool_call()
                            yield {"tool_calls": [tool_call]}
                        else:
                            timeout_disabled = True
                        continue

                if kind == "error":
                    raise value
                if kind == "done":
                    if tool_call is not None:
                        yield _make_tool_result_delta(tool_call)
                    break

                token = cast(str, value)
                received_token = True

                if inject_prompt:
                    self._prompted_sessions.add(session_id)
                    inject_prompt = False

                if tool_call is not None:
                    yield _make_tool_result_delta(tool_call)
                    tool_call = None
                    yield {"role": "assistant", "content": token}
                    first = False
                elif first:
                    yield {"role": "assistant", "content": token}
                    first = False
                else:
                    yield {"content": token}
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            with contextlib.suppress(Exception):
                await stream.aclose()

        if not received_token:
            raise NanobotAPIError("nanobot returned an empty reply")
