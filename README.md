# nanobot for Home Assistant

[nanobot](https://github.com/HKUDS/nanobot) integration for Home Assistant.

This custom component connects the nanobot voice gateway to a [Home Assistant conversation agent](https://www.home-assistant.io/integrations/conversation/). You can use it from the Assist voice pipeline or the `conversation.process` service.

The integration requires the bundled [voice gateway](voice_gateway/README.md). It does not call nanobot's OpenAI-compatible endpoint directly, because that protocol cannot mark which text is safe to speak or when a turn was cancelled.

## Features

- Turn-based voice conversation through the nanobot voice gateway
- Sentence-sized segments from the voice gateway, so Assist can start speaking early
- Tool events and an optional spoken filler while a tool is running
- Cancellation-aware streaming for Assist barge-in
- Conversation history handled by nanobot using `session_id`
- Optional model selection
- Optional API key for LAN / remote gateway instances
- Config Flow setup from the Home Assistant UI

## Installation

### HACS

1. Open **HACS** → **Integrations**.
2. Click the menu and select **Custom repositories**.
3. Add `https://github.com/yyamanoi1222/home-assistant-nanobot` as type **Integration**.
4. Install **nanobot** from the HACS store.
5. Restart Home Assistant.

### Manual

1. Copy the `custom_components/nanobot` directory to your Home Assistant `custom_components/` directory.
2. Restart Home Assistant.

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**.
2. Search for **nanobot**.
3. Enter your voice gateway URL (default: `http://127.0.0.1:8900`).
4. Fill in the API key only if the gateway is bound to `0.0.0.0` or a remote interface.
5. Optionally enter a model name.
6. Adjust the response start timeout if needed. It bounds the initial connection and the wait for the first segment; long tool calls are not cut off by the HA-side timeout.
7. Optionally edit the voice response prompt, which keeps spoken replies short and free of formatting. Leave it empty to disable.
8. Optionally set a tool-call filler. It is spoken once if a tool starts before any reply text. Leave it empty to disable.

## Usage

After setup, select **nanobot** as the conversation agent for an Assist pipeline or call:

```yaml
service: conversation.process
 data:
  agent_id: <nanobot agent id>
  text: "Hello!"
```

## Requirements

- Home Assistant 2025.7.0 or newer
- A running [nanobot voice gateway](voice_gateway/README.md) connected to a nanobot instance

Start the gateway with:

```bash
python -m pip install "nanobot-voice-gateway @ git+https://github.com/yyamanoi1222/home-assistant-nanobot#subdirectory=voice_gateway"
nanobot-voice-gateway --port 8900
```

## Notes

- The integration always sends only the latest utterance and uses the conversation `session_id` for continuity.
- Device control is available through nanobot tools executed by the gateway. Tool side effects cannot be rolled back when a turn is cancelled.
- Streaming TTS requires a TTS engine that implements `async_stream_tts_audio` (for example, Microsoft Edge TTS). Assist only starts streaming after roughly 60 characters, so short replies still wait for the full answer.
- If Assist cancels a turn, the integration asks the gateway to stop the run and roll back that turn's session history.
- The voice response prompt is prepended to the first request of each HA session and is therefore stored in nanobot's session history. Other nanobot channels such as Discord are unaffected.
- If the gateway is unreachable or does not support the voice protocol, requests fail. There is intentionally no fallback to the OpenAI-compatible endpoint.
