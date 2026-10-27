# nanobot for Home Assistant

[nanobot](https://github.com/HKUDS/nanobot) integration for Home Assistant.

This custom component connects the nanobot voice gateway to a [Home Assistant conversation agent](https://www.home-assistant.io/integrations/conversation/). You can use it from the Assist voice pipeline or the `conversation.process` service.

The integration requires the bundled [voice gateway](voice_gateway/README.md). It does not call nanobot's OpenAI-compatible endpoint directly, because that protocol cannot mark which text is safe to speak or when a turn was cancelled.

## Features

- Turn-based voice conversation through the nanobot voice gateway
- Sentence-sized segments from the voice gateway, so Assist can start speaking early
- Tool events, with the model asked to announce tool use before it runs
- Optional fallback filler for models that do not announce tool use
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
7. Optionally edit the voice response prompt, which keeps spoken replies short and free of formatting. The integration always adds a fixed internal prompt that asks the model to announce tool use first.
8. Optionally set a fallback tool-call filler. It is only spoken if a tool starts before the model has produced any text. Leave it empty to rely on the model's announcement. Edge TTS needs at least 20 characters for the first streamed chunk, so keep that in mind if you use a fallback.

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
- Streaming TTS requires a TTS engine that implements `async_stream_tts_audio` (for example, Microsoft Edge TTS). Assist starts streaming after roughly 60 characters or after a tool call, so short replies without tools still wait for the full answer.
- Edge TTS buffers the first streamed chunk until it has at least 20 characters. A fallback filler shorter than that is merged into the final answer. The internal prompt asks the model to announce tool use in a sentence of 20 or more characters.
- If Assist cancels a turn, the integration asks the gateway to stop the run and roll back that turn's session history.
- A fixed internal prompt is always prepended to the first request of each HA session, followed by the optional voice response prompt. This is stored in nanobot's session history. Other nanobot channels such as Discord are unaffected.
- If the gateway is unreachable or does not support the voice protocol, requests fail. There is intentionally no fallback to the OpenAI-compatible endpoint.
