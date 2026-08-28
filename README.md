# nanobot for Home Assistant

[nanobot](https://github.com/HKUDS/nanobot) integration for Home Assistant.

This custom component exposes nanobot's OpenAI-compatible chat API as a [Home Assistant conversation agent](https://www.home-assistant.io/integrations/conversation/). You can use it from the Assist voice pipeline or the `conversation.process` service.

## Features

- Text-only chat via nanobot's `/v1/chat/completions` endpoint
- Conversation history handled by nanobot using `session_id`
- Optional model selection
- Optional API key for LAN / remote nanobot instances
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
3. Enter your nanobot API URL (default: `http://127.0.0.1:8900`).
4. Fill in the API key only if nanobot is bound to `0.0.0.0` or a remote interface.
5. Optionally enter a model name returned by `GET /v1/models`.
6. Adjust the request timeout if needed.

## Usage

After setup, select **nanobot** as the conversation agent for an Assist pipeline or call:

```yaml
service: conversation.process
 data:
  agent_id: <nanobot agent id>
  text: "Hello!"
```

## Requirements

- Home Assistant 2024.12.0 or newer
- A running [nanobot](https://github.com/HKUDS/nanobot) instance with the OpenAI-compatible API enabled

## Notes

- nanobot accepts exactly one user message per request. This integration always sends only the latest utterance and uses the conversation `session_id` for continuity.
- Device control through Home Assistant is not supported because nanobot's OpenAI-compatible endpoint does not expose tool/function calling to external clients.
