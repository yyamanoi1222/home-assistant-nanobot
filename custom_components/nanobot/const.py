"""Constants for the nanobot integration."""

from logging import Logger, getLogger

DOMAIN: str = "nanobot"
DEFAULT_TIMEOUT: int = 120
DEFAULT_API_URL: str = "http://127.0.0.1:8900"

CONF_API_URL: str = "api_url"
CONF_API_KEY: str = "api_key"
CONF_TIMEOUT: str = "timeout"
CONF_MODEL: str = "model"
CONF_VOICE_PROMPT: str = "voice_prompt"

STREAM_IDLE_TIMEOUT: float = 1.5
ACTIVITY_TOOL_NAME: str = "nanobot_activity"

DEFAULT_VOICE_PROMPT: str = (
    "あなたはHome Assistantの音声アシスタントです。"
    "自然な話し言葉で、1〜2文の短い回答をしてください。"
    "Markdown記法、箇条書き、番号付きリスト、絵文字、コード、URLは使わないでください。"
    "記号や装飾を読み上げず、そのまま話せる文章にしてください。"
    "詳しい説明が必要な場合は、要点だけを簡潔に伝えて、続きを聞くよう促してください。"
    "ツール（Web検索など）を使う前には、何をするかを「Webで調べますね」のように短い一言で伝えてください。"
    "ツール名や英語の識別子はそのまま読み上げないでください。"
)

LOGGER: Logger = getLogger(__package__)
