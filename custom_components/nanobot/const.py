"""Constants for the nanobot integration."""

from logging import Logger, getLogger

DOMAIN: str = "nanobot"
DEFAULT_TIMEOUT: int = 120
DEFAULT_API_URL: str = "http://127.0.0.1:8900"

CONF_API_URL: str = "api_url"
CONF_API_KEY: str = "api_key"
CONF_TIMEOUT: str = "timeout"
CONF_MODEL: str = "model"

LOGGER: Logger = getLogger(__package__)
