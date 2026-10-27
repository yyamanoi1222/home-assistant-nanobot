"""Gateway configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return float(value) if value is not None else default


@dataclass(slots=True)
class GatewayConfig:
    """Runtime configuration for the voice gateway."""

    host: str = "127.0.0.1"
    port: int = 8900
    api_key: str | None = None
    nanobot_config: str | None = None
    workspace: str | None = None
    model_preset: str | None = None
    max_concurrent_turns: int = 4
    cancel_timeout: float = 5.0
    heartbeat_interval: float = 15.0
    turn_ttl: float = 300.0
    snapshot_enabled: bool = True

    @classmethod
    def from_env(cls) -> GatewayConfig:
        """Build a configuration from environment variables."""
        return cls(
            host=_env_str("NANOBOT_VOICE_HOST", "127.0.0.1") or "127.0.0.1",
            port=_env_int("NANOBOT_VOICE_PORT", 8900),
            api_key=_env_str("NANOBOT_VOICE_API_KEY"),
            nanobot_config=_env_str("NANOBOT_CONFIG"),
            workspace=_env_str("NANOBOT_VOICE_WORKSPACE"),
            model_preset=_env_str("NANOBOT_VOICE_MODEL_PRESET"),
            max_concurrent_turns=_env_int("NANOBOT_VOICE_MAX_TURNS", 4),
            cancel_timeout=_env_float("NANOBOT_VOICE_CANCEL_TIMEOUT", 5.0),
            heartbeat_interval=_env_float("NANOBOT_VOICE_HEARTBEAT", 15.0),
            turn_ttl=_env_float("NANOBOT_VOICE_TURN_TTL", 300.0),
        )
