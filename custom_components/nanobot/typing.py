"""Shared typing for the nanobot integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NanobotRuntimeData:
    """Runtime data for a nanobot config entry."""

    config: dict
    title: str
