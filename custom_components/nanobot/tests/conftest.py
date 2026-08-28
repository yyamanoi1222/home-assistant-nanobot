"""Shared test fixtures for nanobot integration."""

import pytest


@pytest.fixture(autouse=True)
def mock_aiohttp_imports() -> None:
    """Ensure aiohttp imports are available for mocked tests."""
    pass
