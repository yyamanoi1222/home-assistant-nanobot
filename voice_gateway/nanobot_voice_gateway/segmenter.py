"""Sentence splitting and speech sanitising."""

from __future__ import annotations

import re

_SENTENCE_END = "。！？!?\n"

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s{0,3}\d+[.)]\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"[*_~]{1,3}")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_for_speech(text: str) -> str:
    """Return text that is safe to send to a TTS engine."""
    if not text:
        return ""
    text = _FENCE_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


class Segmenter:
    """Split a token stream into speakable sentence-sized chunks.

    The segmenter intentionally keeps the last complete sentence buffered so
    callers can mark it as final only once the model turn is known to be over.
    """

    def __init__(self, min_length: int = 8, max_length: int = 120) -> None:
        if min_length < 1:
            raise ValueError("min_length must be positive")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        self.min_length = min_length
        self.max_length = max_length
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Return unflushed text."""
        return self._buffer

    def feed(self, text: str) -> list[str]:
        """Add text and return any completed segments."""
        if text:
            self._buffer += text
        return self._extract()

    def flush(self) -> list[str]:
        """Return all remaining text, including a short tail."""
        segments = self._extract()
        tail = self._buffer.strip()
        if tail:
            segments.append(tail)
            self._buffer = ""
        return segments

    def _extract(self) -> list[str]:
        segments: list[str] = []
        while True:
            index = self._find_boundary(self._buffer)
            if index is None:
                break

            candidate = self._buffer[:index].strip()
            while len(candidate) < self.min_length:
                next_index = self._find_boundary(self._buffer, index)
                if next_index is None:
                    index = None
                    break
                index = next_index
                candidate = self._buffer[:index].strip()

            if index is None:
                break

            segments.append(candidate)
            self._buffer = self._buffer[index:].lstrip()

        segments.extend(self._split_long(self._buffer))
        return segments

    def _split_long(self, text: str) -> list[str]:
        segments: list[str] = []
        while len(text) >= self.max_length:
            cut = self._soft_break(text, self.max_length)
            chunk = text[:cut].strip()
            if chunk:
                segments.append(chunk)
            text = text[cut:].lstrip()
        self._buffer = text
        return segments

    def _soft_break(self, text: str, limit: int) -> int:
        floor = max(1, limit // 2)
        for index in range(min(limit, len(text)) - 1, floor - 1, -1):
            if text[index] in " 、,;":
                return index + 1
        return limit

    @staticmethod
    def _find_boundary(text: str, start: int = 0) -> int | None:
        for index in range(start, len(text)):
            char = text[index]
            if char in _SENTENCE_END:
                return index + 1
            if char == ".":
                following = text[index + 1] if index + 1 < len(text) else ""
                previous = text[index - 1] if index > 0 else ""
                if (not following or following.isspace()) and not previous.isdigit():
                    return index + 1
        return None
