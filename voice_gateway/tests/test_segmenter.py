"""Tests for sentence splitting and sanitising."""

from nanobot_voice_gateway.segmenter import Segmenter, sanitize_for_speech


def test_splits_japanese_sentences() -> None:
    segmenter = Segmenter(min_length=4)
    assert segmenter.feed("こんにちは。元気ですか？") == [
        "こんにちは。",
        "元気ですか？",
    ]


def test_merges_segments_below_min_length() -> None:
    segmenter = Segmenter(min_length=20)
    assert segmenter.feed("はい。そうです。") == []
    assert segmenter.flush() == ["はい。そうです。"]


def test_flush_emits_short_tail() -> None:
    segmenter = Segmenter(min_length=4)
    assert segmenter.feed("途中まで") == []
    assert segmenter.flush() == ["途中まで"]


def test_force_splits_long_text() -> None:
    segmenter = Segmenter(min_length=1, max_length=10)
    chunks = segmenter.feed("あ" * 25)
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) + segmenter.pending == "あ" * 25
    assert segmenter.flush() == ["あ" * 5]


def test_sanitize_removes_markdown_code_and_urls() -> None:
    raw = (
        "## 見出し\n"
        "- [リンク](https://example.com) です\n"
        "```python\nprint(1)\n```\n"
        "完了"
    )
    clean = sanitize_for_speech(raw)
    assert "https://" not in clean
    assert "```" not in clean
    assert "print" not in clean
    assert "見出し" in clean
    assert "リンク" in clean
    assert "完了" in clean


def test_sanitize_keeps_inline_code_content() -> None:
    assert sanitize_for_speech("`nanobot` を使う") == "nanobot を使う"
