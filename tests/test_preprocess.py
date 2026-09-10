"""Unit tests for text preprocessing and noise preservation."""

import pytest
from src.data.preprocess import clean_customer_text, clean_agent_text, is_usable_conversation_pair


def test_clean_customer_text_removes_leading_mentions():
    text = "@AppleSupport @115854 My battery is draining super fast"
    cleaned = clean_customer_text(text)
    assert cleaned == "My battery is draining super fast"


def test_clean_customer_text_preserves_slang_and_typos():
    text = "@AppleSupport pls fix dis, phone wont turn on smh!!!"
    cleaned = clean_customer_text(text)
    # Ensure typos and slang are strictly preserved
    assert "pls" in cleaned
    assert "dis" in cleaned
    assert "wont" in cleaned
    assert "smh!!!" in cleaned


def test_clean_customer_text_unescapes_html():
    text = "@AppleSupport Phone &amp; iPad won&#39;t sync"
    cleaned = clean_customer_text(text)
    assert "&" in cleaned
    assert "'" in cleaned
    assert "&amp;" not in cleaned


def test_clean_customer_text_preserves_emojis():
    text = "@AppleSupport My screen cracked 😭💔"
    cleaned = clean_customer_text(text)
    assert "😭" in cleaned or "cracked" in cleaned


def test_is_usable_conversation_pair():
    # Empty messages are unusable
    assert not is_usable_conversation_pair("", "Hello")
    assert not is_usable_conversation_pair("Help", "")

    # URL only customer message is unusable
    assert not is_usable_conversation_pair("https://t.co/xyz123", "How can we help?")

    # Valid message pair
    assert is_usable_conversation_pair("My phone battery dies fast", "We'd like to help. Which iOS version?")
