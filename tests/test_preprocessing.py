"""Unit tests for root preprocessing module."""

from src.preprocessing import clean_customer_text, clean_agent_text, is_usable_conversation_pair


def test_clean_customer_text_mentions_and_entities():
    raw = "@AppleSupport @115854 iPhone 7 &amp; AirPods won&#39;t connect"
    res = clean_customer_text(raw)
    assert res == "iPhone 7 & AirPods won't connect"


def test_preserves_slang_and_typos():
    raw = "@AppleSupport yo pls halp my battery ded af"
    res = clean_customer_text(raw)
    assert "yo pls halp my battery ded af" == res


def test_usable_pair_filtering():
    assert is_usable_conversation_pair("battery dies", "try restarting")
    assert not is_usable_conversation_pair("https://t.co/xyz", "how can we help?")
    assert not is_usable_conversation_pair("", "hello")
