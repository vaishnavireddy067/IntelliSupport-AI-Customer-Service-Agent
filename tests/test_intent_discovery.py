"""Unit tests for intent discovery and taxonomy definition."""

import os
from src.intent_discovery import discover_top_ngrams, discover_tfidf_keywords, load_intent_taxonomy


def test_discover_top_ngrams():
    texts = [
        "battery draining fast on iPhone",
        "battery health dropped drastically",
        "screen cracked on iPhone",
    ]
    res = discover_top_ngrams(texts, top_n=5)
    unigram_dict = dict(res["unigrams"])
    assert "battery" in unigram_dict
    assert unigram_dict["battery"] == 2


def test_load_intent_taxonomy_schema():
    taxonomy = load_intent_taxonomy("configs/intents.yaml")
    assert "intents" in taxonomy
    assert len(taxonomy["intents"]) == 10

    for intent in taxonomy["intents"]:
        assert "intent_id" in intent
        assert "intent_name" in intent
        assert "description" in intent
        assert "positive_examples" in intent
        assert len(intent["positive_examples"]) >= 2
        assert "confusing_intents" in intent
        assert "escalation_considerations" in intent
        assert "default_action" in intent["escalation_considerations"]
