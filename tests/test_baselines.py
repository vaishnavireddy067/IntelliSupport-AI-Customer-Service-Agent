"""Unit tests for Baseline 1 and Baseline 2 support models."""

import pytest
from baselines.majority import MajorityIntentClassifier, MajoritySupportAgent
from baselines.tfidf_baseline import TfidfIntentClassifier, TfidfRetriever, TfidfSupportAgent


def test_majority_baseline_predicts_frequent_class():
    X_train = ["screen broken", "battery dead", "battery draining", "battery dead"]
    y_train = ["screen_display", "battery_power", "battery_power", "battery_power"]

    clf = MajorityIntentClassifier().fit(X_train, y_train)
    preds = clf.predict(["new message", "another message"])

    assert preds == ["battery_power", "battery_power"]
    intent, conf = clf.predict_with_confidence("sample query")
    assert intent == "battery_power"
    assert conf == 0.75


def test_majority_agent_structure():
    agent = MajoritySupportAgent()
    agent.fit(["hello"], ["other_general"])
    res = agent.process_message("How are you?")

    assert "intent" in res
    assert "confidence" in res
    assert "decision" in res
    assert "draft_reply" in res
    assert res["decision"] == "AUTO_HANDLE"


def test_tfidf_baseline_classification_and_retrieval():
    records = [
        {"customer_tweet_id": "1", "customer_text": "iPhone battery dies fast", "agent_text": "Check battery health in settings."},
        {"customer_tweet_id": "2", "customer_text": "Screen cracked on iPhone 8", "agent_text": "You can visit Apple Store for repair."},
        {"customer_tweet_id": "3", "customer_text": "Apple ID password reset", "agent_text": "Go to iforgot.apple.com to reset."},
    ]
    labels = ["battery_power", "screen_display", "apple_id_account"]

    agent = TfidfSupportAgent(confidence_threshold=0.50)
    agent.fit(records, labels)

    res = agent.process_message("My iPhone battery is dying really fast")
    assert res["intent"] == "battery_power"
    assert res["confidence"] > 0.3
    assert len(res["evidence"]) > 0
    assert "Check battery health" in res["evidence"][0]["historical_reply"]
