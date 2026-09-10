"""Unit tests for the escalation policy decision rules."""

import pytest
from src.escalation.policy import EscalationPolicy


def test_escalation_triggers_on_stolen_phone():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="My iPhone was stolen at the train station, help police report!",
        predicted_intent="store_hardware_service",
        intent_confidence=0.95,
        retrieved_examples=[{"similarity_score": 0.85}],
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "stolen" in res["reason"]


def test_escalation_triggers_on_repeated_customer_frustration():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="I tried everything already and no one is replying, this is unacceptable",
        predicted_intent="battery_power",
        intent_confidence=0.88,
        retrieved_examples=[{"similarity_score": 0.75}],
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "frustration" in res["reason"] or "tried everything" in res["reason"]


def test_escalation_triggers_on_low_confidence():
    policy = EscalationPolicy(min_intent_confidence=0.40)
    res = policy.evaluate(
        customer_message="something weird is happening",
        predicted_intent="other_general",
        intent_confidence=0.22,
        retrieved_examples=[{"similarity_score": 0.60}],
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "confidence" in res["reason"].lower()


def test_escalation_triggers_on_low_retrieval_similarity():
    policy = EscalationPolicy(min_retrieval_similarity=0.50)
    res = policy.evaluate(
        customer_message="Unusual custom query with no historical match",
        predicted_intent="battery_power",
        intent_confidence=0.85,
        retrieved_examples=[{"similarity_score": 0.28}],
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "retrieval similarity" in res["reason"].lower()


def test_auto_handle_on_routine_high_confidence_query():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="How do I check my battery health on iPhone 7?",
        predicted_intent="battery_power",
        intent_confidence=0.92,
        retrieved_examples=[{"similarity_score": 0.88}],
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert "routine self-service" in res["reason"]
