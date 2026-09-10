"""Unit tests for intent classification and baseline models."""

import pytest
import os
import tempfile
from src.intents.baseline import MajorityIntentClassifier
from src.intents.classifier import TfidfIntentClassifier


def test_majority_classifier():
    clf = MajorityIntentClassifier()
    train_y = ["battery_power", "battery_power", "screen_display", "battery_power"]
    clf.fit(train_y)

    assert clf.majority_class == "battery_power"
    preds = clf.predict(["My screen broke", "Random text"])
    assert preds == ["battery_power", "battery_power"]

    pred, conf = clf.predict_with_confidence("Query")
    assert pred == "battery_power"
    assert conf == 0.75


def test_tfidf_classifier():
    X_train = [
        "battery draining fast dying quickly",
        "battery not charging at all",
        "cracked screen touch not responding display",
        "screen black display broken",
        "apple id locked password forgotten",
        "forgot icloud password account locked",
    ]
    y_train = [
        "battery_power",
        "battery_power",
        "screen_display",
        "screen_display",
        "apple_id_account",
        "apple_id_account",
    ]

    clf = TfidfIntentClassifier(max_features=500, random_state=42)
    clf.fit(X_train, y_train)

    # Test prediction
    intent, conf = clf.predict_intent_with_confidence("battery dying so fast")
    assert intent == "battery_power"
    assert 0.0 <= conf <= 1.0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_tfidf.pkl")
        clf.save(save_path)
        loaded_clf = TfidfIntentClassifier.load(save_path)
        loaded_intent, loaded_conf = loaded_clf.predict_intent_with_confidence("battery dying so fast")
        assert loaded_intent == intent
        assert round(loaded_conf, 4) == round(conf, 4)
