"""Baseline 1: Trivial Support Agent Baseline.

Components:
1. Intent Classification: Majority-class classifier (always predicts most common training intent).
2. Response Generation: Generic brand fallback response.
3. Escalation Decision: Simple default policy (default AUTO_HANDLE, or conservative escalation).
"""

from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class MajorityIntentClassifier:
    """Trivial baseline predicting the single most frequent training class."""

    def __init__(self):
        self.majority_class: str = "other_general"
        self.class_frequencies: Dict[str, float] = {}
        self.total_samples: int = 0

    def fit(self, X_train: List[str], y_train: List[str]) -> "MajorityIntentClassifier":
        """Fit prior probability distribution on training labels."""
        counts = Counter(y_train)
        self.total_samples = len(y_train)
        self.majority_class = counts.most_common(1)[0][0]
        self.class_frequencies = {k: v / self.total_samples for k, v in counts.items()}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict majority class for all input messages."""
        return [self.majority_class] * len(texts)

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict majority class and its prior training frequency."""
        return self.majority_class, float(self.class_frequencies.get(self.majority_class, 0.0))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return static prior probability distribution for all inputs."""
        all_classes = sorted(list(self.class_frequencies.keys()))
        probs = np.array([self.class_frequencies.get(c, 0.0) for c in all_classes])
        return np.tile(probs, (len(texts), 1))


class MajoritySupportAgent:
    """Trivial End-to-End Agent Baseline combining majority classifier and generic fallback."""

    def __init__(self, majority_class: str = "other_general"):
        self.classifier = MajorityIntentClassifier()
        self.fallback_reply = (
            "Thanks for reaching out to Apple Support. We'd like to help. "
            "Could you please DM us with your device model and the exact iOS version you are running?"
        )

    def fit(self, X_train: List[str], y_train: List[str]) -> "MajoritySupportAgent":
        self.classifier.fit(X_train, y_train)
        return self

    def process_message(self, message: str) -> Dict[str, Any]:
        intent, conf = self.classifier.predict_with_confidence(message)
        # Trivial escalation policy: always AUTO_HANDLE or default conservative
        decision = "AUTO_HANDLE"
        reason = "Majority baseline default handling."
        return {
            "intent": intent,
            "confidence": conf,
            "decision": decision,
            "escalation_reason": reason,
            "draft_reply": self.fallback_reply,
            "evidence": [],
        }
