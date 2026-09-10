"""Baseline 1: Majority Class Intent Classifier.

Mandatory baseline predicting the most frequent training intent for every sample.
"""

from collections import Counter
from typing import List, Tuple, Dict, Any
import numpy as np


class MajorityIntentClassifier:
    """Trivial baseline that always predicts the most frequent class in training data."""

    def __init__(self):
        self.majority_class: str = "other_general"
        self.class_frequencies: Dict[str, float] = {}
        self.total_samples: int = 0

    def fit(self, y_train: List[str]) -> "MajorityIntentClassifier":
        """Fit on training class labels."""
        counts = Counter(y_train)
        self.total_samples = len(y_train)
        self.majority_class = counts.most_common(1)[0][0]
        self.class_frequencies = {k: v / self.total_samples for k, v in counts.items()}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict the majority class for every input sample."""
        return [self.majority_class] * len(texts)

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Return majority class and its prior probability in training data."""
        confidence = self.class_frequencies.get(self.majority_class, 0.0)
        return self.majority_class, float(confidence)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability matrix representing class prior probabilities."""
        all_classes = sorted(list(self.class_frequencies.keys()))
        probs = np.array([self.class_frequencies.get(c, 0.0) for c in all_classes])
        return np.tile(probs, (len(texts), 1))
