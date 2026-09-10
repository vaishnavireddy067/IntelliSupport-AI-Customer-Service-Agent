"""Intent Classification Module.

Provides semantic intent classification using Sentence-Transformer embeddings
(all-MiniLM-L6-v2) mapped to an empirical 10-class support taxonomy.
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.intents.classifier import EmbeddingIntentClassifier, TfidfIntentClassifier
from src.intents.taxonomy import INTENT_NAMES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "data/processed/embedding_classifier.pkl"


def get_default_classifier(model_path: str = DEFAULT_MODEL_PATH) -> EmbeddingIntentClassifier:
    """Load pre-trained embedding classifier from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained intent classifier not found at '{model_path}'. "
            "Please run 'python scripts/run_evaluation.py' to train and save the model."
        )
    return EmbeddingIntentClassifier.load(model_path)


class IntentClassifier:
    """High-level semantic intent classifier interface for the AI Agent."""

    def __init__(
        self,
        classifier: Optional[EmbeddingIntentClassifier] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.75,
    ):
        self.confidence_threshold = confidence_threshold
        if classifier is not None:
            self.model = classifier
        elif os.path.exists(model_path):
            self.model = get_default_classifier(model_path)
        else:
            self.model = EmbeddingIntentClassifier()

    def predict(self, customer_message: str) -> str:
        """Predict top intent label for a single message."""
        return self.model.predict_intent(customer_message)

    def predict_with_confidence(self, customer_message: str) -> Tuple[str, float]:
        """Predict intent and softmax confidence score."""
        return self.model.predict_intent_with_confidence(customer_message)

    def is_confident(self, confidence: float) -> bool:
        """Check if confidence meets the operational decision threshold."""
        return confidence >= self.confidence_threshold


__all__ = [
    "IntentClassifier",
    "EmbeddingIntentClassifier",
    "TfidfIntentClassifier",
    "get_default_classifier",
    "INTENT_NAMES",
]
