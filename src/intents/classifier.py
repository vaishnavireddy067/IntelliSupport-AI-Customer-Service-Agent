"""Intent classification models: TF-IDF baseline and Sentence-Transformer classifier.

Provides unified prediction interfaces:
- predict_intent(text) -> str
- predict_intent_with_confidence(text) -> (str, float)
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Ensure offline loading from local cache for sub-second latency
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class TfidfIntentClassifier:
    """Baseline 2: TF-IDF feature extraction + Logistic Regression."""

    def __init__(self, max_features: int = 10000, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_train: List[str], y_train: List[str]) -> "TfidfIntentClassifier":
        """Fit vectorizer and logistic regression model."""
        X_vec = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_vec, y_train)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict intent for a list of customer messages."""
        X_vec = self.vectorizer.transform(texts)
        return list(self.clf.predict(X_vec))

    def predict_intent(self, text: str) -> str:
        """Predict intent for a single message."""
        return self.predict([text])[0]

    def predict_intent_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict intent and associated maximum softmax probability."""
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        max_idx = int(np.argmax(probs))
        predicted_class = str(self.classes_[max_idx])
        confidence = float(probs[max_idx])
        return predicted_class, confidence

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probability distributions across all classes."""
        X_vec = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X_vec)

    def save(self, file_path: str):
        """Serialize model to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf, "classes_": self.classes_}, f)

    @classmethod
    def load(cls, file_path: str) -> "TfidfIntentClassifier":
        """Load serialized model from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        obj.classes_ = data["classes_"]
        return obj


class EmbeddingIntentClassifier:
    """Final Intent Classifier: Sentence-Transformer embeddings + Logistic Regression."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.random_state = random_state
        self._encoder = None
        self.clf = LogisticRegression(
            C=1.5,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        self.classes_: Optional[np.ndarray] = None

    @property
    def encoder(self):
        """Lazy load SentenceTransformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("Failed to load SentenceTransformer: %s. Using TF-IDF fallback.", e)
                self._encoder = None
        return self._encoder

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts into embedding vectors."""
        if self.encoder is not None:
            return self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        # Fallback if sentence-transformers unavailable
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(n_features=384)
        return hv.transform(texts).toarray()

    def fit(self, X_train: List[str], y_train: List[str]) -> "EmbeddingIntentClassifier":
        """Fit classifier on sentence embeddings."""
        logger.info("Computing embeddings for %d training samples...", len(X_train))
        embeddings = self._encode(X_train)
        self.clf.fit(embeddings, y_train)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict intent for a list of customer messages."""
        embeddings = self._encode(texts)
        return list(self.clf.predict(embeddings))

    def predict_intent(self, text: str) -> str:
        """Predict intent for a single message."""
        return self.predict([text])[0]

    def predict_intent_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict intent and softmax probability confidence."""
        embeddings = self._encode([text])
        probs = self.clf.predict_proba(embeddings)[0]
        max_idx = int(np.argmax(probs))
        predicted_class = str(self.classes_[max_idx])
        confidence = float(probs[max_idx])
        return predicted_class, confidence

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probability distributions across all classes."""
        embeddings = self._encode(texts)
        return self.clf.predict_proba(embeddings)

    def save(self, file_path: str):
        """Serialize model classifier (encoder reloaded by model_name)."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "random_state": self.random_state,
                "clf": self.clf,
                "classes_": self.classes_,
            }, f)

    @classmethod
    def load(cls, file_path: str) -> "EmbeddingIntentClassifier":
        """Load serialized model from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        obj = cls(model_name=data["model_name"], random_state=data["random_state"])
        obj.clf = data["clf"]
        obj.classes_ = data["classes_"]
        return obj
