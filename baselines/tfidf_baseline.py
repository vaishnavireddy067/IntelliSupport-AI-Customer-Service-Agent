"""Baseline 2: TF-IDF Statistical ML Support Agent Baseline.

Components:
1. Intent Classification: Sublinear TF-IDF + Logistic Regression.
2. Historical Evidence Retrieval: TF-IDF nearest-neighbor search with Cosine Similarity.
3. Response Generation: Retrieval-based template matching from top-1 historical reply.
4. Escalation Decision: Simple confidence threshold policy.
"""

import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


class TfidfIntentClassifier:
    """TF-IDF feature extraction with calibrated Logistic Regression."""

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
        X_vec = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_vec, y_train)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, texts: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(texts)
        return list(self.clf.predict(X_vec))

    def predict_intent_with_confidence(self, text: str) -> Tuple[str, float]:
        X_vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_vec)[0]
        max_idx = int(np.argmax(probs))
        return str(self.classes_[max_idx]), float(probs[max_idx])

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X_vec)


class TfidfRetriever:
    """Historical resolution retriever using sparse TF-IDF and cosine similarity."""

    def __init__(self, max_features: int = 15000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
        )
        self.corpus_matrix = None
        self.corpus_records: List[Dict[str, Any]] = []

    def fit(self, records: List[Dict[str, Any]]) -> "TfidfRetriever":
        self.corpus_records = records
        queries = [r["customer_text"] for r in records]
        self.corpus_matrix = self.vectorizer.fit_transform(queries)
        return self

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.corpus_matrix is None or len(self.corpus_records) == 0:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.corpus_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rec = self.corpus_records[idx]
            results.append({
                "conversation_id": rec.get("customer_tweet_id", str(idx)),
                "similarity": float(sims[idx]),
                "historical_query": rec.get("customer_text", ""),
                "historical_reply": rec.get("agent_text", ""),
            })
        return results


class TfidfSupportAgent:
    """Baseline 2 Agent: TF-IDF Intent Classifier + TF-IDF Retrieval + Threshold Escalation."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.classifier = TfidfIntentClassifier()
        self.retriever = TfidfRetriever()
        self.confidence_threshold = confidence_threshold

    def fit(self, train_records: List[Dict[str, Any]], labels: List[str]) -> "TfidfSupportAgent":
        queries = [r["customer_text"] for r in train_records]
        self.classifier.fit(queries, labels)
        self.retriever.fit(train_records)
        return self

    def process_message(self, message: str) -> Dict[str, Any]:
        intent, confidence = self.classifier.predict_intent_with_confidence(message)
        evidence = self.retriever.search(message, top_k=3)

        # Baseline decision policy
        if confidence < self.confidence_threshold:
            decision = "ESCALATE"
            reason = f"TF-IDF confidence ({confidence:.2f}) below threshold ({self.confidence_threshold:.2f})."
        elif not evidence or evidence[0]["similarity"] < 0.30:
            decision = "ESCALATE"
            reason = "No sufficiently similar historical resolution retrieved."
        else:
            decision = "AUTO_HANDLE"
            reason = f"Routine {intent} query with high TF-IDF confidence."

        # Template/retrieval grounded reply
        if evidence and decision == "AUTO_HANDLE":
            draft_reply = evidence[0]["historical_reply"]
        else:
            draft_reply = (
                "Thanks for reaching out to Apple Support. We'd like to take a closer look at this with you. "
                "Please send us a DM with your device details so we can assist."
            )

        return {
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
            "escalation_reason": reason,
            "draft_reply": draft_reply,
            "evidence": evidence,
        }
