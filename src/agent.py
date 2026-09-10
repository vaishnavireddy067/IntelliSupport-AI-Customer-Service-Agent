"""Complete end-to-end AppleSupport AI Agent pipeline.

Orchestrates:
1. Message preprocessing
2. Intent classification & confidence scoring
3. Historical resolution retrieval (Top-K)
4. Grounded draft response generation
5. Rule-based escalation evaluation
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import logging
from typing import Dict, Any, Optional

from src.data.preprocess import clean_customer_text
from src.intents.classifier import EmbeddingIntentClassifier, TfidfIntentClassifier
from src.retrieval.index import VectorIndex
from src.retrieval.retrieve import HistoricalRetriever
from src.generation.reply_generator import GroundedReplyGenerator
from src.escalation.policy import EscalationPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AppleSupportAgent:
    """Production-grade AI Customer Support Agent for AppleSupport."""

    def __init__(
        self,
        classifier=None,
        retriever=None,
        reply_generator=None,
        escalation_policy=None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.reply_generator = reply_generator or GroundedReplyGenerator()
        self.escalation_policy = escalation_policy or EscalationPolicy()

    @classmethod
    def load_from_artifacts(
        cls,
        model_dir: str = "data/processed",
        index_dir: str = "data/processed/retrieval_index",
    ) -> "AppleSupportAgent":
        """Load fully trained agent components from disk artifacts."""
        # Load classifier
        embedding_model_path = os.path.join(model_dir, "embedding_classifier.pkl")
        tfidf_model_path = os.path.join(model_dir, "tfidf_classifier.pkl")

        if os.path.exists(embedding_model_path):
            logger.info("Loading EmbeddingIntentClassifier from %s", embedding_model_path)
            classifier = EmbeddingIntentClassifier.load(embedding_model_path)
        elif os.path.exists(tfidf_model_path):
            logger.info("Loading TfidfIntentClassifier from %s", tfidf_model_path)
            classifier = TfidfIntentClassifier.load(tfidf_model_path)
        else:
            logger.warning("No pre-trained classifier found. Using fallback classifier.")
            classifier = None

        # Load retrieval index
        if os.path.exists(index_dir):
            logger.info("Loading VectorIndex from %s", index_dir)
            index = VectorIndex.load(index_dir, use_faiss=True)
            retriever = HistoricalRetriever(index=index)
        else:
            logger.warning("No retrieval index found at %s.", index_dir)
            retriever = None

        if classifier is not None and retriever is not None:
            if hasattr(retriever.index, "encoder") and hasattr(classifier, "_encoder"):
                classifier._encoder = retriever.index.encoder

        return cls(classifier=classifier, retriever=retriever)

    def process_message(self, raw_message: str) -> Dict[str, Any]:
        """Execute the end-to-end support pipeline on a customer query."""
        import time
        import numpy as np

        t0 = time.perf_counter()

        # 1. Preprocess query
        cleaned_text = clean_customer_text(raw_message)
        if not cleaned_text:
            cleaned_text = raw_message.strip()

        # 2. Intent classification & confidence
        top_intents = []
        if self.classifier is not None:
            intent, confidence = self.classifier.predict_intent_with_confidence(cleaned_text)
            if hasattr(self.classifier, "predict_proba"):
                try:
                    probs = self.classifier.predict_proba([cleaned_text])[0]
                    classes = self.classifier.classes_
                    sorted_indices = np.argsort(probs)[::-1]
                    top_intents = [
                        {"intent": str(classes[i]), "confidence": round(float(probs[i]), 4)}
                        for i in sorted_indices[:3]
                    ]
                except Exception:
                    top_intents = [{"intent": intent, "confidence": round(float(confidence), 4)}]
            else:
                top_intents = [{"intent": intent, "confidence": round(float(confidence), 4)}]
        else:
            intent, confidence = "other_general", 0.35
            top_intents = [{"intent": intent, "confidence": 0.35}]

        # 3. Historical retrieval (Top-3)
        retrieved_examples = []
        if self.retriever is not None:
            retrieved_examples = self.retriever.retrieve(cleaned_text, top_k=3)

        # 4. Grounded reply generation
        gen_result = self.reply_generator.generate_reply(
            customer_message=cleaned_text,
            predicted_intent=intent,
            retrieved_examples=retrieved_examples,
        )
        draft_reply = gen_result.get("draft_reply", "")

        # 5. Escalation policy
        escalation = self.escalation_policy.evaluate(
            customer_message=cleaned_text,
            predicted_intent=intent,
            intent_confidence=confidence,
            retrieved_examples=retrieved_examples,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Format evidence strictly matching prompt specification
        evidence_list = []
        for ex in retrieved_examples:
            evidence_list.append({
                "conversation_id": str(ex.get("customer_tweet_id", ex.get("tweet_id", ""))),
                "similarity": round(float(ex.get("similarity_score", 0.0)), 4),
                "historical_query": ex.get("customer_text", ""),
                "historical_reply": ex.get("agent_text", ""),
            })

        decision_str = "ESCALATE" if "ESCALATE" in escalation["decision"] else "AUTO_HANDLE"
        reason_str = escalation.get("reason", "")

        return {
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "top_intents": top_intents,
            "draft_reply": draft_reply,
            "decision": decision_str,
            "escalation_reason": reason_str,
            "evidence": evidence_list,
            "latency_ms": latency_ms,
            # Additional debug context
            "message": raw_message,
            "cleaned_message": cleaned_text,
            "retrieved_examples": retrieved_examples,
            "reason": reason_str,
        }


def main():
    """CLI test interface for interactive testing."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "@AppleSupport my iPhone 8 battery drops from 80% to 15% in one hour after iOS 11 update"

    agent = AppleSupportAgent.load_from_artifacts()
    result = agent.process_message(query)
    
    # Print clean formatted view
    output_view = {
        "intent": result["intent"],
        "confidence": result["confidence"],
        "draft_reply": result["draft_reply"],
        "decision": result["decision"],
        "escalation_reason": result["escalation_reason"],
        "evidence": result["evidence"],
    }
    print(json.dumps(output_view, indent=2))


if __name__ == "__main__":
    main()
