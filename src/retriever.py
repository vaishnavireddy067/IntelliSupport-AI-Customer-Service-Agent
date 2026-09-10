"""Historical Evidence Retrieval Module.

Grounds agent replies in historically verified customer-support conversations from
the target brand (AppleSupport).
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.retrieval.index import VectorIndex
from src.retrieval.retrieve import HistoricalRetriever

logger = logging.getLogger(__name__)


def get_default_retriever(
    index_path: str = "data/processed/retrieval_index",
    model_name: str = "all-MiniLM-L6-v2",
) -> HistoricalRetriever:
    """Load pre-built VectorIndex from disk and return ready-to-use HistoricalRetriever."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Retrieval index not found at '{index_path}'. "
            "Please run 'python scripts/build_index.py' first."
        )
    index = VectorIndex.load(index_path)
    return HistoricalRetriever(index=index, model_name=model_name)


class EvidenceRetriever:
    """Unified wrapper around HistoricalRetriever providing standardized structured evidence."""

    def __init__(
        self,
        retriever: Optional[HistoricalRetriever] = None,
        index_path: str = "data/processed/retrieval_index",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = get_default_retriever(index_path=index_path, model_name=model_name)

    def retrieve_evidence(self, customer_message: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k historical examples formatted for prompt grounding and evidence logging.

        Returns list of structured records:
        - conversation_id: original parent tweet / conversation id
        - similarity: cosine similarity score [0.0, 1.0]
        - historical_query: past customer tweet
        - historical_reply: verified AppleSupport resolution
        """
        raw_results = self.retriever.retrieve(customer_message, top_k=top_k)
        formatted_evidence = []

        for r in raw_results:
            formatted_evidence.append({
                "conversation_id": str(r.get("customer_tweet_id", r.get("tweet_id", ""))),
                "similarity": round(float(r.get("similarity_score", 0.0)), 4),
                "historical_query": r.get("customer_text", ""),
                "historical_reply": r.get("agent_text", ""),
            })

        return formatted_evidence


__all__ = [
    "VectorIndex",
    "HistoricalRetriever",
    "EvidenceRetriever",
    "get_default_retriever",
]
