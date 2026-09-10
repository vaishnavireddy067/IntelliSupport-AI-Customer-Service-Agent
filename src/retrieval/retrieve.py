"""Historical response retriever.

Encodes incoming customer queries and searches the vector index for historically
similar AppleSupport conversations and their associated resolutions.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure offline loading from local cache for sub-second latency
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


class HistoricalRetriever:
    """Retrieves top-k historically similar customer inquiries and brand responses."""

    def __init__(
        self,
        index: VectorIndex,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.index = index
        self.model_name = model_name
        self._encoder = None

    @property
    def encoder(self):
        """Lazy load SentenceTransformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single query string into a normalized embedding vector."""
        return self.encoder.encode([text], normalize_embeddings=True)[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts into normalized embeddings."""
        return self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def retrieve(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k historical customer conversations matching the query."""
        if not query_text or not query_text.strip():
            return []

        q_vec = self.encode_text(query_text)
        results = self.index.search(q_vec, top_k=top_k)
        return results

    def retrieve_batch(self, queries: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """Retrieve top-k historical conversations for a batch of queries."""
        if not queries:
            return []
        q_vecs = self.encode_batch(queries)
        return [self.index.search(vec, top_k=top_k) for vec in q_vecs]
