"""Vector index for historical AppleSupport customer inquiries and resolutions.

Supports FAISS when available, with automatic high-performance numpy/scipy cosine fallback.
Saves normalized embeddings and metadata to disk for low-latency retrieval.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Check for FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not found. Utilizing high-performance normalized numpy vector search.")


class VectorIndex:
    """Vector index storing customer inquiry embeddings and associated resolution metadata."""

    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.faiss_index = None

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Build the vector index from normalized embeddings and metadata.

        Embeddings must be L2-normalized for cosine similarity via inner product.
        """
        # Ensure L2-normalized float32
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.metadata = metadata

        if self.use_faiss:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            logger.info("Built FAISS IndexFlatIP with %d vectors (dimension: %d)", len(self.metadata), dim)
        else:
            logger.info("Built Numpy vector index with %d vectors (dimension: %d)", len(self.metadata), self.embeddings.shape[1])

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search top-k most similar historical records for a query vector.

        query_embedding: shape (1, dim) or (dim,)
        """
        if self.embeddings is None or len(self.metadata) == 0:
            return []

        # Flatten / ensure 2D float32
        q_vec = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_vec, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Normalized dot product = cosine similarity
            sims = np.dot(self.embeddings, q_vec.T).flatten()
            # Top-k indices
            top_k_actual = min(top_k, len(sims))
            indices = np.argpartition(sims, -top_k_actual)[-top_k_actual:]
            indices = indices[np.argsort(-sims[indices])]
            scores = sims[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["similarity_score"] = float(score)
            meta["rank"] = rank
            results.append(meta)

        return results

    def save(self, directory: str):
        """Save vector embeddings and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        emb_path = os.path.join(directory, "embeddings.npy")
        meta_path = os.path.join(directory, "metadata.pkl")

        np.save(emb_path, self.embeddings)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved index to %s (%d records)", directory, len(self.metadata))

    @classmethod
    def load(cls, directory: str, use_faiss: bool = True) -> "VectorIndex":
        """Load index from directory."""
        emb_path = os.path.join(directory, "embeddings.npy")
        meta_path = os.path.join(directory, "metadata.pkl")

        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files missing in {directory}")

        embeddings = np.load(emb_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        idx = cls(use_faiss=use_faiss)
        idx.build(embeddings, metadata)
        return idx
