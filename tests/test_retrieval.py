"""Unit tests for historical resolution retrieval and vector indexing."""

import pytest
import os
import tempfile
import numpy as np
from src.retrieval.index import VectorIndex


def test_vector_index_search_and_ranking():
    # Synthetic normalized 4-dimensional embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Item 1
        [0.0, 1.0, 0.0, 0.0],  # Item 2
        [0.7071, 0.7071, 0.0, 0.0],  # Item 3 (45 deg to 1 & 2)
    ], dtype=np.float32)

    metadata = [
        {"agent_tweet_id": "101", "customer_text": "Battery drain issue", "agent_text": "Check battery health"},
        {"agent_tweet_id": "102", "customer_text": "Screen frozen", "agent_text": "Force restart phone"},
        {"agent_tweet_id": "103", "customer_text": "Battery and screen glitch", "agent_text": "Update iOS"},
    ]

    index = VectorIndex(use_faiss=False)
    index.build(embeddings, metadata)

    # Query identical to Item 1
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["agent_tweet_id"] == "101"
    assert round(results[0]["similarity_score"], 2) == 1.00
    assert results[1]["agent_tweet_id"] == "103"
    assert round(results[1]["similarity_score"], 2) == 0.71


def test_vector_index_save_and_load():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = [
        {"customer_text": "A", "agent_text": "Resp A"},
        {"customer_text": "B", "agent_text": "Resp B"},
    ]
    index = VectorIndex(use_faiss=False)
    index.build(embeddings, metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        index.save(tmpdir)
        loaded = VectorIndex.load(tmpdir, use_faiss=False)
        assert len(loaded.metadata) == 2
        res = loaded.search(np.array([1.0, 0.0]), top_k=1)
        assert res[0]["customer_text"] == "A"
