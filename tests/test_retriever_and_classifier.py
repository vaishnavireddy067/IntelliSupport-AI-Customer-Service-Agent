"""Unit tests for top-level retriever and intent classifier wrappers."""

import pytest
from src.retriever import EvidenceRetriever, VectorIndex, HistoricalRetriever
from src.intent_classifier import IntentClassifier, EmbeddingIntentClassifier
import numpy as np


def test_vector_index_search_top_k():
    index = VectorIndex(use_faiss=False)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], dtype=np.float32)
    metadata = [
        {"customer_tweet_id": "1", "customer_text": "query A", "agent_text": "reply A"},
        {"customer_tweet_id": "2", "customer_text": "query B", "agent_text": "reply B"},
        {"customer_tweet_id": "3", "customer_text": "query C", "agent_text": "reply C"},
    ]
    index.build(embeddings, metadata)

    retriever = HistoricalRetriever(index=index)
    results = index.search(np.array([1.0, 0.0]), top_k=2)

    assert len(results) == 2
    assert results[0]["customer_tweet_id"] == "1"
    assert results[0]["similarity_score"] > 0.99


def test_evidence_retriever_formatting():
    index = VectorIndex(use_faiss=False)
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    metadata = [{"customer_tweet_id": "101", "customer_text": "my battery dies", "agent_text": "check settings"}]
    index.build(embeddings, metadata)

    h_retriever = HistoricalRetriever(index=index)
    ev_retriever = EvidenceRetriever(retriever=h_retriever)
    
    # Mock index search
    results = ev_retriever.retriever.index.search(np.array([1.0, 0.0]), top_k=1)
    assert len(results) == 1
    assert results[0]["agent_text"] == "check settings"
