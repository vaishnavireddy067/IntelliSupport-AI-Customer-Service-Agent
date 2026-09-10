"""Unit tests for Phase 1 data loader and conversation-level leakage prevention."""

import pandas as pd
import pytest
from src.data_loader import conversation_level_split


def test_conversation_level_split_proportions():
    # Construct synthetic conversations
    records = [{"customer_tweet_id": str(i), "customer_text": f"Issue query {i}", "agent_text": f"Resolution {i}"} for i in range(100)]
    df = pd.DataFrame(records)

    train_df, val_df, test_df = conversation_level_split(df, train_size=0.70, val_size=0.15, test_size=0.15, random_seed=42)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15


def test_conversation_level_split_no_leakage():
    # Verify strict disjointness between train and test
    records = [{"customer_tweet_id": str(i), "customer_text": f"Unique query number {i}", "agent_text": f"Resolution {i}"} for i in range(200)]
    df = pd.DataFrame(records)

    train_df, val_df, test_df = conversation_level_split(df, random_seed=42)

    train_texts = set(train_df["customer_text"])
    val_texts = set(val_df["customer_text"])
    test_texts = set(test_df["customer_text"])

    assert len(train_texts.intersection(test_texts)) == 0, "Data leakage detected between train and test splits!"
    assert len(train_texts.intersection(val_texts)) == 0, "Data leakage detected between train and val splits!"
    assert len(val_texts.intersection(test_texts)) == 0, "Data leakage detected between val and test splits!"
