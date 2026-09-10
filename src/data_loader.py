"""Data loading, extraction, and split management module.

Handles streaming extraction from raw customer-support archives, conversation-level
splitting to prevent data leakage, and loading cached processed datasets.
"""

import os
import io
import csv
import zipfile
import logging
from typing import Optional, Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_PATHS = [
    r"C:\Users\anugu vaishnavi\Downloads\archive.zip",
    os.path.join(os.getcwd(), "data", "raw", "archive.zip"),
    os.path.join(os.getcwd(), "data", "raw", "twcs.csv"),
]


def find_dataset_source(explicit_path: Optional[str] = None) -> str:
    """Locate the dataset source archive or CSV file."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for p in DEFAULT_ARCHIVE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find Kaggle Customer Support dataset (archive.zip or twcs.csv). "
        "Please specify path or place archive.zip in data/raw/ or Downloads."
    )


def extract_brand_pairs(
    source_path: str,
    brand_name: str = "AppleSupport",
    max_pairs: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Stream twcs.csv and reconstruct Customer -> Brand conversation pairs.

    Uses a memory-efficient two-pass streaming strategy:
    Pass 1: Collect brand outbound replies and their required parent customer tweet IDs.
    Pass 2: Stream raw customer inbound tweets to match required parent IDs.
    """
    is_zip = zipfile.is_zipfile(source_path)

    def get_reader():
        if is_zip:
            z = zipfile.ZipFile(source_path, "r")
            csv_name = "twcs/twcs.csv" if "twcs/twcs.csv" in z.namelist() else "twcs.csv"
            f = z.open(csv_name)
            return io.TextIOWrapper(f, encoding="utf-8", errors="replace"), z
        else:
            return open(source_path, "r", encoding="utf-8", errors="replace"), None

    logger.info("Pass 1: Scanning for '%s' outbound support responses...", brand_name)
    brand_replies: List[Dict[str, str]] = []
    needed_customer_tweet_ids = set()

    text_f, zip_handle = get_reader()
    try:
        reader = csv.DictReader(text_f)
        count = 0
        for row in reader:
            count += 1
            if row["author_id"] == brand_name:
                parent_id = row.get("in_response_to_tweet_id", "").strip()
                if parent_id:
                    brand_replies.append({
                        "agent_tweet_id": row["tweet_id"],
                        "parent_tweet_id": parent_id,
                        "agent_text": row["text"],
                        "agent_created_at": row["created_at"],
                    })
                    needed_customer_tweet_ids.add(parent_id)
                    if max_pairs and len(brand_replies) >= max_pairs * 2:
                        break
    finally:
        text_f.close()
        if zip_handle:
            zip_handle.close()

    logger.info("Found %d outbound replies referencing %d parent customer tweets.",
                len(brand_replies), len(needed_customer_tweet_ids))

    logger.info("Pass 2: Resolving matching customer queries...")
    customer_tweets: Dict[str, Dict[str, str]] = {}
    text_f, zip_handle = get_reader()
    try:
        reader = csv.DictReader(text_f)
        for row in reader:
            t_id = row["tweet_id"]
            if t_id in needed_customer_tweet_ids:
                customer_tweets[t_id] = {
                    "customer_tweet_id": t_id,
                    "customer_author_id": row["author_id"],
                    "customer_text": row["text"],
                    "customer_created_at": row["created_at"],
                }
                if len(customer_tweets) >= len(needed_customer_tweet_ids):
                    break
    finally:
        text_f.close()
        if zip_handle:
            zip_handle.close()

    # Reconstruct paired conversations
    conversation_pairs: List[Dict[str, str]] = []
    for reply in brand_replies:
        parent_id = reply["parent_tweet_id"]
        if parent_id in customer_tweets:
            cust = customer_tweets[parent_id]
            conversation_pairs.append({
                "customer_tweet_id": cust["customer_tweet_id"],
                "customer_author_id": cust["customer_author_id"],
                "customer_text": cust["customer_text"],
                "customer_created_at": cust["customer_created_at"],
                "agent_tweet_id": reply["agent_tweet_id"],
                "agent_text": reply["agent_text"],
                "agent_created_at": reply["agent_created_at"],
            })
            if max_pairs and len(conversation_pairs) >= max_pairs:
                break

    logger.info("Assembled %d valid Customer -> %s pairs.", len(conversation_pairs), brand_name)
    return conversation_pairs


def conversation_level_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset at the conversation level to strictly prevent data leakage.

    Splits distinct customer threads so no conversation or customer prompt
    appears in both the retrieval/training index and the test evaluation split.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split proportions must sum to 1.0"
    
    # Deduplicate on customer text first to ensure identical customer inquiries do not leak
    deduped_df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    
    train_df, temp_df = train_test_split(
        deduped_df,
        test_size=(val_size + test_size),
        random_state=random_seed,
    )
    
    relative_test_ratio = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_processed_splits(
    processed_dir: str = "data/processed",
) -> Dict[str, pd.DataFrame]:
    """Load train, val, test splits from disk."""
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Processed split files missing in {processed_dir}. Run scripts/sample_data.py first.")

    return {
        "train": pd.read_csv(train_path),
        "val": pd.read_csv(val_path) if os.path.exists(val_path) else None,
        "test": pd.read_csv(test_path),
    }
