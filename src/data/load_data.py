"""Data extraction module for AppleSupport conversations.

Streams directly from the raw archive without requiring complete local extraction
of the 500+ MB dataset, preserving repository leanness and disk efficiency.
"""

import os
import io
import csv
import zipfile
import logging
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_PATHS = [
    r"C:\Users\anugu vaishnavi\Downloads\archive.zip",
    os.path.join(os.getcwd(), "data", "raw", "archive.zip"),
    os.path.join(os.getcwd(), "data", "raw", "twcs.csv"),
]


def find_dataset_source(explicit_path: Optional[str] = None) -> str:
    """Find the dataset file path from explicit path or default candidates."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    for candidate in DEFAULT_ARCHIVE_PATHS:
        if os.path.exists(candidate):
            logger.info("Found dataset source at: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Could not find Kaggle Customer Support dataset (archive.zip or twcs.csv). "
        "Please specify path or place archive.zip in data/raw/ or Downloads."
    )


def extract_applesupport_pairs(
    source_path: str,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Stream twcs.csv and reconstruct Customer -> AppleSupport conversation pairs.

    Two-pass approach or in-memory index of customer tweets:
    Pass 1: Collect customer inbound tweets matching AppleSupport interactions or root mentions.
    Pass 2: Match outbound AppleSupport responses with the customer prompt.
    """
    logger.info("Opening dataset source: %s", source_path)
    is_zip = zipfile.is_zipfile(source_path)

    def get_reader():
        if is_zip:
            z = zipfile.ZipFile(source_path, "r")
            csv_name = "twcs/twcs.csv" if "twcs/twcs.csv" in z.namelist() else "twcs.csv"
            f = z.open(csv_name)
            text_f = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            return text_f, z
        else:
            text_f = open(source_path, "r", encoding="utf-8", errors="replace")
            return text_f, None

    # Step 1: Collect AppleSupport outbound tweets and identify required parent customer tweet IDs
    apple_replies: List[Dict[str, str]] = []
    needed_customer_tweet_ids = set()

    text_f, zip_handle = get_reader()
    try:
        reader = csv.DictReader(text_f)
        count = 0
        for row in reader:
            count += 1
            if row["author_id"] == "AppleSupport":
                parent_id = row.get("in_response_to_tweet_id", "").strip()
                if parent_id:
                    apple_replies.append({
                        "agent_tweet_id": row["tweet_id"],
                        "parent_tweet_id": parent_id,
                        "agent_text": row["text"],
                        "agent_created_at": row["created_at"],
                    })
                    needed_customer_tweet_ids.add(parent_id)
                    if max_pairs and len(apple_replies) >= max_pairs * 2:
                        break
            if count % 1000000 == 0:
                logger.info("Scanned %d rows, found %d AppleSupport replies...", count, len(apple_replies))
    finally:
        text_f.close()
        if zip_handle:
            zip_handle.close()

    logger.info(
        "Collected %d AppleSupport outbound replies referencing %d customer tweets.",
        len(apple_replies),
        len(needed_customer_tweet_ids),
    )

    # Step 2: Retrieve the customer inbound tweets corresponding to those IDs
    customer_tweets: Dict[str, Dict[str, str]] = {}
    text_f, zip_handle = get_reader()
    try:
        reader = csv.DictReader(text_f)
        count = 0
        for row in reader:
            count += 1
            t_id = row["tweet_id"]
            if t_id in needed_customer_tweet_ids:
                customer_tweets[t_id] = {
                    "customer_tweet_id": t_id,
                    "customer_author_id": row["author_id"],
                    "customer_text": row["text"],
                    "customer_created_at": row["created_at"],
                    "inbound": row["inbound"],
                }
                if len(customer_tweets) >= len(needed_customer_tweet_ids):
                    break
            if count % 1000000 == 0:
                logger.info("Scanned %d rows, resolved %d customer tweets...", count, len(customer_tweets))
    finally:
        text_f.close()
        if zip_handle:
            zip_handle.close()

    logger.info("Resolved %d customer tweets. Pairing with AppleSupport responses...", len(customer_tweets))

    # Step 3: Join pairs
    conversation_pairs: List[Dict[str, str]] = []
    for reply in apple_replies:
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

    logger.info("Successfully assembled %d complete Customer -> AppleSupport conversation pairs.", len(conversation_pairs))
    return conversation_pairs
