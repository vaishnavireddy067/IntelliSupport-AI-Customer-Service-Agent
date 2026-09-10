"""Script to extract, preprocess, sample, and split AppleSupport conversations.

Extracts pairs directly from the archive without copying the 500+ MB file into git.
Generates reproducible train, validation, and test splits with fixed seed.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import find_dataset_source, extract_applesupport_pairs
from src.data.preprocess import preprocess_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(
    source_path: str = None,
    output_dir: str = "data/processed",
    sample_size: int = 25000,
    random_seed: int = 42,
):
    """Main data preparation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    resolved_source = find_dataset_source(source_path)
    logger.info("Extracting AppleSupport conversation pairs from %s...", resolved_source)

    # Extract conversation pairs from archive
    raw_pairs = extract_applesupport_pairs(resolved_source, max_pairs=sample_size * 2 if sample_size else None)
    logger.info("Extracted %d raw conversation pairs.", len(raw_pairs))

    # Preprocess and clean records
    cleaned_records = []
    for pair in raw_pairs:
        cleaned = preprocess_pair(pair)
        if cleaned:
            cleaned_records.append(cleaned)

    logger.info("Retained %d usable conversation pairs after preprocessing.", len(cleaned_records))
    df = pd.DataFrame(cleaned_records)

    # Deduplicate on customer_text to avoid identical tweets
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    logger.info("Retained %d unique customer conversations after deduplication.", len(df))

    # Sample if sample_size is requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        logger.info("Sampled down to %d conversations (random_seed=%d).", len(df), random_seed)

    # Save full processed dataset
    full_path = os.path.join(output_dir, "apple_conversations.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    logger.info("Saved full processed dataset to %s", full_path)

    # Stratified or standard split: Train (70%), Val (15%), Test (15%)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info("Data splits generated:")
    logger.info("  Train: %d rows -> %s", len(train_df), train_path)
    logger.info("  Val:   %d rows -> %s", len(val_df), val_path)
    logger.info("  Test:  %d rows -> %s", len(test_df), test_path)

    return df, train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AppleSupport dataset from Kaggle twcs")
    parser.add_argument("--source", type=str, default=None, help="Path to archive.zip or twcs.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--sample_size", type=int, default=25000, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    prepare_dataset(
        source_path=args.source,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
