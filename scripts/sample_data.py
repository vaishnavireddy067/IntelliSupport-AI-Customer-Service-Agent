"""Reproducible sampling and conversation pair extraction pipeline.

Implements Phase 1 data pipeline:
- Streams raw Twitter archive without copying 500+ MB into Git
- Filters target brand (default: AppleSupport)
- Cleans and deduplicates customer-agent pairs
- Performs conversation-level splits with fixed random seed
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import find_dataset_source, extract_brand_pairs, conversation_level_split
from src.preprocessing import preprocess_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str = "configs/config.yaml") -> dict:
    """Load config.yaml if available."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_sampling(
    source_path: str = None,
    brand_name: str = "AppleSupport",
    sample_size: int = 25000,
    output_dir: str = "data/processed",
    random_seed: int = 42,
) -> None:
    """Run end-to-end data extraction and splitting."""
    os.makedirs(output_dir, exist_ok=True)
    resolved_source = find_dataset_source(source_path)
    logger.info("Starting Phase 1 extraction from: %s", resolved_source)
    logger.info("Target Brand: %s | Target Pairs: %d | Seed: %d", brand_name, sample_size, random_seed)

    # 1. Extract raw conversation pairs
    raw_pairs = extract_brand_pairs(
        source_path=resolved_source,
        brand_name=brand_name,
        max_pairs=sample_size * 2 if sample_size else None,
    )
    logger.info("Extracted %d raw paired interactions.", len(raw_pairs))

    # 2. Clean and validate pairs
    cleaned_records = []
    for pair in raw_pairs:
        cleaned = preprocess_pair(pair)
        if cleaned:
            cleaned_records.append(cleaned)
    logger.info("Retained %d usable pairs after text cleaning.", len(cleaned_records))

    df = pd.DataFrame(cleaned_records)

    # 3. Deduplicate customer texts to eliminate identical repeat inquiries
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    logger.info("Retained %d unique customer inquiries after deduplication.", len(df))

    # 4. Sample down to target size
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        logger.info("Downsampled to %d records.", len(df))

    # Save full processed dataset
    full_path = os.path.join(output_dir, f"{brand_name.lower()}_conversations.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    logger.info("Saved complete dataset to: %s", full_path)

    # 5. Conversation-level train/val/test split
    train_df, val_df, test_df = conversation_level_split(
        df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_seed=random_seed,
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info("Splits generated successfully:")
    logger.info("  Train: %d rows (70%%) -> %s", len(train_df), train_path)
    logger.info("  Val:   %d rows (15%%) -> %s", len(val_df), val_path)
    logger.info("  Test:  %d rows (15%%) -> %s", len(test_df), test_path)


if __name__ == "__main__":
    cfg = load_yaml_config()
    default_brand = cfg.get("brand", {}).get("name", "AppleSupport")
    default_size = cfg.get("data", {}).get("brand_sample_size", 25000)
    default_seed = cfg.get("data", {}).get("random_seed", 42)

    parser = argparse.ArgumentParser(description="Extract, clean, and sample customer support conversations")
    parser.add_argument("--source", type=str, default=None, help="Path to raw archive.zip or twcs.csv")
    parser.add_argument("--brand", type=str, default=default_brand, help="Target brand handle")
    parser.add_argument("--sample_size", type=int, default=default_size, help="Target number of conversations")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed")

    args = parser.parse_args()
    run_sampling(
        source_path=args.source,
        brand_name=args.brand,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
