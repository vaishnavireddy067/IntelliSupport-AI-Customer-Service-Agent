"""CLI script to run intent discovery on the AppleSupport dataset."""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_discovery import print_discovery_report


def main():
    parser = argparse.ArgumentParser(description="Run empirical intent discovery on customer inquiries")
    parser.add_argument("--data_path", type=str, default="data/processed/train.csv", help="Path to training set")
    parser.add_argument("--taxonomy_path", type=str, default="configs/intents.yaml", help="Path to taxonomy YAML")
    args = parser.parse_args()

    print_discovery_report(data_path=args.data_path, taxonomy_path=args.taxonomy_path)


if __name__ == "__main__":
    main()
