"""CLI script to analyze brands in the Twitter Customer Support dataset.

Loads a sample of raw records, prints brand comparison metrics (volume, linkage,
average length, diagnostic content vs generic redirects), and outputs the defensible
selection rationale.
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brand_selection import analyze_brands, print_brand_report


def main():
    parser = argparse.ArgumentParser(description="Analyze brands from Twitter Customer Support dataset")
    parser.add_argument("--source", type=str, default=None, help="Path to archive.zip or twcs.csv")
    parser.add_argument("--sample_rows", type=int, default=500000, help="Number of rows to scan (default: 500,000)")
    args = parser.parse_args()

    results = analyze_brands(source_path=args.source, sample_rows=args.sample_rows)
    print_brand_report(results)


if __name__ == "__main__":
    main()
