# Data Directory Structure & Management

This directory contains processed datasets, splits, and the golden evaluation benchmark.
To ensure repository leanness and stay within GitHub upload limits, the raw 516 MB Kaggle archive is **not** committed to git.

---

## Directory Contents

- `golden_eval.csv`: 200 hand-curated, stratified evaluation examples across all 10 empirical intents and difficulty tiers (short, noisy, slang, multi-intent, high emotion). Documented in detail in `docs/golden_set.md`.
- `processed/`: Cached conversation-level dataset splits generated via `scripts/sample_data.py`:
  - `apple_conversations.csv`: 25,000 deduplicated Customer -> AppleSupport dialogue pairs.
  - `train.csv`: 17,500 pairs (70%) for intent model training and retrieval indexing.
  - `val.csv`: 3,750 pairs (15%) for validation and threshold calibration.
  - `test.csv`: 3,750 pairs (15%) strictly held out for evaluation (0% query overlap with train/val).
  - `retrieval_index/`: Normalized dense vector embeddings (`embeddings.npy`) and resolution metadata (`metadata.pkl`) for 15,000 historical Apple resolutions.
  - `embedding_classifier.pkl`: Serialized `all-MiniLM-L6-v2` calibrated classifier.
  - `tfidf_classifier.pkl`: Serialized Baseline 2 TF-IDF classifier.
- `golden/`: Supporting human rating audit templates (`human_ratings_template.csv`).

---

## Zero-Leakage Guarantee

All splits are partitioned at the **conversation level** after customer query deduplication.
There are **zero overlapping customer inquiries** between `train.csv` and `test.csv`.
