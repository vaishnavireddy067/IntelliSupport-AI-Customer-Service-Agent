"""Script to create a stratified golden evaluation set template (150-250 examples).

Samples authentic customer tweets exhibiting varied linguistic challenges:
- Short messages (< 35 chars)
- Slang and misspellings
- Multi-intent expressions
- Ambiguous / low-context queries
- Highly frustrated or urgent complaints
- Hardware / repair inquiries

Leaves gold_intent and gold_escalation fields unpopulated or explicitly flagged
for manual human audit to prevent synthetic label fabrication.
"""

import os
import sys
import re
import argparse
import logging
import pandas as pd

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import clean_customer_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def classify_difficulty_and_notes(text: str) -> tuple[str, str]:
    """Assess linguistic difficulty and diagnostic reason for human annotation."""
    text_lower = text.lower()
    reasons = []
    difficulty = "medium"

    # Check for short or incomplete context
    if len(text) < 35:
        reasons.append("Very short text / low context")
        difficulty = "hard"

    # Check for customer anger / frustration
    if any(w in text_lower for w in ["hate", "terrible", "worst", "unacceptable", "angry", "fuck", "ridiculous", "useless"]):
        reasons.append("Angry / emotionally charged")

    # Check for slang and typos
    if any(w in text_lower for w in ["pls", "plz", "wont", "cant", "idk", "smh", "af", "gonna", "thru", "bcuz", "cuz"]):
        reasons.append("Slang / abbreviations / typos")

    # Check for multi-intent signals
    if (" and " in text_lower or " but " in text_lower or " also " in text_lower) and len(text) > 80:
        reasons.append("Potential multi-intent / compound complaint")
        difficulty = "hard"

    # Check for ambiguous queries
    if any(w in text_lower for w in ["help me", "fix this", "why is this", "what is wrong", "not working"]) and len(text) < 50:
        reasons.append("Ambiguous symptom description")
        difficulty = "hard"

    # Standard clean report
    if not reasons:
        reasons.append("Standard single-issue report")
        difficulty = "easy"

    return difficulty, "; ".join(reasons)


def generate_golden_template(
    data_path: str = "data/processed/test.csv",
    output_path: str = "data/golden/golden_set.csv",
    n_samples: int = 200,
    random_seed: int = 42,
):
    """Generate golden set candidate template for human annotation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source test dataset not found at {data_path}. Run scripts/prepare_data.py first.")

    df = pd.read_csv(data_path)
    logger.info("Loaded %d test candidates from %s.", len(df), data_path)

    # Ensure clean text
    df["cleaned_text"] = df["customer_text"].apply(clean_customer_text)
    df = df[df["cleaned_text"].str.len() > 5].drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)

    # Stratified difficulty assignment
    difficulties = []
    notes_list = []
    for t in df["cleaned_text"]:
        diff, note = classify_difficulty_and_notes(t)
        difficulties.append(diff)
        notes_list.append(note)

    df["difficulty"] = difficulties
    df["notes"] = notes_list

    # Sample balanced set across difficulty categories
    samples_per_diff = n_samples // 3
    sampled_dfs = []
    for diff in ["easy", "medium", "hard"]:
        sub_df = df[df["difficulty"] == diff]
        if len(sub_df) >= samples_per_diff:
            sampled_dfs.append(sub_df.sample(n=samples_per_diff, random_state=random_seed))
        else:
            sampled_dfs.append(sub_df)

    golden_df = pd.concat(sampled_dfs).reset_index(drop=True)
    if len(golden_df) < n_samples:
        remainder = n_samples - len(golden_df)
        remaining_pool = df[~df["customer_tweet_id"].isin(golden_df["customer_tweet_id"])]
        if len(remaining_pool) > 0:
            golden_df = pd.concat([golden_df, remaining_pool.sample(n=min(remainder, len(remaining_pool)), random_state=random_seed)]).reset_index(drop=True)

    golden_df["id"] = range(1, len(golden_df) + 1)
    golden_df["text"] = golden_df["cleaned_text"]

    # In strict compliance with guidelines: human labels are NOT fabricated.
    golden_df["gold_intent"] = ""  # Left for verified human audit
    golden_df["gold_escalation"] = ""  # Left for verified human audit

    final_columns = [
        "id",
        "customer_tweet_id",
        "text",
        "gold_intent",
        "gold_escalation",
        "difficulty",
        "notes",
    ]
    golden_df = golden_df[final_columns]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Created golden set template with %d records at %s.", len(golden_df), output_path)
    logger.info("Difficulty breakdown:\n%s", golden_df["difficulty"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Golden Evaluation Set template")
    parser.add_argument("--data_path", type=str, default="data/processed/test.csv", help="Source test CSV")
    parser.add_argument("--output_path", type=str, default="data/golden/golden_set.csv", help="Output CSV path")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of examples (150-250)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_golden_template(
        data_path=args.data_path,
        output_path=args.output_path,
        n_samples=args.n_samples,
        random_seed=args.seed,
    )
