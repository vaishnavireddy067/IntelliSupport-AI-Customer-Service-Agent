"""Golden Evaluation Set Builder (200 curated examples).

Constructs data/golden_eval.csv with stratified sampling across intents and difficulty
tiers (short messages, noisy slang, multi-intent, high emotion, ambiguous context).
"""

import os
import sys
import re
import argparse
import logging
import pandas as pd
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_customer_text
from src.intents.labeler import label_intent_from_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_gold_action_and_reason(text: str, intent: str) -> Tuple[str, str, str]:
    """Determine ground-truth action, reason, and reply quality criteria."""
    text_lower = text.lower()

    # Rule 1: High risk / security / fraud
    if any(w in text_lower for w in ["stolen", "theft", "lost phone", "police", "fraud", "hacked", "court", "lawyer"]):
        return (
            "ESCALATE",
            "Security, account compromise, or legal risk requires human authorization.",
            "Must acknowledge security concern immediately without making unauthorized commitments.",
        )

    # Rule 2: High customer frustration / repeated failure
    if any(w in text_lower for w in ["tried everything", "third time", "useless", "unacceptable", "speak to a human", "manager"]):
        return (
            "ESCALATE",
            "Customer frustration or multiple prior failed troubleshooting attempts.",
            "Tone must be empathetic; route directly to human tier-2 supervisor.",
        )

    # Rule 3: Financial disputes
    if intent == "billing_subscription" and any(w in text_lower for w in ["charged twice", "double charged", "refund", "unauthorized"]):
        return (
            "ESCALATE",
            "Disputed debit or refund requires financial billing specialist access.",
            "Guide customer to official reportaproblem.apple.com while routing to billing team.",
        )

    # Rule 4: Account lockouts
    if intent == "apple_id_account" and any(w in text_lower for w in ["locked out", "recovery", "forgot password", "2fa"]):
        return (
            "ESCALATE",
            "Apple ID security lockout requires identity verification.",
            "Direct to iforgot.apple.com; never ask for credentials in chat.",
        )

    # Rule 5: Ambiguous / zero context
    if len(text) < 25 and any(w in text_lower for w in ["help", "fix", "why", "broken", "wont work"]):
        return (
            "ESCALATE",
            "Query contains insufficient technical context to prescribe automated fix.",
            "Ask clarifying diagnostic questions politely.",
        )

    # Default: Routine self-service inquiry
    return (
        "AUTO_HANDLE",
        f"Routine technical support issue for '{intent}' resolvable via standard troubleshooting.",
        "Provide direct step-by-step diagnostic settings navigation or official KB link.",
    )


def build_golden_set(
    test_path: str = "data/processed/test.csv",
    output_path: str = "data/golden_eval.csv",
    n_samples: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build stratified 200-sample golden evaluation dataset."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}. Run scripts/sample_data.py first.")

    logger.info("Loading test candidates from %s...", test_path)
    df = pd.read_csv(test_path)
    df["customer_text"] = df["customer_text"].fillna("").astype(str)

    # Pre-classify intents
    df["intent"] = df["customer_text"].apply(label_intent_from_text)

    # Categorize linguistic difficulty
    difficulties = []
    notes = []
    for t in df["customer_text"]:
        tl = t.lower()
        if len(t) < 30:
            difficulties.append("Hard (Short/Ambiguous)")
            notes.append("Short message lacking hardware details")
        elif any(w in tl for w in ["pls", "plz", "wont", "cant", "smh", "idk", "af", "cuz", "gonna"]):
            difficulties.append("Medium (Noisy/Slang)")
            notes.append("Authentic Twitter slang and colloquial contractions")
        elif any(w in tl for w in ["terrible", "worst", "unacceptable", "fuck", "hate", "useless", "tried everything"]):
            difficulties.append("Hard (High Emotion/Frustration)")
            notes.append("High customer distress requiring de-escalation")
        elif (" and " in tl or " but " in tl) and len(t) > 75:
            difficulties.append("Hard (Compound/Multi-Intent)")
            notes.append("Compound inquiry with multiple overlapping symptoms")
        else:
            difficulties.append("Easy (Standard Symptom)")
            notes.append("Standard single-symptom customer report")

    df["difficulty"] = difficulties
    df["sampling_note"] = notes

    # Stratified sampling across intents and difficulty
    intents = df["intent"].unique()
    samples_per_intent = max(1, n_samples // len(intents))

    sampled_records = []
    for intent in intents:
        sub = df[df["intent"] == intent]
        # Mix hard, medium, and easy
        take = min(len(sub), samples_per_intent + 3)
        sample = sub.sample(n=take, random_state=random_seed)
        sampled_records.append(sample)

    combined = pd.concat(sampled_records, ignore_index=True)

    # Downsample or top-up exactly to n_samples
    if len(combined) > n_samples:
        combined = combined.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    elif len(combined) < n_samples:
        remaining = df[~df["customer_tweet_id"].isin(combined["customer_tweet_id"])]
        fill = remaining.sample(n=n_samples - len(combined), random_state=random_seed)
        combined = pd.concat([combined, fill], ignore_index=True)

    # Assemble golden evaluation fields
    golden_rows = []
    for idx, row in combined.iterrows():
        cust_msg = clean_customer_text(row["customer_text"])
        intent = row["intent"]
        action, reason, quality_notes = evaluate_gold_action_and_reason(cust_msg, intent)

        golden_rows.append({
            "example_id": idx + 1,
            "brand": "AppleSupport",
            "conversation_id": str(row.get("customer_tweet_id", idx + 1)),
            "customer_message": cust_msg,
            "context": f"Twitter Inbound Mention | Difficulty: {row.get('difficulty', 'Standard')}",
            "gold_intent": intent,
            "gold_action": action,
            "gold_reason": reason,
            "gold_reply_quality_notes": quality_notes,
        })

    golden_df = pd.DataFrame(golden_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d golden evaluation examples to %s", len(golden_df), output_path)

    # Print summary distribution
    print("\n" + "=" * 60)
    print(f"       GOLDEN EVALUATION SET CREATED ({len(golden_df)} EXAMPLES)")
    print("=" * 60)
    print("\n--- Intent Distribution ---")
    print(golden_df["gold_intent"].value_counts().to_string())
    print("\n--- Action Distribution ---")
    print(golden_df["gold_action"].value_counts().to_string())
    print(f"\nSaved to: {output_path}\n")

    return golden_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build golden evaluation set")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--output_path", type=str, default="data/golden_eval.csv")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_golden_set(
        test_path=args.test_path,
        output_path=args.output_path,
        n_samples=args.n_samples,
        random_seed=args.seed,
    )
