"""Interactive CLI labeling tool for manual audit of golden evaluation set examples.

Allows reviewers to inspect customer tweets, view candidate classifications, and
manually accept or override intent, action, and notes in data/golden_eval.csv.
"""

import os
import sys
import pandas as pd

CSV_PATH = "data/golden_eval.csv"


def run_interactive_labeler():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run scripts/build_golden_set.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} golden evaluation records from {CSV_PATH}.")
    print("Press Enter to accept default, or type a new value. Type 'q' to quit and save.\n")

    for idx, row in df.iterrows():
        print("-" * 60)
        print(f"Example [{idx + 1}/{len(df)}] - ID: {row['example_id']}")
        print(f"Message:    \"{row['customer_message']}\"")
        print(f"Current Intent: {row['gold_intent']}")
        print(f"Current Action: {row['gold_action']}")
        print(f"Current Reason: {row['gold_reason']}")
        print("-" * 60)

        # Prompt for intent
        val = input(f"New Intent [{row['gold_intent']}] (or 'q' to quit, Enter to keep): ").strip()
        if val.lower() == "q":
            break
        if val:
            df.at[idx, "gold_intent"] = val

        # Prompt for action
        act = input(f"New Action [{row['gold_action']}] (A: AUTO_HANDLE / E: ESCALATE, Enter to keep): ").strip().upper()
        if act == "A":
            df.at[idx, "gold_action"] = "AUTO_HANDLE"
        elif act == "E":
            df.at[idx, "gold_action"] = "ESCALATE"
        elif act in ["AUTO_HANDLE", "ESCALATE"]:
            df.at[idx, "gold_action"] = act

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\nChanges saved to {CSV_PATH}.")


if __name__ == "__main__":
    run_interactive_labeler()
