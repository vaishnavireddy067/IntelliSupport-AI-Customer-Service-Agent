"""Script to train and evaluate Baseline 1 and Baseline 2 on the test set.

Runs:
1. Baseline 1: Majority Class Classifier + Generic Fallback
2. Baseline 2: TF-IDF + Logistic Regression + TF-IDF Retrieval
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.majority import MajorityIntentClassifier, MajoritySupportAgent
from baselines.tfidf_baseline import TfidfIntentClassifier, TfidfSupportAgent
from src.intents.labeler import label_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_classifier(name: str, y_true: list, y_pred: list) -> dict:
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "model": name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Baseline models evaluation")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--output_json", type=str, default="reports/baseline_comparison.json")
    args = parser.parse_args()

    if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
        logger.error("Data splits missing. Run scripts/sample_data.py first.")
        return

    logger.info("Loading train and test splits...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Label intents using taxonomy rules
    label_dataframe(train_df, text_column="customer_text", target_column="intent")
    label_dataframe(test_df, text_column="customer_text", target_column="intent")

    X_train = train_df["customer_text"].fillna("").tolist()
    y_train = train_df["intent"].tolist()
    X_test = test_df["customer_text"].fillna("").tolist()
    y_test = test_df["intent"].tolist()

    logger.info("Train samples: %d | Test samples: %d", len(X_train), len(X_test))

    # Verify zero leakage
    overlap = set(X_train).intersection(set(X_test))
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping queries between train and test!"
    logger.info("Leakage verification passed: 0 query overlaps.")

    # 1. Baseline 1: Majority Class
    logger.info("Fitting Baseline 1: MajorityIntentClassifier...")
    majority_clf = MajorityIntentClassifier().fit(X_train, y_train)
    y_pred_maj = majority_clf.predict(X_test)
    maj_metrics = evaluate_classifier("Baseline 1: Majority Class", y_test, y_pred_maj)

    # 2. Baseline 2: TF-IDF + Logistic Regression
    logger.info("Fitting Baseline 2: TfidfIntentClassifier...")
    tfidf_clf = TfidfIntentClassifier(max_features=10000).fit(X_train, y_train)
    y_pred_tfidf = tfidf_clf.predict(X_test)
    tfidf_metrics = evaluate_classifier("Baseline 2: TF-IDF + Logistic Reg", y_test, y_pred_tfidf)

    # Print comparative Markdown Table
    print("\n" + "=" * 60)
    print("           BASELINE EVALUATION RESULTS TABLE")
    print("=" * 60 + "\n")
    print("| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in [maj_metrics, tfidf_metrics]:
        print(f"| {m['model']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} |")

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"majority": maj_metrics, "tfidf": tfidf_metrics}, f, indent=2)
    logger.info("Saved baseline metrics to %s", args.output_json)


if __name__ == "__main__":
    main()
