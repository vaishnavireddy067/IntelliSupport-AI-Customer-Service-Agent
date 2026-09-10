"""Complete experiment execution and evaluation harness.

Runs all 3 intent classifiers on identical test data, evaluates historical retrieval (Recall@K),
tests the escalation policy, generates reports/baseline_results.json, and produces confusion matrix plots.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents.taxonomy import INTENT_NAMES
from src.intents.labeler import label_dataframe
from src.intents.baseline import MajorityIntentClassifier
from src.intents.classifier import TfidfIntentClassifier, EmbeddingIntentClassifier
from src.retrieval.index import VectorIndex
from src.retrieval.retrieve import HistoricalRetriever
from src.generation.reply_generator import GroundedReplyGenerator
from src.escalation.policy import EscalationPolicy
from src.agent import AppleSupportAgent
from src.evaluation.evaluate import (
    evaluate_intent_classifiers,
    evaluate_retrieval_system,
    evaluate_agent_escalation,
)
from src.evaluation.judge import SupportResponseJudge
from src.evaluation.metrics import compute_judge_human_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_and_save_confusion_matrix(cm_data: list, labels: list, output_path: str):
    """Plot and save confusion matrix visualization."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[l.replace("_", "\n") for l in labels],
            yticklabels=[l.replace("_", "\n") for l in labels],
        )
        plt.title("Confusion Matrix: Embedding Intent Classifier (all-MiniLM-L6-v2)", fontsize=14)
        plt.xlabel("Predicted Intent", fontsize=12)
        plt.ylabel("Ground Truth Intent", fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info("Saved confusion matrix plot to %s", output_path)
    except Exception as e:
        logger.warning("Could not generate confusion matrix plot: %s", e)


def run_full_evaluation(
    data_dir: str = "data/processed",
    index_dir: str = "data/processed/retrieval_index",
    reports_dir: str = "reports",
    max_train_samples: int = 15000,
    max_test_samples: int = 3000,
):
    """Execute complete experimental pipeline."""
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load data
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed train/test files not found. Run scripts/prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Label intents
    label_dataframe(train_df, text_column="customer_text", target_column="intent")
    label_dataframe(test_df, text_column="customer_text", target_column="intent")

    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.iloc[:max_train_samples].copy()
    if max_test_samples and len(test_df) > max_test_samples:
        test_df = test_df.iloc[:max_test_samples].copy()

    X_train = train_df["customer_text"].tolist()
    y_train = train_df["intent"].tolist()
    X_test = test_df["customer_text"].tolist()
    y_test = test_df["intent"].tolist()

    logger.info("Training set size: %d | Test set size: %d", len(X_train), len(X_test))

    # 2. Train & Evaluate Intent Classifiers
    # Baseline 1: Majority
    majority_clf = MajorityIntentClassifier()
    majority_clf.fit(y_train)

    # Baseline 2: TF-IDF + Logistic Regression
    logger.info("Fitting TF-IDF + Logistic Regression Baseline...")
    tfidf_clf = TfidfIntentClassifier(max_features=10000, random_state=42)
    tfidf_clf.fit(X_train, y_train)
    tfidf_clf.save(os.path.join(data_dir, "tfidf_classifier.pkl"))

    # Final: Sentence-Transformer + Logistic Regression
    logger.info("Fitting Sentence-Transformer (all-MiniLM-L6-v2) Classifier...")
    embedding_clf = EmbeddingIntentClassifier(model_name="all-MiniLM-L6-v2", random_state=42)
    embedding_clf.fit(X_train, y_train)
    embedding_clf.save(os.path.join(data_dir, "embedding_classifier.pkl"))

    # Evaluate all on identical test set
    models = {
        "baseline_majority": majority_clf,
        "baseline_tfidf_logistic": tfidf_clf,
        "final_embedding_minilm": embedding_clf,
    }

    intent_results = evaluate_intent_classifiers(
        models=models,
        X_test=X_test,
        y_test=y_test,
        labels=INTENT_NAMES,
    )

    # Plot confusion matrix for final classifier
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(
        cm_data=intent_results["final_embedding_minilm"]["confusion_matrix"],
        labels=INTENT_NAMES,
        output_path=cm_path,
    )

    # 3. Evaluate Historical Retrieval System
    logger.info("Evaluating Historical Retrieval System...")
    retrieval_results = {}
    if os.path.exists(index_dir):
        vector_index = VectorIndex.load(index_dir, use_faiss=True)
        retriever = HistoricalRetriever(index=vector_index, model_name="all-MiniLM-L6-v2")
        retrieval_results = evaluate_retrieval_system(
            retriever=retriever,
            test_df=test_df,
            k_list=[1, 3, 5],
            sample_limit=500,
        )
    else:
        logger.warning("Retrieval index not found at %s. Skipping retrieval eval.", index_dir)
        retriever = None

    # 4. Evaluate Escalation Policy
    logger.info("Evaluating Escalation Policy with End-to-End Agent...")
    agent = AppleSupportAgent(
        classifier=embedding_clf,
        retriever=retriever,
        reply_generator=GroundedReplyGenerator(),
        escalation_policy=EscalationPolicy(),
    )
    escalation_results = evaluate_agent_escalation(
        agent=agent,
        test_df=test_df,
        sample_limit=300,
    )

    # 5. LLM Judge & Human Validation Setup
    logger.info("Setting up LLM Judge & Human Validation...")
    judge = SupportResponseJudge()
    judge_eval_samples = []
    sample_queries = test_df.head(50)

    for _, row in sample_queries.iterrows():
        c_text = row["customer_text"]
        retrieved = retriever.retrieve(c_text, top_k=3) if retriever else []
        gen = GroundedReplyGenerator().generate_reply(c_text, row["intent"], retrieved)
        eval_score = judge.judge_single(c_text, retrieved, gen["draft_reply"])
        judge_eval_samples.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": c_text,
            "draft_reply": gen["draft_reply"],
            "judge_result": eval_score,
        })

    # Prepare Human Rating Template for the 50 validation samples
    human_rating_rows = []
    for idx, s in enumerate(judge_eval_samples, start=1):
        human_rating_rows.append({
            "sample_id": idx,
            "customer_tweet_id": s["customer_tweet_id"],
            "customer_text": s["customer_text"],
            "draft_reply": s["draft_reply"],
            "human_correctness_1_to_5": "",
            "human_groundedness_1_to_5": "",
            "human_relevance_1_to_5": "",
            "human_helpfulness_1_to_5": "",
            "human_brand_consistency_1_to_5": "",
            "human_unsupported_claims_bool": "",
            "human_overall_score_1_to_5": "",
            "llm_judge_score": s["judge_result"].get("overall_score"),
        })

    human_template_path = os.path.join(data_dir, "..", "golden", "human_ratings_template.csv")
    pd.DataFrame(human_rating_rows).to_csv(human_template_path, index=False, encoding="utf-8")
    logger.info("Saved 50-sample human validation rating template to %s", human_template_path)

    agreement_summary = compute_judge_human_agreement([], [])

    # Assemble comprehensive results dictionary
    combined_results = {
        "experiment_metadata": {
            "dataset": "Customer Support on Twitter (twcs.csv) - AppleSupport Scope",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "intent_taxonomy_size": len(INTENT_NAMES),
            "random_seed": 42,
        },
        "intent_classification": intent_results,
        "retrieval_performance": retrieval_results,
        "escalation_policy": escalation_results,
        "llm_judge_sample_evaluation": {
            "evaluated_samples": len(judge_eval_samples),
            "judge_mode": judge_eval_samples[0]["judge_result"].get("status", "LLM_API_GROUNDED") if judge_eval_samples else "NONE",
            "average_overall_score": round(float(np.mean([s["judge_result"].get("overall_score", 0.0) for s in judge_eval_samples])), 3) if judge_eval_samples else 0.0,
            "average_groundedness": round(float(np.mean([s["judge_result"].get("groundedness", 0.0) for s in judge_eval_samples])), 3) if judge_eval_samples else 0.0,
            "unsupported_claim_rate": round(float(np.mean([1.0 if s["judge_result"].get("unsupported_claims") else 0.0 for s in judge_eval_samples])), 4) if judge_eval_samples else 0.0,
        },
        "llm_judge_vs_human_agreement": agreement_summary,
    }

    # Save to baseline_results.json
    results_path = os.path.join(reports_dir, "baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    logger.info("Saved full evaluation results to %s", results_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("              FINAL INTENT CLASSIFIER BENCHMARK")
    print("=" * 70)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 70)
    for name, res in intent_results.items():
        print(f"{name:<30} | {res['accuracy']:<10.4f} | {res['macro_f1']:<10.4f} | {res['weighted_f1']:<10.4f}")
    print("=" * 70)

    if retrieval_results:
        print("\n" + "=" * 70)
        print("              HISTORICAL RETRIEVAL BENCHMARK")
        print("=" * 70)
        print(f"Recall@1: {retrieval_results.get('recall@1', 0.0):.4f}")
        print(f"Recall@3: {retrieval_results.get('recall@3', 0.0):.4f}")
        print(f"Recall@5: {retrieval_results.get('recall@5', 0.0):.4f}")
        print(f"Mean Top-1 Cosine Similarity: {retrieval_results.get('mean_top1_similarity', 0.0):.4f}")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("              ESCALATION POLICY PERFORMANCE")
    print("=" * 70)
    print(f"Auto-Handle Rate: {escalation_results['auto_handle_rate'] * 100:.2f}%")
    print(f"Escalation Rate:  {escalation_results['escalation_rate'] * 100:.2f}%")
    print("=" * 70 + "\n")

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete evaluation suite")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--index_dir", type=str, default="data/processed/retrieval_index", help="Index directory")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--train_samples", type=int, default=15000, help="Train samples limit")
    parser.add_argument("--test_samples", type=int, default=3000, help="Test samples limit")

    args = parser.parse_args()
    run_full_evaluation(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        reports_dir=args.reports_dir,
        max_train_samples=args.train_samples,
        max_test_samples=args.test_samples,
    )
