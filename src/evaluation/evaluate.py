"""Core evaluation pipeline orchestrating Intent, Retrieval, and Escalation evaluation.

Evaluates all models on the identical test set with zero fabricated data.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_retrieval_recall_at_k,
    compute_escalation_metrics,
)

logger = logging.getLogger(__name__)


def evaluate_intent_classifiers(
    models: Dict[str, Any],
    X_test: List[str],
    y_test: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """Evaluate multiple classifiers on the identical test set."""
    results = {}
    for model_name, model in models.items():
        logger.info("Evaluating classifier: %s...", model_name)
        y_pred = model.predict(X_test)
        metrics = compute_intent_metrics(y_true=y_test, y_pred=y_pred, labels=labels)
        results[model_name] = metrics
        logger.info(
            "  %s -> Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
            model_name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )
    return results


def evaluate_retrieval_system(
    retriever: Any,
    test_df: pd.DataFrame,
    k_list: List[int] = [1, 3, 5],
    sample_limit: int = 500,
) -> Dict[str, Any]:
    """Evaluate vector retrieval accuracy and Recall@K on test customer inquiries."""
    logger.info("Evaluating retrieval system on %d test inquiries...", min(len(test_df), sample_limit))
    eval_df = test_df.head(sample_limit)

    queries = eval_df["customer_text"].tolist()
    target_agent_ids = eval_df["agent_tweet_id"].astype(str).tolist()

    retrieved_batches = retriever.retrieve_batch(queries, top_k=max(k_list))

    recall_metrics = compute_retrieval_recall_at_k(
        actual_target_ids=target_agent_ids,
        retrieved_results=retrieved_batches,
        k_list=k_list,
    )

    # Calculate average similarity score of top-1 match
    top1_sims = [
        batch[0]["similarity_score"] if len(batch) > 0 else 0.0
        for batch in retrieved_batches
    ]
    mean_top1_similarity = float(np.mean(top1_sims)) if top1_sims else 0.0

    recall_metrics["mean_top1_similarity"] = round(mean_top1_similarity, 4)
    recall_metrics["evaluated_queries"] = len(queries)

    logger.info("Retrieval metrics: %s", recall_metrics)
    return recall_metrics


def evaluate_agent_escalation(
    agent: Any,
    test_df: pd.DataFrame,
    sample_limit: int = 200,
) -> Dict[str, Any]:
    """Evaluate end-to-end agent decision distribution and escalation safety."""
    logger.info("Evaluating escalation policy on %d sample queries...", min(len(test_df), sample_limit))
    eval_df = test_df.head(sample_limit)

    decisions = []
    reasons = []

    for _, row in eval_df.iterrows():
        res = agent.process_message(row["customer_text"])
        decisions.append(res["decision"])
        reasons.append(res["reason"])

    total = len(decisions)
    auto_count = decisions.count("AUTO_HANDLE")
    escalate_count = decisions.count("ESCALATE_TO_HUMAN")

    return {
        "total_evaluated": total,
        "auto_handle_count": auto_count,
        "escalate_to_human_count": escalate_count,
        "auto_handle_rate": round(auto_count / total, 4) if total > 0 else 0.0,
        "escalation_rate": round(escalate_count / total, 4) if total > 0 else 0.0,
        "sample_escalation_reasons": reasons[:5],
    }
