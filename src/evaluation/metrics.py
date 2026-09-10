"""Comprehensive evaluation metrics calculation.

Computes:
1. Classification metrics: Accuracy, Macro F1, Weighted F1, Per-intent F1, Confusion Matrix
2. Retrieval metrics: Recall@1, Recall@3, Recall@5
3. Escalation metrics: Accuracy, False Auto-Handle Rate, False Escalation Rate
4. LLM Judge vs. Human validation: Spearman correlation, Pearson, Exact & Adjacent Agreement
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "weighted_precision": round(float(weighted_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "weighted_recall": round(float(weighted_rec), 4),
        "per_intent_report": report,
        "confusion_matrix": cm,
        "labels": labels if labels else sorted(list(set(y_true))),
    }


def compute_retrieval_recall_at_k(
    actual_target_ids: List[str],
    retrieved_results: List[List[Dict[str, Any]]],
    k_list: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Compute Recall@K: whether the true resolution tweet ID is in the top-K retrieved.

    actual_target_ids: List of true agent_tweet_id (or customer_tweet_id) for each test query.
    retrieved_results: List of top-K retrieved candidate dictionaries for each test query.
    """
    total = len(actual_target_ids)
    if total == 0:
        return {f"recall@{k}": 0.0 for k in k_list}

    recalls = {k: 0 for k in k_list}

    for target_id, retrieved in zip(actual_target_ids, retrieved_results):
        candidate_ids = [r.get("agent_tweet_id") or r.get("customer_tweet_id") for r in retrieved]
        for k in k_list:
            if target_id in candidate_ids[:k]:
                recalls[k] += 1

    return {f"recall@{k}": round(float(recalls[k] / total), 4) for k in k_list}


def compute_escalation_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Compute escalation safety and efficiency metrics.

    Decisions: 'AUTO_HANDLE' vs 'ESCALATE_TO_HUMAN'.
    False Auto-Handle Rate = False Negatives / Total Actual Escalations (CRITICAL SAFETY RISK)
    False Escalation Rate = False Positives / Total Actual Auto-Handles (EFFICIENCY LOSS)
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "false_auto_handle_rate": 0.0, "false_escalation_rate": 0.0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / total

    actual_escalate = sum(1 for t in y_true if t == "ESCALATE_TO_HUMAN")
    actual_auto = sum(1 for t in y_true if t == "AUTO_HANDLE")

    # False auto-handle: actual was ESCALATE_TO_HUMAN, but system erroneously predicted AUTO_HANDLE
    false_auto_handle_count = sum(
        1 for t, p in zip(y_true, y_pred) if t == "ESCALATE_TO_HUMAN" and p == "AUTO_HANDLE"
    )

    # False escalation: actual was AUTO_HANDLE, but system unnecessarily escalated
    false_escalate_count = sum(
        1 for t, p in zip(y_true, y_pred) if t == "AUTO_HANDLE" and p == "ESCALATE_TO_HUMAN"
    )

    false_auto_handle_rate = (false_auto_handle_count / actual_escalate) if actual_escalate > 0 else 0.0
    false_escalation_rate = (false_escalate_count / actual_auto) if actual_auto > 0 else 0.0

    return {
        "escalation_accuracy": round(float(acc), 4),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "false_escalation_rate": round(float(false_escalation_rate), 4),
        "total_samples": total,
        "actual_escalate_count": actual_escalate,
        "actual_auto_count": actual_auto,
    }


def compute_judge_human_agreement(
    human_scores: List[float],
    judge_scores: List[float],
) -> Dict[str, Any]:
    """Compute agreement and correlation between human ratings and LLM judge ratings."""
    if len(human_scores) != len(judge_scores) or len(human_scores) == 0:
        return {
            "status": "PENDING HUMAN INPUT",
            "sample_size": 0,
            "spearman_correlation": None,
            "pearson_correlation": None,
            "exact_agreement_rate": None,
            "adjacent_agreement_rate": None,
        }

    h = np.array(human_scores, dtype=float)
    j = np.array(judge_scores, dtype=float)
    n = len(h)

    # Exact agreement
    exact = float(np.mean(h == j))
    # Adjacent agreement (within 1 point on 1-5 scale)
    adjacent = float(np.mean(np.abs(h - j) <= 1.0))

    # Pearson
    if np.std(h) > 0 and np.std(j) > 0:
        pearson = float(np.corrcoef(h, j)[0, 1])
    else:
        pearson = 0.0

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman, p_val = spearmanr(h, j)

    return {
        "status": "COMPLETED",
        "sample_size": n,
        "spearman_correlation": round(float(spearman), 4),
        "spearman_p_value": round(float(p_val), 6),
        "pearson_correlation": round(float(pearson), 4),
        "exact_agreement_rate": round(exact, 4),
        "adjacent_agreement_rate": round(adjacent, 4),
    }
