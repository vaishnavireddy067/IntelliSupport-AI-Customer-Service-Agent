"""Evaluation Metrics Module.

Computes:
1. Intent Classification Metrics: Accuracy, Macro F1, Weighted F1, Precision, Recall
2. Escalation Metrics: Accuracy, False Auto-Handle Rate, False Escalation Rate
3. Retrieval Metrics: Recall@1, Recall@3, Recall@5
4. Judge vs Human Agreement: Exact Agreement %, Adjacent Agreement %, Spearman & Pearson Correlation
"""

import os
import sys

# Ensure root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_retrieval_recall_at_k,
    compute_escalation_metrics,
    compute_judge_human_agreement,
)

__all__ = [
    "compute_intent_metrics",
    "compute_retrieval_recall_at_k",
    "compute_escalation_metrics",
    "compute_judge_human_agreement",
]
