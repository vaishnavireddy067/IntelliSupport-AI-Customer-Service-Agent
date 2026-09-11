"""LLM-as-a-Judge Response Quality Auditor — Public Re-export Module.

This module re-exports the full judge implementation from src/evaluation/judge.py
for convenience when running standalone evaluation scripts from the project root.

Full implementation lives at: src/evaluation/judge.py (145 lines)
  - SupportResponseJudge class: LLM API call (gpt-4o-mini) + heuristic offline fallback
  - JUDGE_RUBRIC_PROMPT: 6-dimension evaluation rubric (Correctness, Groundedness,
    Relevance, Helpfulness, Brand Consistency, Unsupported Claims)
  - Offline mode: rule-based heuristic estimator when no OPENAI_API_KEY is set

Rubric dimensions (each scored 1-5):
  1. Correctness       — Are troubleshooting steps accurate for Apple devices?
  2. Groundedness      — Is the response derived from retrieved evidence only?
  3. Relevance         — Does the response address the customer's specific issue?
  4. Helpfulness       — Is the response actionable and clear?
  5. Brand Consistency — Matches Apple's supportive, professional tone?
  6. Unsupported Claims (boolean) — Any hallucinated policies or false promises?
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.judge import SupportResponseJudge, JUDGE_RUBRIC_PROMPT

__all__ = ["SupportResponseJudge", "JUDGE_RUBRIC_PROMPT"]
