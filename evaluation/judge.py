"""LLM-as-a-Judge Response Quality Auditor Module.

Rubric:
1. Correctness (1-5)
2. Groundedness (1-5)
3. Relevance (1-5)
4. Helpfulness (1-5)
5. Brand Consistency (1-5)
6. Unsupported Claims (Boolean)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.judge import SupportResponseJudge, JUDGE_RUBRIC_PROMPT

__all__ = ["SupportResponseJudge", "JUDGE_RUBRIC_PROMPT"]
