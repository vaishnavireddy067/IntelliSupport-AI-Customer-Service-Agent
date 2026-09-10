"""LLM-as-a-Judge evaluation rubric and scoring engine.

Evaluates generated support replies across 6 dimensions:
1. Correctness (1-5)
2. Groundedness (1-5)
3. Relevance (1-5)
4. Helpfulness (1-5)
5. Brand Consistency (1-5)
6. Unsupported Claims (Binary: True/False)

Includes human rating rubric template and agreement evaluation.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an expert QA auditor for Apple Customer Support.
Evaluate the following AI-generated draft response given the Customer Message and the Retrieved Historical Evidence.

Evaluation Rubric (Score each dimension 1 to 5):
1. CORRECTNESS (1-5): Are the troubleshooting steps, settings paths, and technical claims accurate for Apple devices?
2. GROUNDEDNESS (1-5): Is the response strictly derived from the retrieved evidence, without hallucinating unsupported steps?
3. RELEVANCE (1-5): Does the response directly address the specific issue mentioned by the customer?
4. HELPFULNESS (1-5): Is the response actionable, clear, and likely to assist the customer?
5. BRAND_CONSISTENCY (1-5): Does the response match Apple's signature supportive, professional tone?
6. UNSUPPORTED_CLAIMS (true/false): Does the response make false promises (e.g. free refunds, warranty replacements, fake feature dates) not justified by the evidence?

Output ONLY a JSON object with this exact schema:
{
  "correctness": <int 1-5>,
  "groundedness": <int 1-5>,
  "relevance": <int 1-5>,
  "helpfulness": <int 1-5>,
  "brand_consistency": <int 1-5>,
  "unsupported_claims": <true/false>,
  "overall_score": <float 1.0-5.0>,
  "rationale": "<brief 1-sentence justification>"
}
"""


class SupportResponseJudge:
    """Judge engine for customer support replies."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def judge_single(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        draft_reply: str,
    ) -> Dict[str, Any]:
        """Judge a single draft reply against retrieved evidence."""
        if not self.api_key:
            # Deterministic heuristic / pending flag if API key is not provided
            return self._heuristic_judge(customer_message, retrieved_evidence, draft_reply)

        try:
            import urllib.request

            evidence_text = "\n".join([
                f"- Hist Agent: {ex.get('agent_text', '')}"
                for ex in retrieved_evidence[:3]
            ])

            user_prompt = f"""Customer Message: {customer_message}

Retrieved Historical Evidence:
{evidence_text}

Draft Reply to Evaluate:
{draft_reply}
"""
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JUDGE_RUBRIC_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            logger.warning("LLM Judge evaluation failed: %s. Using heuristic evaluation.", e)
            return self._heuristic_judge(customer_message, retrieved_evidence, draft_reply)

    def _heuristic_judge(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        draft_reply: str,
    ) -> Dict[str, Any]:
        """Rule-based heuristic estimation when LLM judge API key is not present."""
        reply_lower = draft_reply.lower()

        # Check unsupported claims
        unsupported = any(w in reply_lower for w in ["refund processed", "free replacement guaranteed", "credited your account"])

        # Check brand consistency
        has_brand_tone = any(w in reply_lower for w in ["help", "let's", "settings", "check", "reach out", "we'd like"])
        brand_score = 4.5 if has_brand_tone else 3.5

        # Check groundedness: token overlap with retrieved evidence
        evidence_text = " ".join([ex.get("agent_text", "").lower() for ex in retrieved_evidence])
        overlap_tokens = set(re.findall(r"\w+", reply_lower)) & set(re.findall(r"\w+", evidence_text))
        groundedness_score = 5.0 if len(overlap_tokens) >= 5 else 3.5

        relevance_score = 4.0 if len(draft_reply) > 20 else 2.5
        helpfulness_score = 4.0
        correctness_score = 4.5 if not unsupported else 2.0

        overall = (correctness_score + groundedness_score + relevance_score + helpfulness_score + brand_score) / 5.0

        return {
            "status": "HEURISTIC_ESTIMATE (PENDING API CREDENTIAL FOR LLM JUDGE)",
            "correctness": round(correctness_score, 1),
            "groundedness": round(groundedness_score, 1),
            "relevance": round(relevance_score, 1),
            "helpfulness": round(helpfulness_score, 1),
            "brand_consistency": round(brand_score, 1),
            "unsupported_claims": unsupported,
            "overall_score": round(overall, 2),
            "rationale": "Evaluated via token grounding and safety constraint heuristic.",
        }
