"""Explicit rule-based escalation policy for customer support.

Evaluates multiple deterministic risk signals:
- Intent classification confidence
- Retrieval similarity score
- Security / high-risk keywords (stolen, fraud, legal)
- Financial / billing disputes requiring human authorization
- Repeated unresolved customer frustration ("tried everything", "3rd time", "no response")
- Actions requiring human authorization (unlocking Apple ID, issuing refunds)
"""

import re
from typing import List, Dict, Any, Tuple


# Keywords that immediately indicate high-risk, security, or legal exposure
HIGH_RISK_PATTERNS = [
    r"\b(stolen|theft|lost phone|robbed)\b",
    r"\b(police|fbi|crime|scam|fraud|unauthorized charges?)\b",
    r"\b(lawsuit|lawyer|sue|legal action|court|attorney)\b",
    r"\b(death|deceased|passed away|estate)\b",
    r"\b(hacked|compromised|blackmail)\b",
]

# Frustration / repeated failure indicators
REPEATED_FAILURE_PATTERNS = [
    r"\b(tried everything|already tried that|doesn'?t help)\b",
    r"\b(called support|been on hold|waiting for \d+ hours?)\b",
    r"\b(second time|third time|4th time|again and again)\b",
    r"\b(terrible service|useless|unacceptable|no one (is )?replying)\b",
    r"\b(speak to a human|real person|manager|supervisor)\b",
]

# Sensitive intents that inherently require human review or account-level authorization
SENSITIVE_INTENTS = {
    "billing_subscription",  # Refund authorizations and disputed charges
    "apple_id_account",      # Security lockouts and 2FA recovery
}


class EscalationPolicy:
    """Deterministic escalation decision engine."""

    def __init__(
        self,
        min_intent_confidence: float = 0.40,
        min_retrieval_similarity: float = 0.45,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_examples: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Evaluate whether to AUTO_HANDLE or ESCALATE_TO_HUMAN with an explicit explanation."""
        text_lower = customer_message.lower()

        # Signal 1: High-risk, security, legal, or fraud indicators
        for pattern in HIGH_RISK_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": f"High-risk security or legal term detected: '{match.group(0)}'. Human specialist required.",
                }

        # Signal 2: Repeated failure, escalation request, or severe customer frustration
        for pattern in REPEATED_FAILURE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "reason": f"Customer frustration or repeated unresolved issue detected: '{match.group(0)}'.",
                }

        # Signal 3: No retrieved examples or historical evidence
        if not retrieved_examples:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": "No historical support evidence found to ground an automated reply.",
            }

        # Signal 4: Low retrieval similarity
        top_similarity = float(retrieved_examples[0].get("similarity_score", 0.0))
        if top_similarity < self.min_retrieval_similarity:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": (
                    f"Low retrieval similarity ({top_similarity:.3f} < threshold {self.min_retrieval_similarity:.2f}). "
                    "Insufficient verified historical precedents."
                ),
            }

        # Signal 5: Low intent classification confidence
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": (
                    f"Ambiguous intent: confidence ({intent_confidence:.3f}) below safe handling threshold "
                    f"({self.min_intent_confidence:.2f})."
                ),
            }

        # Signal 6: Sensitive intents with financial or identity consequences
        if predicted_intent in SENSITIVE_INTENTS and top_similarity < 0.65:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": (
                    f"Sensitive category '{predicted_intent}' with moderate historical match ({top_similarity:.3f}). "
                    "Escalated for user privacy and authorization safety."
                ),
            }

        # If all checks pass, safe for automated handling
        return {
            "decision": "AUTO_HANDLE",
            "reason": (
                f"High intent confidence ({intent_confidence:.3f}) and strong historical match ({top_similarity:.3f}) "
                "for routine self-service resolution."
            ),
        }
