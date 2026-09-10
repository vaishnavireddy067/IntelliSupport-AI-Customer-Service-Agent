"""Escalation Policy Engine Module.

Evaluates multi-signal risk factors to deterministically decide between AUTO_HANDLE
and ESCALATE with an explicit human-readable reason.
"""

from typing import List, Dict, Any, Optional
from src.escalation.policy import EscalationPolicy, HIGH_RISK_PATTERNS, REPEATED_FAILURE_PATTERNS


class EscalationDecisionEngine:
    """Evaluates whether to auto-handle or escalate an incoming support query."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        retrieval_threshold: float = 0.65,
    ):
        self.policy = EscalationPolicy(
            min_intent_confidence=confidence_threshold,
            min_retrieval_similarity=retrieval_threshold,
        )

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Return decision ('AUTO_HANDLE' or 'ESCALATE') and rationale."""
        # Convert evidence format if necessary
        retrieved_examples = []
        for e in evidence:
            retrieved_examples.append({
                "customer_text": e.get("historical_query", ""),
                "agent_text": e.get("historical_reply", ""),
                "similarity_score": e.get("similarity", 0.0),
            })

        res = self.policy.evaluate(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            retrieved_examples=retrieved_examples,
        )

        decision = "ESCALATE" if "ESCALATE" in res.get("decision", "") else "AUTO_HANDLE"
        return {
            "decision": decision,
            "reason": res.get("reason", "Routine support request."),
        }


__all__ = ["EscalationDecisionEngine", "EscalationPolicy", "HIGH_RISK_PATTERNS"]
