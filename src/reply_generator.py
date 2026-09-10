"""Grounded Reply Generation Module.

Drafts customer support replies strictly grounded in retrieved historical AppleSupport
troubleshooting precedents, avoiding hallucinated policies or promises.
"""

from typing import List, Dict, Any, Optional
from src.generation.reply_generator import GroundedReplyGenerator, SYSTEM_PROMPT


class ReplyGenerator:
    """Grounded draft reply generator for AppleSupport inquiries."""

    def __init__(
        self,
        provider: str = "mock",
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
    ):
        self.generator = GroundedReplyGenerator(
            openai_api_key=openai_api_key,
            model_name=model_name,
        )

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        evidence: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Generate a grounded reply using retrieved historical precedents."""
        # Adapt evidence dict to generator format
        generator_examples = []
        for e in evidence:
            generator_examples.append({
                "customer_text": e.get("historical_query", ""),
                "agent_text": e.get("historical_reply", ""),
                "similarity_score": e.get("similarity", 0.0),
            })

        return self.generator.generate(
            customer_message=customer_message,
            intent=predicted_intent,
            retrieved_examples=generator_examples,
        )


__all__ = ["ReplyGenerator", "GroundedReplyGenerator", "SYSTEM_PROMPT"]
