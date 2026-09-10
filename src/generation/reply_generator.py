"""Grounded support reply generator.

Generates customer support replies strictly grounded in retrieved historical AppleSupport
resolutions. Supports LLM API (OpenAI/Anthropic) when configured, with a 100% deterministic,
zero-cost retrieval fallback when no API key is provided.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an official Apple Support AI specialist.
Your task is to draft a helpful, professional, and concise customer support tweet/reply grounded strictly in the provided historical AppleSupport resolutions.

RULES:
1. Rely ONLY on the verified troubleshooting steps and guidelines shown in the historical examples.
2. DO NOT invent Apple policies, warranty terms, replacement guarantees, repair costs, refund promises, or release dates.
3. If the retrieved evidence does not provide an actionable solution for the customer's specific problem, clearly advise them to connect via DM or visit getsupport.apple.com.
4. Maintain Apple's signature supportive, polite tone (e.g. "We're here to help", "Let's take a look").
5. Keep the response concise (under 280 characters if possible, standard Twitter format).
6. Do NOT claim you have personally performed any account actions or modifications.
"""


def _generate_with_openai(
    customer_message: str,
    intent: str,
    retrieved_examples: List[Dict[str, Any]],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """Call OpenAI API if available."""
    try:
        import urllib.request

        evidence_text = ""
        for i, ex in enumerate(retrieved_examples[:3], start=1):
            evidence_text += f"\nExample {i}:\nCustomer: {ex.get('customer_text', '')}\nAppleSupport: {ex.get('agent_text', '')}\n"

        user_content = f"""Customer Issue: {customer_message}
Identified Category: {intent}

Historical AppleSupport Evidence:
{evidence_text}

Draft an official grounded AppleSupport response:"""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 120,
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("OpenAI API call failed or timed out: %s. Falling back to deterministic retrieval.", e)
        return None


def _format_deterministic_fallback(
    customer_message: str,
    intent: str,
    retrieved_examples: List[Dict[str, Any]],
) -> str:
    """Deterministic fallback: selects and adapts the best historical AppleSupport reply."""
    if not retrieved_examples:
        return (
            "We're here to help. To take a closer look into this with you, "
            "please connect with our team directly: https://getsupport.apple.com"
        )

    best = retrieved_examples[0]
    agent_text = best.get("agent_text", "").strip()

    # Clean out customer handle mentions like @115854 or @username
    cleaned_reply = re.sub(r"^(?:@\w+\s*)+", "", agent_text).strip()

    # Ensure polite opening if missing
    if not any(cleaned_reply.lower().startswith(p) for p in ["we're here", "we'd like", "thanks for", "let's", "hello", "hi"]):
        cleaned_reply = f"We're here to help. {cleaned_reply}"

    return cleaned_reply


class GroundedReplyGenerator:
    """Orchestrates grounded reply drafting using LLM or deterministic fallback."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a grounded reply for the customer query."""
        # 1. Try LLM generation if API key is present
        if self.api_key:
            llm_draft = _generate_with_openai(
                customer_message=customer_message,
                intent=predicted_intent,
                retrieved_examples=retrieved_examples,
                api_key=self.api_key,
                model=self.model,
            )
            if llm_draft:
                return {
                    "draft_reply": llm_draft,
                    "generation_mode": "llm_grounded",
                    "model": self.model,
                }

        # 2. Deterministic zero-cost retrieval fallback
        fallback_draft = _format_deterministic_fallback(
            customer_message=customer_message,
            intent=predicted_intent,
            retrieved_examples=retrieved_examples,
        )
        return {
            "draft_reply": fallback_draft,
            "generation_mode": "retrieval_fallback",
            "model": "historical_resolution_grounding",
        }
