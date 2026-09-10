"""Text preprocessing and normalization for customer support conversations.

Preserves realistic customer characteristics (slang, typos, emotional punctuation,
short messages, emojis) while removing unusable noise, HTML artifacts, and redundant
Twitter username handles.
"""

import re
import html
from typing import Optional, Dict, Any


def clean_customer_text(text: Optional[str]) -> str:
    """Normalize customer tweet text while preserving realistic noise.

    - Decodes HTML entities (e.g. &amp;, &gt;, &#39;)
    - Strips leading mention handles (e.g., '@AppleSupport', '@115854') so model
      focuses on the customer issue, but retains internal text.
    - Preserves case, emojis, typos, punctuation, slang, and exclamation marks.
    - Normalizes excessive whitespace and carriage returns.
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities
    cleaned = html.unescape(text)

    # Remove leading mentions like @AppleSupport @12345
    cleaned = re.sub(r"^(?:@\w+\s*)+", "", cleaned)

    # Normalize weird Unicode zero-width or variation selectors
    cleaned = re.sub(r"[\ufe00-\ufe0f\u200b-\u200d]", "", cleaned)

    # Normalize excessive internal whitespace/newlines to single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_agent_text(text: Optional[str]) -> str:
    """Clean brand support response text while preserving links and structure."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = html.unescape(text)
    # Remove leading customer handle mention like @115854
    cleaned = re.sub(r"^(?:@\w+\s*)+", "", cleaned)
    cleaned = re.sub(r"[\ufe00-\ufe0f\u200b-\u200d]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_usable_conversation_pair(customer_text: str, agent_text: str) -> bool:
    """Determine whether a customer-agent pair contains usable semantic content.

    Discards:
    - Empty or whitespace-only messages
    - Customer messages containing ONLY a URL or media link without words
    - Extremely short messages with zero alphabetical characters
    """
    if not customer_text or not agent_text:
        return False

    # Check if customer text is purely a URL link (e.g., just "https://t.co/xyz")
    url_stripped = re.sub(r"https?://\S+", "", customer_text).strip()
    if len(url_stripped) < 3 and not any(c.isalnum() for c in url_stripped):
        return False

    # Ensure there is at least one word character in both
    if not re.search(r"[a-zA-Z0-9]", customer_text):
        return False
    if not re.search(r"[a-zA-Z0-9]", agent_text):
        return False

    return True


def preprocess_pair(pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Preprocess a single conversation record.

    Returns the cleaned dictionary, or None if the record is not usable.
    """
    cleaned_cust = clean_customer_text(pair.get("customer_text", ""))
    cleaned_agent = clean_agent_text(pair.get("agent_text", ""))

    if not is_usable_conversation_pair(cleaned_cust, cleaned_agent):
        return None

    return {
        "customer_tweet_id": pair.get("customer_tweet_id", ""),
        "customer_author_id": pair.get("customer_author_id", ""),
        "customer_text": cleaned_cust,
        "customer_created_at": pair.get("customer_created_at", ""),
        "agent_tweet_id": pair.get("agent_tweet_id", ""),
        "agent_text": cleaned_agent,
        "agent_created_at": pair.get("agent_created_at", ""),
    }
