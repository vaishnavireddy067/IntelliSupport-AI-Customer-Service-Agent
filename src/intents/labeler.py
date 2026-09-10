"""Deterministic intent labeler implementing the taxonomy inclusion/exclusion rules.

Applies high-precision lexical and regex pattern matching derived directly from
src/intents/taxonomy.py to assign empirical intent labels across the dataset.
"""

import re
from typing import Optional, Tuple
from src.intents.taxonomy import INTENT_NAMES

# High-precision patterns based on taxonomy inclusion criteria
PATTERNS = {
    "battery_power": [
        r"\b(battery|batteries|drain|draining|charge|charging|charger|shut(?:ting)? (?:down|off)|power down)\b",
        r"\b(overheat|overheating|battery life|battery health)\b",
    ],
    "os_update_bug": [
        r"\b(ios\s*\d+|high sierra|update|updated|updating|software update|os update)\b.*?\b(bug|glitch|lag|lagging|freeze|freezing|crash|slow|stuck|broke|ruined)\b",
        r"\b(ever since|after) (the )?(latest |new )?(ios|update|high sierra)\b",
        r"\b(verification failed|unable to verify update)\b",
    ],
    "screen_display": [
        r"\b(screen|display|touchscreen|touch screen|lcd|oled|black screen)\b",
        r"\b(unresponsive|lines on|ghost touch|glitching screen|cracked screen|3d touch)\b",
    ],
    "apple_id_account": [
        r"\b(apple\s*id|icloud|itunes account)\b",
        r"\b(password|passcode|locked out|security questions?|verification code|two-?factor|2fa|activation lock)\b",
    ],
    "connectivity_network": [
        r"\b(wi-?fi|wifi|bluetooth|airdrop|hotspot|cellular|lte|4g|sim card|no service|searching\.\.\.)\b",
        r"\b(connect(?:ing|ion)?|pair(?:ing)?|disconnect(?:ing)?|drop(?:ping)?)\b.*?\b(wifi|bluetooth|network|carrier)\b",
    ],
    "app_store_issues": [
        r"\b(app store|apps? crashing|apps? crash|wont open|won't open|waiting\.\.\.|unable to download)\b",
        r"\b(downloading|installing) (apps?|updates?)\b",
    ],
    "billing_subscription": [
        r"\b(billing|charged|charge|refund|subscription|invoice|receipt|payment|credit card|apple pay|in-app purchase|cancelled subscription)\b",
        r"\b(money back|unauthorized charge|double charged|charged twice)\b",
    ],
    "media_services": [
        r"\b(apple music|itunes|playlist|playlists|songs?|album|podcast|podcasts)\b",
        r"\b(camera|mic|microphone|speaker|sound|audio|volume|photos?|camera roll)\b",
    ],
    "store_hardware_service": [
        r"\b(genius bar|apple store|appointment|repair|replace|replacement|trade-?in|warranty|applecare|apple care|shipping|tracking|order status)\b",
    ],
}


def label_intent_from_text(text: str) -> str:
    """Classify a message into one of the 10 taxonomy intents using priority rules."""
    if not text or not isinstance(text, str):
        return "other_general"

    text_lower = text.lower()

    # Match scores
    scores = {intent: 0 for intent in INTENT_NAMES}

    for intent, pattern_list in PATTERNS.items():
        for pat in pattern_list:
            if re.search(pat, text_lower):
                scores[intent] += 1

    # Find highest scoring intent
    best_intent, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score > 0:
        return best_intent

    return "other_general"


def label_dataframe(df, text_column: str = "customer_text", target_column: str = "intent"):
    """Assign intent labels to a pandas DataFrame."""
    df[target_column] = df[text_column].fillna("").apply(label_intent_from_text)
    return df
