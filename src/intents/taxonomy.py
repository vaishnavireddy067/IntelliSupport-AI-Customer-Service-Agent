"""Empirical intent taxonomy for AppleSupport customer conversations.

Constructed from direct frequency and n-gram analysis of customer inquiries in twcs.csv.
Defines 10 compact, mutually exclusive intents with formal inclusion/exclusion criteria.
"""

from typing import List, Dict, Any

INTENTS: List[Dict[str, Any]] = [
    {
        "name": "battery_power",
        "description": "Issues concerning battery drain, battery health degradation, charging failures, or unexpected device shutdowns.",
        "examples": [
            "My iPhone 7 battery is draining from 100% to 20% in two hours.",
            "Phone won't charge past 80% when plugged into the original charger.",
            "My phone keeps shutting off randomly at 30% battery remaining.",
            "Battery health dropped drastically overnight.",
        ],
        "inclusion_criteria": (
            "Mentions battery life, battery drain, rapid percentage drops, phone dying, "
            "overheating while charging, charger not recognized, or unexpected shutdowns."
        ),
        "exclusion_criteria": (
            "Phone totally bricked and not turning on after a software update (os_update_bug) "
            "or physical port damage (device_hardware)."
        ),
    },
    {
        "name": "os_update_bug",
        "description": "Bugs, freezes, UI glitches, or performance drops immediately following an iOS or macOS software update.",
        "examples": [
            "Ever since updating to iOS 11.1 my phone has been lagging terribly.",
            "Updated to High Sierra and now the system freezes on boot.",
            "The new iOS update broke my keyboard autocorrect with a weird symbol.",
            "Cannot complete software update, keeps saying verification failed.",
        ],
        "inclusion_criteria": (
            "Explicitly connects an issue to an iOS or macOS update, software installation failure, "
            "slowdown post-update, or system software freeze."
        ),
        "exclusion_criteria": (
            "Third-party app crashes (app_store_issues) or battery drain where update is not the primary symptom."
        ),
    },
    {
        "name": "screen_display",
        "description": "Display problems, unresponsive touch screens, black screens, lines on screen, or 3D Touch glitches.",
        "examples": [
            "My iPhone screen is completely black but it still vibrates when someone calls.",
            "Touch screen is unresponsive on the top half of my screen.",
            "There are green vertical lines running down my iPhone X screen.",
            "Ghost touching on my screen without me pressing anything.",
        ],
        "inclusion_criteria": (
            "Display rendering issues, touch responsiveness failure, backlight failure, "
            "cracked screen inquiries, or OLED/LCD anomalies."
        ),
        "exclusion_criteria": (
            "Lock screen software notifications issues (os_update_bug) or device audio issues."
        ),
    },
    {
        "name": "apple_id_account",
        "description": "Account management, Apple ID security lockouts, password resets, two-factor authentication, or iCloud login.",
        "examples": [
            "My Apple ID has been locked for security reasons and I can't unlock it.",
            "Forgot my iCloud password and the recovery phone number is old.",
            "Not receiving the two-factor authentication verification code on my trusted device.",
            "Unable to sign out of iCloud on my iPad.",
        ],
        "inclusion_criteria": (
            "Authentication, password resets, account verification, Apple ID lockouts, "
            "two-factor SMS codes, or activation lock recovery."
        ),
        "exclusion_criteria": (
            "Subscriptions or card billing linked to Apple ID (billing_subscription)."
        ),
    },
    {
        "name": "connectivity_network",
        "description": "Network connectivity issues including Wi-Fi drops, Bluetooth pairing, cellular data, or AirPods connection.",
        "examples": [
            "My iPhone keeps disconnecting from home Wi-Fi every few minutes.",
            "Bluetooth won't detect my car stereo or AirPods.",
            "Says No Service or Searching after toggling airplane mode.",
            "Cannot make or receive cellular calls on my carrier network.",
        ],
        "inclusion_criteria": (
            "Wi-Fi, Bluetooth, LTE/cellular data, cellular reception, SIM card errors, "
            "AirDrop failures, or personal hotspot disconnects."
        ),
        "exclusion_criteria": (
            "Music streaming pauses due to Apple Music catalog error (media_services)."
        ),
    },
    {
        "name": "app_store_issues",
        "description": "Third-party or native app crashes, App Store download/update loops, or app compatibility errors.",
        "examples": [
            "Instagram and Twitter keep crashing immediately upon opening.",
            "App Store says waiting on all pending app downloads.",
            "Unable to download apps, keeps prompting for Apple ID password repeatedly.",
            "WhatsApp won't open after the latest app update.",
        ],
        "inclusion_criteria": (
            "App crashing, App Store download failures, 'waiting' app icons, "
            "app permissions errors, or inability to launch specific applications."
        ),
        "exclusion_criteria": (
            "System OS freezing across the entire phone (os_update_bug) or billing disputes for purchases (billing_subscription)."
        ),
    },
    {
        "name": "billing_subscription",
        "description": "Charges, in-app purchases, Apple Music/iCloud subscriptions, unauthorized debits, or refund requests.",
        "examples": [
            "I was charged $9.99 for a subscription I cancelled last week.",
            "How do I request a refund for an accidental in-app purchase my child made?",
            "Apple charged my credit card twice for my iCloud storage upgrade.",
            "Want to cancel my Apple Music renewal before it charges.",
        ],
        "inclusion_criteria": (
            "Credit card charges, invoices, subscription renewal/cancellation, refund requests, "
            "payment method declined, or billing disputes."
        ),
        "exclusion_criteria": (
            "Free app download failures without monetary charge (app_store_issues)."
        ),
    },
    {
        "name": "media_services",
        "description": "Apple Music streaming, iTunes sync, photo library sync, podcast playback, or camera/microphone issues.",
        "examples": [
            "My Apple Music playlists disappeared from my library.",
            "Songs won't download for offline listening in Apple Music.",
            "Camera app is just showing a black screen when I switch to rear camera.",
            "Microphone sounds muffled during voice memos and phone calls.",
        ],
        "inclusion_criteria": (
            "Music playback, iTunes sync, Photos sync, iCloud Drive files, camera sensor glitches, "
            "speaker audio, or headphone jack/lightning audio issues."
        ),
        "exclusion_criteria": (
            "Bluetooth headphone disconnection (connectivity_network) or App Store purchasing (billing_subscription)."
        ),
    },
    {
        "name": "store_hardware_service",
        "description": "Physical repair, Apple Store appointments, Genius Bar bookings, warranty status, or order shipments.",
        "examples": [
            "How do I book an appointment at the Genius Bar to replace my screen?",
            "Can I trade in my iPhone 6s at the local Apple Store?",
            "When will my pre-ordered iPhone X be delivered?",
            "Is my iPad still covered under AppleCare+ for accidental damage?",
        ],
        "inclusion_criteria": (
            "Genius Bar, Apple Store visits, AppleCare+ coverage check, repair quotes, "
            "trade-in program, or online order delivery tracking."
        ),
        "exclusion_criteria": (
            "Self-troubleshooting a software or setting issue at home."
        ),
    },
    {
        "name": "other_general",
        "description": "General praise, complaints without specific technical symptoms, conversational greetings, or out-of-scope inquiries.",
        "examples": [
            "Thank you AppleSupport for the quick help earlier today!",
            "Why is customer service so difficult to reach?",
            "Hello, is someone there to answer a question?",
            "Does Apple have any jobs available in Seattle?",
        ],
        "inclusion_criteria": (
            "Queries lacking specific technical symptoms, feedback, greetings, or questions outside typical support categories."
        ),
        "exclusion_criteria": (
            "Any query presenting a concrete technical symptom or actionable support issue."
        ),
    },
]

INTENT_NAMES = [intent["name"] for intent in INTENTS]
INTENT_LOOKUP = {intent["name"]: intent for intent in INTENTS}


def get_intent_names() -> List[str]:
    """Return ordered list of canonical intent names."""
    return INTENT_NAMES.copy()


def get_intent_details(name: str) -> Dict[str, Any]:
    """Return dictionary of intent metadata."""
    return INTENT_LOOKUP.get(name, INTENT_LOOKUP["other_general"])
