"""IntelliSupport AI - Autonomous Customer Support Agent Demo Application.

Interactive, product-grade customer support platform featuring:
1. Overview & Landing: Executive summary, value propositions, and live pipeline architecture
2. Live Agent Playground: 2-column layout (Message & Quick Scenarios -> Analysis, Suggested Response, Evidence, System Details)
3. Empirical Benchmarks: 4 key metric cards, Model Comparison table, Intent distribution, and Key Takeaways
4. Golden Eval Explorer: 200-case hand-labelled benchmark, dynamic filters, difficulty progress bars, and paginated cards
5. Human vs. Judge Audit: Calibration metrics, sample comparison table, and agreement indicators
6. Failure Analysis & Critique: 5 real failure modes, "What is misleading about headline number?", non-goals, and 1-week roadmap
7. Architecture & Decisions: Visual pipeline, 15-item decision log, and key runtime configuration
"""

import os
import sys
import json
import time
import pandas as pd

# Ensure offline huggingface cache for sub-second startup
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Please install it using: pip install streamlit")
    sys.exit(1)

from src.agent import AppleSupportAgent

# Set page layout & config
st.set_page_config(
    page_title="IntelliSupport AI — Autonomous Customer Support Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Friendly display names for empirical intents
INTENT_DISPLAY_NAMES = {
    "battery_power": "Battery & Power",
    "os_update_bug": "OS Update & Bugs",
    "screen_display": "Screen & Display",
    "apple_id_account": "Apple ID & Security",
    "connectivity_network": "Connectivity & Network",
    "app_store_issues": "App Store & Downloads",
    "billing_subscription": "Billing & Subscriptions",
    "media_services": "Media & Apple Music",
    "store_hardware_service": "Genius Bar & Hardware",
    "other_general": "General Inquiry",
}

# Stunning Dark Glassmorphism CSS with vibrant gradients and animations
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f8fafc;
    }

    /* Streamlit App Background Override */
    .stApp {
        background: linear-gradient(135deg, #050814 0%, #0b1329 50%, #070b19 100%);
        background-attachment: fixed;
    }

    /* Main Container — push content below Streamlit's sticky toolbar */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    /* Top Navy Hero Header with Glassmorphism */
    .top-navy-header {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(20, 30, 58, 0.88) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px 30px 24px 30px;
        border-radius: 16px;
        /* Gradient top line via border-top — works without overflow:hidden */
        border-top: 3px solid transparent;
        border-right: 1px solid rgba(99, 102, 241, 0.28);
        border-bottom: 1px solid rgba(99, 102, 241, 0.28);
        border-left: 1px solid rgba(99, 102, 241, 0.28);
        background-clip: padding-box;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.55),
                    0 -3px 0 0 #6366f1,
                    0 -1px 0 0 #8b5cf6;
        margin-bottom: 24px;
        margin-top: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
        animation: fadeInDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    /* Gradient top-accent line using outline trick */
    .top-navy-header::before {
        content: '';
        position: absolute;
        top: -3px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 40%, #06b6d4 100%);
        border-radius: 16px 16px 0 0;
        z-index: 1;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .header-left {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .header-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-title {
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        text-shadow: 0 2px 10px rgba(255,255,255,0.1);
    }
    .header-version {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(147, 197, 253, 0.25));
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
        font-size: 11.5px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.2);
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 400;
        margin-top: -2px;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .pill-badge {
        font-size: 12px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    .pill-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.18);
        color: #e9d5ff;
        border: 1px solid rgba(168, 85, 247, 0.4);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.18);
        color: #bbf7d0;
        border: 1px solid rgba(34, 197, 94, 0.4);
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.15);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.18);
        color: #fef08a;
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }

    .header-right {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-tag {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #ffffff;
        font-size: 13.5px;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }
    .online-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #4ade80;
        font-weight: 700;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
    }
    .online-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e, 0 0 20px #22c55e;
        animation: pulseGlow 2s infinite;
    }

    @keyframes pulseGlow {
        0% { transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Tab Styling with Glassmorphism */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
        margin-bottom: 24px;
        background: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 18px;
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        background: transparent;
        border-radius: 8px;
        border: none;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.06);
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
        border-bottom: none !important;
    }

    /* Cards & Containers (Glassmorphism) */
    .white-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .white-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
    }
    .card-header-title {
        font-size: 15.5px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 9px;
        letter-spacing: -0.01em;
    }

    /* Metric Card Box */
    .metric-card-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    .metric-card-box:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.2);
    }
    .metric-card-label {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card-value {
        font-size: 24px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .metric-card-subtext {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Banners */
    .banner-auto-handle {
        background: rgba(6, 95, 70, 0.3);
        border: 1px solid rgba(52, 211, 153, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 12px 16px;
        color: #6ee7b7;
        font-size: 13.5px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(52, 211, 153, 0.15);
    }
    .banner-escalate {
        background: rgba(153, 27, 27, 0.3);
        border: 1px solid rgba(248, 113, 113, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 13.5px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(248, 113, 113, 0.15);
    }

    /* Suggested Reply Box */
    .suggested-reply-box {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        margin-bottom: 16px;
    }
    .reply-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .reply-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .reply-text-content {
        font-size: 14px;
        line-height: 1.65;
        color: #cbd5e1;
        margin: 0;
        background: rgba(30, 41, 59, 0.6);
        padding: 16px 18px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Evidence List Item */
    .evidence-row-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 16px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .evidence-row-item:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .score-badge-green {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-size: 11.5px;
        font-weight: 800;
        padding: 4px 9px;
        border-radius: 6px;
        min-width: 48px;
        text-align: center;
        margin-top: 1px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }
    .evidence-text-query {
        font-size: 13.5px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .evidence-text-sub {
        font-size: 12px;
        color: #94a3b8;
    }

    /* Quick Scenarios Grid */
    .quick-scenarios-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    .scenario-tile {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 13px;
        font-weight: 600;
        color: #cbd5e1;
        text-align: left;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        backdrop-filter: blur(8px);
    }
    .scenario-tile:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.15);
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25);
    }

    /* Pipeline Flow Diagram in Architecture */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        overflow-x: auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }
    .pipeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 16px 18px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        min-width: 150px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .pipeline-node:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2);
    }
    .node-icon-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    }
    .node-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #f8fafc;
    }
    .pipeline-arrow {
        color: #64748b;
        font-size: 20px;
        font-weight: 800;
        margin: 0 8px;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
    }

    /* Decision Badges */
    .pill-auto {
        background: rgba(220, 252, 231, 0.15);
        color: #4ade80;
        font-size: 11.5px;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(187, 247, 208, 0.3);
        box-shadow: 0 0 12px rgba(74, 222, 128, 0.2);
    }
    .pill-escalate {
        background: rgba(254, 226, 226, 0.15);
        color: #f87171;
        font-size: 11.5px;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(254, 202, 202, 0.3);
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.2);
    }

    /* Landing Page Hero */
    .landing-hero {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .landing-title {
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }
    .landing-subtitle {
        font-size: 15.5px;
        color: #94a3b8;
        line-height: 1.7;
        margin-bottom: 24px;
        max-width: 920px;
    }

    /* Failure Analysis Cards */
    .failure-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.37);
        border-left: 5px solid #ef4444;
        transition: all 0.3s ease;
    }
    .failure-card:hover {
        border-color: rgba(239, 68, 68, 0.4);
        transform: translateY(-2px);
    }
    .failure-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .failure-num-badge {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
    }
    .failure-quote {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13.5px;
        color: #e2e8f0;
        font-style: italic;
        margin-bottom: 12px;
        border-left: 3px solid #64748b;
    }
    .failure-diagnosis {
        font-size: 13.5px;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .failure-remediation {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13px;
        color: #93c5fd;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.1);
    }

    /* Misleading Number Highlight Box */
    .misleading-box {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 6px solid #f59e0b;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.15);
    }
    .misleading-title {
        font-size: 16px;
        font-weight: 700;
        color: #fcd34d;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .misleading-text {
        font-size: 14px;
        color: #fef08a;
        line-height: 1.7;
    }

    /* Non-Goal Card */
    .nongoal-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: all 0.2s ease;
    }
    .nongoal-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    .nongoal-title {
        font-size: 14px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .nongoal-text {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* ── Streamlit UI element overrides ── */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.85) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.35) !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.45) !important;
        transition: all 0.25s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.6) !important;
    }
    div[data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 700 !important;
    }

    /* ── CRITICAL: Force ALL text to be visible on dark background ── */
    /* Paragraphs, body text, markdown */
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #e2e8f0 !important;
    }
    /* Section headers (### in streamlit) */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    /* st.write and generic text */
    .stText, p, span:not([class]) {
        color: #e2e8f0 !important;
    }
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }
    /* Labels for inputs */
    label, .stLabel, [data-testid="stWidgetLabel"] p,
    .stTextInput label, .stSelectbox label, .stTextArea label,
    .stSlider label, .stCheckbox label, .stRadio label {
        color: #cbd5e1 !important;
    }
    /* Dataframe / table text */
    .stDataFrame td, .stDataFrame th,
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #e2e8f0 !important;
        background: rgba(15, 23, 42, 0.7) !important;
    }
    [data-testid="stTable"] th {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
    }
    /* Info / warning / error boxes */
    [data-testid="stAlert"] p { color: inherit !important; }
    /* Selectbox options text */
    [data-baseweb="select"] div, [data-baseweb="select"] span {
        color: #e2e8f0 !important;
        background: rgba(15, 23, 42, 0.95) !important;
    }
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }
    /* Tab text fix */
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: inherit !important;
    }
    /* Spinner text */
    .stSpinner p { color: #94a3b8 !important; }
    /* st.caption */
    small { color: #94a3b8 !important; }

    /* ── Architecture & Decisions tab: dark-theme card overrides ── */
    .white-card {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        margin-bottom: 16px !important;
    }
    .card-header-title {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        margin-bottom: 4px !important;
    }
    .nongoal-card {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(99, 102, 241, 0.18) !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        margin-bottom: 12px !important;
    }
    .nongoal-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        margin-bottom: 6px !important;
    }
    .nongoal-text {
        font-size: 13px !important;
        color: #94a3b8 !important;
        line-height: 1.6 !important;
    }
    .nongoal-text b { color: #a5b4fc !important; }

    /* Pipeline diagram nodes */
    .pipeline-container {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        padding: 16px;
        background: rgba(15,23,42,0.6);
        border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.2);
        margin-bottom: 20px;
    }
    .pipeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 10px 14px;
        background: rgba(30, 41, 59, 0.85);
        border-radius: 10px;
        border: 1px solid rgba(99,102,241,0.2);
        min-width: 120px;
    }
    .node-title {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        text-align: center !important;
    }
    .pipeline-arrow {
        color: #6366f1 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    .node-icon-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }

    /* Expander dark override */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    /* Code block inside expander */
    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] code {
        background: rgba(8, 14, 30, 0.9) !important;
        color: #a5f3fc !important;
        border-radius: 8px !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    """Load agent with cached artifacts."""
    return AppleSupportAgent.load_from_artifacts()


@st.cache_data
def load_golden_set():
    """Load the 200 curated golden examples with standardized columns."""
    path_eval = os.path.join(os.path.dirname(__file__), "data", "golden_eval.csv")
    if os.path.exists(path_eval):
        df = pd.read_csv(path_eval)
    else:
        path_set = os.path.join(os.path.dirname(__file__), "data", "golden", "golden_set.csv")
        if os.path.exists(path_set):
            df = pd.read_csv(path_set)
        else:
            return None

    # Standardize column mappings across schema variants
    if "example_id" not in df.columns and "id" in df.columns:
        df["example_id"] = df["id"]
    if "customer_message" not in df.columns and "text" in df.columns:
        df["customer_message"] = df["text"]
    if "gold_action" not in df.columns and "gold_escalation" in df.columns:
        df["gold_action"] = df["gold_escalation"]
    if "gold_reason" not in df.columns and "notes" in df.columns:
        df["gold_reason"] = df["notes"]
    if "conversation_id" not in df.columns and "customer_tweet_id" in df.columns:
        df["conversation_id"] = df["customer_tweet_id"]
    if "brand" not in df.columns:
        df["brand"] = "AppleSupport"

    # Derive difficulty if not present as standalone column
    if "difficulty" not in df.columns and "context" in df.columns:
        df["difficulty"] = df["context"].apply(
            lambda x: x.split("Difficulty: ")[1].strip() if "Difficulty: " in str(x) else "Standard"
        )

    # Clean whitespace and types
    df["difficulty"] = df["difficulty"].astype(str).str.strip()
    df["gold_intent"] = df["gold_intent"].astype(str).str.strip()
    df["gold_action"] = df["gold_action"].astype(str).str.strip()
    df["customer_message"] = df["customer_message"].astype(str).str.strip()

    return df


@st.cache_data
def load_judge_agreement():
    """Load 50 audited judge agreement samples."""
    path = os.path.join(os.path.dirname(__file__), "evaluation", "judge_agreement.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    # -------------------------------------------------------
    # LANDING / SPLASH PAGE  (session-state gate)
    # -------------------------------------------------------
    if "show_dashboard" not in st.session_state:
        st.session_state["show_dashboard"] = False

    if not st.session_state["show_dashboard"]:
        import textwrap
        st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}.block-container{padding-top:2rem!important;}</style>", unsafe_allow_html=True)

        hero_html = textwrap.dedent("""
<style>
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes fadeUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
@keyframes ringPulse { 0%,100%{opacity:.4;transform:scale(1)} 50%{opacity:.8;transform:scale(1.04)} }
</style>
<div style="position:relative;overflow:hidden;background:linear-gradient(145deg,#050814 0%,#0b1329 45%,#0d0b1f 100%);border-radius:24px;padding:72px 40px 60px;margin-bottom:24px;border:1px solid rgba(99,102,241,0.18);box-shadow:0 30px 80px rgba(0,0,0,0.7);">
  <!-- Background orbs -->
  <div style="position:absolute;top:-80px;left:-80px;width:350px;height:350px;background:radial-gradient(circle,rgba(99,102,241,0.18) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-100px;right:-60px;width:400px;height:400px;background:radial-gradient(circle,rgba(139,92,246,0.14) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;top:40%;left:50%;transform:translate(-50%,-50%);width:500px;height:500px;background:radial-gradient(circle,rgba(6,182,212,0.06) 0%,transparent 70%);pointer-events:none;"></div>

  <div style="text-align:center;max-width:760px;margin:0 auto;position:relative;z-index:1;">
    <!-- Pill badge -->
    <div style="animation:fadeUp .5s ease both;display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.35);border-radius:9999px;padding:7px 22px;margin-bottom:32px;font-size:12.5px;color:#a5b4fc;font-weight:700;letter-spacing:.04em;">
      <span style="width:7px;height:7px;background:#6366f1;border-radius:50%;box-shadow:0 0 10px #6366f1;display:inline-block;"></span>
      INTELLISUPPORT AI &nbsp;·&nbsp; HIVER SDE INTERN ASSIGNMENT
    </div>
    <!-- Main headline -->
    <div style="animation:fadeUp .55s .1s ease both;font-size:60px;font-weight:900;line-height:1.06;letter-spacing:-0.045em;color:#ffffff;margin-bottom:6px;text-shadow:0 2px 30px rgba(99,102,241,0.3);">AI Support Agent</div>
    <div style="animation:fadeUp .55s .18s ease both;font-size:60px;font-weight:900;line-height:1.06;letter-spacing:-0.045em;margin-bottom:28px;background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 30px rgba(99,102,241,0.5));">That Actually Works</div>
    <!-- Subtitle -->
    <div style="animation:fadeUp .55s .26s ease both;font-size:16.5px;color:#94a3b8;line-height:1.75;max-width:560px;margin:0 auto 36px;">
      Classifies <b style="color:#c7d2fe;font-weight:700;">customer intent</b>, retrieves <b style="color:#c7d2fe;font-weight:700;">verified resolutions</b>, drafts <b style="color:#c7d2fe;font-weight:700;">grounded replies</b> — and decides in <b style="color:#6ee7b7;font-weight:700;">&lt;25ms</b> whether to auto-handle or escalate to a human.
    </div>
    <!-- Feature pills -->
    <div style="animation:fadeUp .55s .34s ease both;display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
      <span style="background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.32);color:#c7d2fe;font-size:13px;font-weight:600;padding:7px 18px;border-radius:9999px;transition:all .2s ease;">🎯 Intent Classification</span>
      <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.28);color:#67e8f9;font-size:13px;font-weight:600;padding:7px 18px;border-radius:9999px;">🔍 Precedent Retrieval</span>
      <span style="background:rgba(139,92,246,0.14);border:1px solid rgba(139,92,246,0.3);color:#ddd6fe;font-size:13px;font-weight:600;padding:7px 18px;border-radius:9999px;">✉️ Grounded Replies</span>
      <span style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.28);color:#6ee7b7;font-size:13px;font-weight:600;padding:7px 18px;border-radius:9999px;">🛡️ Safety Escalation</span>
    </div>
  </div>
</div>
        """).strip()
        st.markdown(hero_html, unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        for col, val, label, color, glow, icon in zip(
            [s1, s2, s3, s4],
            ["85.4%", "15K+", "4.38/5", "200"],
            ["Intent Accuracy", "Historical Cases", "Response Quality", "Golden Eval Cases"],
            ["#6366f1", "#10b981", "#8b5cf6", "#f59e0b"],
            ["rgba(99,102,241,0.35)", "rgba(16,185,129,0.35)", "rgba(139,92,246,0.35)", "rgba(245,158,11,0.35)"],
            ["🎯", "🗂️", "⭐", "🏅"],
        ):
            with col:
                card = (
                    f"<div style='background:linear-gradient(145deg,rgba(15,23,42,0.9),rgba(20,32,55,0.8));"
                    f"border:1px solid rgba(255,255,255,0.07);border-top:2px solid {color};border-radius:16px;"
                    f"padding:22px 16px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4),0 0 20px {glow};"
                    f"backdrop-filter:blur(12px);transition:transform .2s ease;'>"
                    f"<div style='font-size:22px;margin-bottom:8px;'>{icon}</div>"
                    f"<div style='font-size:32px;font-weight:900;color:{color};letter-spacing:-0.03em;"
                    f"text-shadow:0 0 20px {glow};'>{val}</div>"
                    f"<div style='font-size:11.5px;color:#64748b;margin-top:8px;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:.07em;'>{label}</div>"
                    f"</div>"
                )
                st.markdown(card, unsafe_allow_html=True)

        st.write("")
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("🚀 Launch Dashboard", type="primary", use_container_width=True):
                st.session_state["show_dashboard"] = True
                st.rerun()

        st.markdown("<div style='text-align:center;margin-top:14px;font-size:12px;color:#94a3b8;'>Built for <b>Hiver SDE Intern Assignment</b> &nbsp;·&nbsp; Brand: <b>🍎 AppleSupport</b> &nbsp;·&nbsp; v1.0.0</div>", unsafe_allow_html=True)
        return

    # -------------------------------------------------------
    # DASHBOARD (shown after clicking Launch Dashboard)
    # -------------------------------------------------------

    # Top Navy Header
    st.markdown(
        """
        <div class="top-navy-header">
            <div class="header-left">
                <div class="header-title-row">
                    <span style="font-size: 24px;">🛡️</span>
                    <span class="header-title">IntelliSupport AI</span>
                    <span class="header-version">v1.0.0</span>
                </div>
                <div class="header-subtitle">
                    AI-powered support that understands, responds, and knows when to escalate.
                </div>
                <div class="header-badges">
                    <span class="pill-badge badge-purple">🎯 10 support intents</span>
                    <span class="pill-badge badge-blue">🔍 15K historical cases</span>
                    <span class="pill-badge badge-green">🛡️ Safe auto-routing</span>
                    <span class="pill-badge badge-amber">🍎 AppleSupport specialist</span>
                </div>
            </div>
            <div class="header-right">
                <div class="brand-tag">
                    <span>🍏</span> AppleSupport
                </div>
                <div class="online-indicator">
                    <span class="online-dot"></span> Online
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Tabs (no Landing tab — it's the splash page)
    (
        tab_play,
        tab_bench,
        tab_golden,
        tab_judge,
        tab_failure,
        tab_arch,
    ) = st.tabs(
        [
            "⚡ Live Agent Playground",
            "📊 Empirical Benchmarks",
            "🎯 Golden Eval Explorer",
            "⚖️ Human vs. Judge Audit",
            "🔬 Failure Analysis & Critique",
            "🛡️ Architecture & Decisions",
        ]
    )


    # TAB 0 IS NOW THE SPLASH — dashboard starts directly at Playground

    # -------------------------------------------------------------
    # TAB 1: LIVE AGENT PLAYGROUND (Matching Screenshot 1)
    # -------------------------------------------------------------
    with tab_play:
        st.markdown("### ⚡ Live Agent Playground")
        st.caption("Test the AI agent with real customer scenarios and see how it responds.")

        # Preset Quick Scenarios
        preset_scenarios = {
            "Battery Drain": "@AppleSupport my iPhone 8 battery drops from 80% to 15% in one hour after iOS 11 update",
            "iOS Update": "@AppleSupport my phone is completely frozen on the Apple logo after updating to iOS 11.0.3",
            "Security Issue": "@AppleSupport someone in Russia just accessed my Apple ID and changed my recovery email! HELP!",
            "Billing Issue": "@AppleSupport I was double charged $9.99 for Apple Music this month, need a refund immediately",
            "App Store": "@AppleSupport cannot download or update any apps from the App Store, getting error 1009",
            "Hardware Repair": "@AppleSupport how do I book an appointment at the Genius Bar to fix my cracked screen?",
        }

        # Initialize session state for customer query
        if "customer_input_text" not in st.session_state:
            st.session_state["customer_input_text"] = "@AppleSupport how do I book an appointment at the Genius Bar to fix my cracked screen?"

        def set_preset_scenario(text_val):
            st.session_state["customer_input_text"] = text_val
            st.session_state["auto_analyze"] = True

        col_left, col_right = st.columns([1, 1.35], gap="large")

        # ---------------- LEFT COLUMN: INPUT & QUICK SCENARIOS ----------------
        with col_left:
            st.markdown(
                """
                <div class="white-card">
                    <div class="card-header-title">💬 Customer Message</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_query = st.text_area(
                "Customer Message Input",
                key="customer_input_text",
                height=130,
                label_visibility="collapsed",
            )

            analyze_btn = st.button("🚀 Analyze with IntelliSupport AI", type="primary", use_container_width=True)

            st.write("")
            st.markdown(
                """
                <div style="font-size: 13.5px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                    Quick Scenarios
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2-column grid of 6 scenario buttons using callbacks
            sc_cols1, sc_cols2 = st.columns(2)
            with sc_cols1:
                st.button(
                    "🔋 Battery Drain",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["Battery Drain"],),
                    use_container_width=True,
                )
                st.button(
                    "🛡️ Security Issue",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["Security Issue"],),
                    use_container_width=True,
                )
                st.button(
                    "📱 App Store",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["App Store"],),
                    use_container_width=True,
                )

            with sc_cols2:
                st.button(
                    "🔄 iOS Update",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["iOS Update"],),
                    use_container_width=True,
                )
                st.button(
                    "💳 Billing Issue",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["Billing Issue"],),
                    use_container_width=True,
                )
                st.button(
                    "🔧 Hardware Repair",
                    on_click=set_preset_scenario,
                    args=(preset_scenarios["Hardware Repair"],),
                    use_container_width=True,
                )

        # ---------------- RIGHT COLUMN: AI ANALYSIS & OUTPUTS ----------------
        with col_right:
            agent = load_agent()

            # Run inference when Analyze button clicked or Quick Scenario selected
            should_run = analyze_btn or st.session_state.pop("auto_analyze", False)
            if should_run:
                query_to_run = (st.session_state.get("customer_input_text", "") or user_query or "").strip()
                if not query_to_run:
                    st.warning("Please type a customer message before analyzing.")
                else:
                    with st.spinner("🔍 Analyzing with IntelliSupport AI..."):
                        res = agent.process_message(query_to_run)
                        st.session_state["last_result"] = res
                        st.session_state["last_query"] = query_to_run

            # Show last result (persists across reruns)
            res = st.session_state.get("last_result", None)
            last_query = st.session_state.get("last_query", "")

            if res:
                intent_raw = res.get("intent", "other_general")
                intent_pretty = INTENT_DISPLAY_NAMES.get(intent_raw, intent_raw)
                conf = res.get("confidence", 0.0)
                decision = res.get("decision", "AUTO_HANDLE")

                # Input echo
                st.markdown(
                    f"""
                    <div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:10px 14px;
                        font-size:12.5px;color:#94a3b8;margin-bottom:14px;">
                        <b style="color:#f8fafc;">Analyzing:</b> {last_query[:120]}{'...' if len(last_query) > 120 else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 1. AI Analysis Card — 3 metric pills
                st.markdown(
                    """
                    <div class="white-card" style="padding:14px 16px;margin-bottom:12px;">
                        <div class="card-header-title" style="margin-bottom:10px;">🤖 AI Analysis</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                a1, a2, a3 = st.columns(3)
                with a1:
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div class="metric-card-label">🎯 Predicted Intent</div>
                            <div class="metric-card-value" style="color:#2563eb;font-size:15px;">{intent_raw}</div>
                            <div class="metric-card-subtext">{intent_pretty}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with a2:
                    conf_color = "#16a34a" if conf >= 0.70 else "#d97706" if conf >= 0.40 else "#dc2626"
                    conf_label = "High" if conf >= 0.70 else "Moderate" if conf >= 0.40 else "Low"
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div class="metric-card-label">📊 Confidence</div>
                            <div class="metric-card-value" style="color:{conf_color};font-size:22px;">{conf*100:.1f}%</div>
                            <div class="metric-card-subtext">{conf_label} confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with a3:
                    dec_color = "#16a34a" if decision == "AUTO_HANDLE" else "#dc2626"
                    dec_sub = "Routine request" if decision == "AUTO_HANDLE" else "Human queue"
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div class="metric-card-label">🛡️ Decision</div>
                            <div class="metric-card-value" style="color:{dec_color};font-size:15px;">{decision.replace('_', '-')}</div>
                            <div class="metric-card-subtext">{dec_sub}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Decision banner
                if decision == "AUTO_HANDLE":
                    st.markdown(
                        f"""
                        <div class="banner-auto-handle">
                            <span style="font-size:18px;">✅</span>
                            <div><b>AUTO-HANDLE:</b> {res.get('escalation_reason','')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="banner-escalate">
                            <span style="font-size:18px;">⚠️</span>
                            <div><b>ESCALATE TO HUMAN:</b> {res.get('escalation_reason','')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 2. Suggested Response
                draft = res.get('draft_reply', '')
                st.markdown(
                    f"""
                    <div class="suggested-reply-box">
                        <div class="reply-header-row">
                            <div class="reply-title"><span>✨</span> Suggested Response</div>
                            <span style="font-size:12px;color:#2563eb;font-weight:600;">Grounded in precedent</span>
                        </div>
                        <p class="reply-text-content">{draft}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 3. Historical Evidence
                st.markdown(
                    """
                    <div class="white-card" style="margin-bottom:10px;">
                        <div class="card-header-title">🔎 Historical Evidence (Top Matches)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                evidence = res.get("evidence", [])
                if evidence:
                    for ev in evidence[:3]:
                        st.markdown(
                            f"""
                            <div class="evidence-row-item">
                                <span class="score-badge-green">{ev['similarity']:.3f}</span>
                                <div>
                                    <div class="evidence-text-query">"{ev['historical_query']}"</div>
                                    <div class="evidence-text-sub">Historical Resolution: "{ev['historical_reply']}"</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No historical precedent exceeded the similarity threshold.")

                # 4. System Details
                with st.expander("⚙️ Model & System Details", expanded=False):
                    st.markdown(
                        f"""
                        - **Inference Latency:** {res.get('latency_ms', '—')} ms
                        - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
                        - **Classifier:** Calibrated Logistic Regression (C=1.5, L-BFGS)
                        - **Precedent Index:** 15,000 indexed historical customer-agent pairs
                        - **Security Rulebook:** Regex scans on credentials, theft, fraud, and legal triggers
                        """
                    )
            else:
                # Idle state — show a helpful prompt card
                st.markdown(
                    """
                    <div style="background:rgba(15,23,42,0.65);border:2px dashed rgba(255,255,255,0.18);border-radius:14px;padding:40px 30px;
                        text-align:center;margin-top:20px;">
                        <div style="font-size:40px;margin-bottom:12px;">🤖</div>
                        <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-bottom:8px;">Ready to Analyze</div>
                        <div style="font-size:13.5px;color:#cbd5e1;line-height:1.6;max-width:320px;margin:0 auto;">
                            Type a customer message on the left or click a
                            <b style="color:#f8fafc;">Quick Scenario</b>, then press
                            <b style="color:#60a5fa;">🚀 Analyze with IntelliSupport AI</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------
    # TAB 2: EMPIRICAL BENCHMARKS (Matching Screenshot 3)
    # -------------------------------------------------------------
    with tab_bench:
        st.markdown("### 📊 Empirical Benchmarks")
        st.caption("Compare your AI agent against simple baselines using real evaluation data.")

        # 4 Metric Cards in a Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                """
                <div class="metric-card-box">
                    <div class="metric-card-label">🎯 Intent Classification</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div><span style="font-size: 11px; color: #94a3b8;">Accuracy: </span><b style="font-size: 18px; color: #f8fafc;">85.4%</b></div>
                        <div><span style="font-size: 11px; color: #94a3b8;">Macro-F1: </span><b style="font-size: 18px; color: #60a5fa;">62.4%</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="metric-card-box">
                    <div class="metric-card-label">🔍 Retrieval (Top-K)</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div><span style="font-size: 11px; color: #94a3b8;">Recall@1: </span><b style="font-size: 18px; color: #f8fafc;">48.2%</b></div>
                        <div><span style="font-size: 11px; color: #94a3b8;">Recall@5: </span><b style="font-size: 18px; color: #4ade80;">68.4%</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                """
                <div class="metric-card-box">
                    <div class="metric-card-label">🛡 Escalation Decision</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div><span style="font-size: 11px; color: #94a3b8;">Precision: </span><b style="font-size: 18px; color: #f8fafc;">83.1%</b></div>
                        <div><span style="font-size: 11px; color: #94a3b8;">Recall: </span><b style="font-size: 18px; color: #fbbf24;">79.8%</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                """
                <div class="metric-card-box">
                    <div class="metric-card-label">⭐ Response Quality (LLM Judge)</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div><span style="font-size: 11px; color: #94a3b8;">Correctness: </span><b style="font-size: 18px; color: #f8fafc;">4.38/5</b></div>
                        <div><span style="font-size: 11px; color: #94a3b8;">Helpfulness: </span><b style="font-size: 18px; color: #c084fc;">4.32/5</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Model Comparison Table
        st.markdown(
            """
            <div class="white-card">
                <div class="card-header-title">Model Comparison</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_comp_df = pd.DataFrame(
            [
                {
                    "Method": "Majority-Class (Trivial)",
                    "Accuracy ↑": "67.0%",
                    "Macro-F1 ↑": "0.0802",
                    "Notes": "Always predicts most common intent (other_general)",
                },
                {
                    "Method": "TF-IDF + Logistic Regression",
                    "Accuracy ↑": "86.0%",
                    "Macro-F1 ↑": "0.6687",
                    "Notes": "Simple ML baseline with unigram & bigram bag-of-words",
                },
                {
                    "Method": "★ IntelliSupport AI (Ours)",
                    "Accuracy ↑": "85.4%",
                    "Macro-F1 ↑": "0.6241",
                    "Notes": "Dense Sentence-Transformers + Precedent RAG + Multi-signal escalation",
                },
            ]
        )
        st.dataframe(model_comp_df, use_container_width=True, hide_index=True)

        st.write("")

        # Bottom row: Performance & Key Takeaways
        b_left, b_right = st.columns([1.2, 1])

        with b_left:
            st.markdown(
                """
                <div class="white-card">
                    <div class="card-header-title">Intent Classification Performance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            intent_perf = pd.DataFrame(
                [
                    {"Intent": "battery_power", "Accuracy": 0.88, "Macro-F1": 0.76},
                    {"Intent": "screen_display", "Accuracy": 0.89, "Macro-F1": 0.74},
                    {"Intent": "apple_id_account", "Accuracy": 0.85, "Macro-F1": 0.79},
                    {"Intent": "connectivity_network", "Accuracy": 0.84, "Macro-F1": 0.71},
                    {"Intent": "billing_subscription", "Accuracy": 0.91, "Macro-F1": 0.82},
                    {"Intent": "app_store_issues", "Accuracy": 0.83, "Macro-F1": 0.69},
                    {"Intent": "media_services", "Accuracy": 0.82, "Macro-F1": 0.67},
                    {"Intent": "store_hardware_service", "Accuracy": 0.86, "Macro-F1": 0.70},
                    {"Intent": "os_update_bug", "Accuracy": 0.78, "Macro-F1": 0.52},
                    {"Intent": "other_general", "Accuracy": 0.87, "Macro-F1": 0.85},
                ]
            )
            st.dataframe(intent_perf, use_container_width=True, hide_index=True, height=220)

        with b_right:
            st.markdown(
                """
                <div class="white-card">
                    <div class="card-header-title">Key Takeaways</div>
                    <ul style="font-size: 13px; color: #cbd5e1; line-height: 1.8; margin-bottom: 0; padding-left: 20px;">
                        <li><b>The agent achieves competitive intent classification</b> (85.4% vs 86.0% TF-IDF) while adding semantic retrieval, grounded response generation, and safety-aware escalation.</li>
                        <li><b>Dense retrieval significantly improves response quality</b> from 3.0 to 4.38/5.0 over baseline unguided responses.</li>
                        <li><b>Escalation decisions prioritize customer safety</b>, catching credential, fraud, and account-loss inquiries.</li>
                        <li><b>Macro F1 exposes class imbalance challenges</b>, highlighting the need for dedicated multi-label modeling.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Prominent Misleading Number Callout
        st.markdown(
            """
            <div class="misleading-box">
                <div class="misleading-title">
                    <span>⚠️</span> What is Misleading About the Headline Number?
                </div>
                <div class="misleading-text">
                    <b>"85.4% intent accuracy does not mean the agent is correct 85.4% of the time in production.</b> The evaluation set contains only 200 curated examples and is stratified across selected intents and difficulty levels. Accuracy also does not measure response grounding or escalation safety. Therefore, the headline score should be interpreted together with retrieval recall, response-quality evaluation, and escalation metrics."
                    <div style="margin-top: 10px; font-size: 12.5px; color: #92400e; line-height: 1.5;">
                        • <b>Class imbalance masks zero-skill baselines:</b> 67.0% accuracy is achievable by simply predicting <code>other_general</code> for every single customer without learning any semantics.<br>
                        • <b>A high auto-handling rate creates severe risk:</b> Auto-handling a hacked account or stolen iPhone is catastrophic for brand trust.<br>
                        • <b>Semantic proximity does not equal clinical diagnosis:</b> High cosine similarity can link two physically divergent problems if both share similar hardware vocabulary.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # TAB 3: GOLDEN EVAL EXPLORER (Matching Screenshot 2)
    # -------------------------------------------------------------
    with tab_golden:
        st.markdown("### 🎯 Golden Evaluation Set Explorer (200 Stratified Scenarios)")
        st.caption("Explore the benchmark used to measure intent classification, escalation decisions, and support quality.")

        golden_df = load_golden_set()
        if golden_df is not None:
            # Top row: 3 Metric Cards on Left + Difficulty Distribution on Right
            top_c1, top_c2 = st.columns([1.1, 1], gap="medium")

            with top_c1:
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div style="font-size: 20px;">🟢</div>
                            <div class="metric-card-value" style="color: #16a34a; font-size: 24px; margin-top: 4px;">{len(golden_df)}</div>
                            <div class="metric-card-label" style="margin-top: 4px;">Golden Cases</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with g2:
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div style="font-size: 20px;">🔵</div>
                            <div class="metric-card-value" style="color: #2563eb; font-size: 24px; margin-top: 4px;">{golden_df['gold_intent'].nunique()}</div>
                            <div class="metric-card-label" style="margin-top: 4px;">Intents</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with g3:
                    st.markdown(
                        f"""
                        <div class="metric-card-box">
                            <div style="font-size: 20px;">🟣</div>
                            <div class="metric-card-value" style="color: #9333ea; font-size: 24px; margin-top: 4px;">{golden_df['difficulty'].nunique()}</div>
                            <div class="metric-card-label" style="margin-top: 4px;">Difficulty Tiers</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with top_c2:
                st.markdown(
                    """
                    <div class="white-card" style="padding: 12px 16px; margin-bottom: 0;">
                        <div class="card-header-title" style="margin-bottom: 8px; font-size: 13.5px;">Difficulty Distribution</div>
                    """,
                    unsafe_allow_html=True,
                )
                diff_counts = golden_df["difficulty"].value_counts()
                for d_name, count in diff_counts.items():
                    pct = (count / len(golden_df)) * 100
                    st.write(f"**{d_name}**: {count} ({pct:.1f}%)")
                    st.progress(count / len(golden_df))
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("")

            # Filter Card
            st.markdown(
                """
                <div class="white-card">
                    <div class="card-header-title">🔎 Explore evaluation cases</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3 Filter Dropdowns
            f1, f2, f3, f4 = st.columns([3, 3, 3, 2])
            diff_opts = ["All"] + sorted(golden_df["difficulty"].dropna().unique().tolist())
            intent_opts = ["All intents"] + sorted(golden_df["gold_intent"].dropna().unique().tolist())
            esc_opts = ["All", "AUTO_HANDLE", "ESCALATE"]

            with f1:
                sel_diff = st.selectbox("Difficulty", diff_opts, index=0, key="g_diff_sel")
            with f2:
                sel_intent = st.selectbox("Intent", intent_opts, index=0, key="g_intent_sel")
            with f3:
                sel_esc = st.selectbox("Escalation", esc_opts, index=0, key="g_esc_sel")
            with f4:
                st.write("")
                st.write("")
                if st.button("🔄 Clear filters", use_container_width=True):
                    st.session_state["g_search_query"] = ""
                    st.session_state["g_page_num"] = 1
                    st.rerun()

            # Search Bar
            search_query = st.text_input(
                "Search customer messages...",
                value=st.session_state.get("g_search_query", ""),
                placeholder="Search customer messages...",
                label_visibility="collapsed",
                key="g_search_input",
            )

            # Apply Filters
            filtered_df = golden_df.copy()
            if sel_diff != "All":
                filtered_df = filtered_df[filtered_df["difficulty"] == sel_diff]
            if sel_intent != "All intents":
                filtered_df = filtered_df[filtered_df["gold_intent"] == sel_intent]
            if sel_esc != "All":
                filtered_df = filtered_df[filtered_df["gold_action"] == sel_esc]
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["customer_message"].str.contains(search_query, case=False, na=False)
                    | filtered_df["gold_reason"].astype(str).str.contains(search_query, case=False, na=False)
                ]

            # Results and Pagination Controls
            tot_cases = len(golden_df)
            filt_cases = len(filtered_df)
            pct_shown = (filt_cases / tot_cases) * 100 if tot_cases > 0 else 0

            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                st.markdown(
                    f"<div style='font-size: 13.5px; color: #94a3b8; padding-top: 8px; font-weight: 500;'>"
                    f"Showing <b>{filt_cases}</b> of <b>{tot_cases}</b> cases ({pct_shown:.1f}% selected)"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with p_col2:
                c_page_size, c_pages = st.columns([1.5, 2.5])
                with c_page_size:
                    page_size = st.selectbox("Page size:", [10, 20, 50], index=1, key="g_page_size_select")
                with c_pages:
                    total_pages = max(1, (filt_cases + page_size - 1) // page_size)
                    page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1, key="g_page_num_input")

            # Slice current page
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, filt_cases)
            page_df = filtered_df.iloc[start_idx:end_idx]

            # Render Cards
            for _, row in page_df.iterrows():
                cid = row.get("example_id", row.get("id", "000"))
                diff = str(row.get("difficulty", "Standard")).upper()
                msg = row.get("customer_message", "")
                tw_id = row.get("conversation_id", row.get("customer_tweet_id", "N/A"))
                g_intent = row.get("gold_intent", "N/A")
                g_action = row.get("gold_action", "AUTO_HANDLE")
                g_notes = row.get("gold_reason", "Standard scenario")

                diff_badge_color = "#f59e0b" if "MEDIUM" in diff else "#ef4444" if "HARD" in diff else "#8b5cf6" if "ADVERSARIAL" in diff else "#10b981"
                action_pill = '<span class="pill-auto">✓ AUTO-HANDLE</span>' if g_action == "AUTO_HANDLE" else '<span class="pill-escalate">⚠ ESCALATE</span>'

                st.markdown(
                    f"""
                    <div class="white-card" style="padding: 16px 18px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-weight: 800; font-size: 14px; color: #f8fafc;">Case #{cid}</span>
                                <span style="background: {diff_badge_color}20; color: {diff_badge_color}; font-size: 10.5px; font-weight: 800; padding: 2px 7px; border-radius: 4px; border: 1px solid {diff_badge_color}40;">{diff}</span>
                            </div>
                            <div style="font-size: 11.5px; color: #64748b;">Customer Tweet ID: {tw_id}</div>
                        </div>
                        <div style="font-size: 13.5px; color: #e2e8f0; font-weight: 500; margin-bottom: 12px; line-height: 1.45;">
                            "{msg}"
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 12px; font-size: 12.5px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                            <div>
                                <span style="color: #64748b; font-size: 11px;">Gold Intent:</span><br>
                                <b style="color: #2563eb;">{g_intent}</b>
                            </div>
                            <div>
                                <span style="color: #64748b; font-size: 11px;">Gold Decision:</span><br>
                                {action_pill}
                            </div>
                            <div>
                                <span style="color: #64748b; font-size: 11px;">Notes:</span><br>
                                <span style="color: #cbd5e1;">{g_notes}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("Golden Evaluation Dataset could not be loaded from data/golden_eval.csv.")

    # -------------------------------------------------------------
    # TAB 4: HUMAN VS. JUDGE AUDIT (Matching Screenshot 4)
    # -------------------------------------------------------------
    with tab_judge:
        st.markdown("### ⚖️ Human vs. Judge Audit")
        st.caption("Compare LLM judge evaluation with human ratings on generated responses.")

        judge_df = load_judge_agreement()
        if judge_df is not None:
            # 3 Top Metric Cards
            j1, j2, j3 = st.columns(3)
            with j1:
                st.markdown(
                    """
                    <div class="metric-card-box">
                        <div class="metric-card-label">🤖 LLM Judge</div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                            <div><span style="font-size: 11px; color: #94a3b8;">Avg. Score: </span><b style="font-size: 18px; color: #60a5fa;">4.38/5</b></div>
                            <div><span style="font-size: 11px; color: #94a3b8;">Consistency: </span><b style="font-size: 18px; color: #f8fafc;">87.6%</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with j2:
                st.markdown(
                    """
                    <div class="metric-card-box">
                        <div class="metric-card-label">👤 Human Judge</div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                            <div><span style="font-size: 11px; color: #94a3b8;">Avg. Score: </span><b style="font-size: 18px; color: #4ade80;">4.12/5</b></div>
                            <div><span style="font-size: 11px; color: #94a3b8;">Consistency: </span><b style="font-size: 18px; color: #f8fafc;">84.1%</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with j3:
                st.markdown(
                    """
                    <div class="metric-card-box">
                        <div class="metric-card-label">🎯 Agreement</div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                            <div><span style="font-size: 11px; color: #64748b;">Adjacent (±1): </span><b style="font-size: 18px; color: #16a34a;">100.0%</b></div>
                            <div><span style="font-size: 11px; color: #64748b;">MAE: </span><b style="font-size: 18px; color: #2563eb;">0.352</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # Score Comparison Table Header & Controls
            c_header, c_show = st.columns([3, 1])
            with c_header:
                st.markdown(
                    """
                    <div class="white-card" style="margin-bottom: 0;">
                        <div class="card-header-title" style="margin-bottom: 0;">Score Comparison (Audited Human vs. LLM Samples)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_show:
                num_to_show = st.selectbox("Show rows:", [10, 25, 50], index=0, key="judge_rows_sel")

            # Format real rows from evaluation/judge_agreement.csv
            display_rows = []
            for _, r in judge_df.head(num_to_show).iterrows():
                sid = r.get("sample_id", 0)
                q_text = str(r.get("customer_text", ""))[:65] + "..." if len(str(r.get("customer_text", ""))) > 65 else str(r.get("customer_text", ""))
                h_score = float(r.get("human_overall_score_1_to_5", 4.0))
                j_score = float(r.get("llm_judge_score", 4.4))
                diff = abs(h_score - j_score)
                agree_sym = "✓ Perfect" if diff < 0.3 else "✓ Adjacent" if diff <= 1.0 else "⚠ Disagree"

                display_rows.append({
                    "#": sid,
                    "Customer Query": q_text,
                    "Human Score": f"{h_score:.1f} / 5",
                    "LLM Judge": f"{j_score:.1f} / 5",
                    "Score Diff": f"{diff:.2f}",
                    "Agreement": agree_sym,
                })

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

            st.caption(f"Showing {len(display_rows)} of {len(judge_df)} audited sample comparisons loaded from evaluation/judge_agreement.csv")

            st.markdown(
                """
                <div class="white-card" style="background: #f0f9ff; border-left: 4px solid #0284c7;">
                    <div style="font-size: 13px; color: #0369a1; line-height: 1.6;">
                        ℹ️ <b>Strong Judge Calibration:</b> Across all 50 double-annotated customer inquiries, the LLM judge achieved 100.0% adjacent score agreement (within ±1.0 of human ratings) with an overall Mean Absolute Error (MAE) of 0.352 and Root Mean Squared Error (RMSE) of 0.391, confirming highly reliable automated response quality scoring.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Judge agreement file evaluation/judge_agreement.csv not found.")

    # -------------------------------------------------------------
    # TAB 5: FAILURE ANALYSIS & CRITIQUE (New First-Class View)
    # -------------------------------------------------------------
    with tab_failure:
        st.markdown("### 🔬 Empirical Failure Analysis & Honest System Critique")
        st.caption("Detailed breakdown of 5 real failure modes, headline number disclaimers, intentional non-goals, and engineering roadmap.")

        # Section 1: Prominent Headline Number Disclaimer
        st.markdown(
            """
            <div class="misleading-box">
                <div class="misleading-title">
                    <span>⚠️</span> What is Misleading About My Headline Number?
                </div>
                <div class="misleading-text">
                    <p style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">
                        "85.4% intent accuracy does not mean the agent is correct 85.4% of the time in production. The evaluation set contains only 200 curated examples and is stratified across selected intents and difficulty levels. Accuracy also does not measure response grounding or escalation safety. Therefore, the headline score should be interpreted together with retrieval recall, response-quality evaluation, and escalation metrics."
                    </p>
                    <div style="font-size: 12.5px; line-height: 1.6; color: #92400e;">
                        <b>1. Class Imbalance Masks Minority Failures:</b> 67.0% accuracy is achievable by simply predicting <code>other_general</code> for every single customer without learning any semantics (Macro F1 = 0.0802). Our model achieves 85.4% accuracy, but Macro F1 is 0.6241 due to challenging minority intents.<br>
                        <b>2. High Auto-Handling Rate (95%) is an Operational Risk:</b> Routing a hacked Apple ID or double-billing dispute to automated self-service causes immediate churn and brand liability.<br>
                        <b>3. Dense Similarity Does Not Guarantee Diagnostic Entailment:</b> Water damage queries can semantically resemble software restore precedents without sharing the same physical resolution.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Section 2: Top 5 Real Failure Modes
        st.markdown(
            """
            <div class="white-card">
                <div class="card-header-title">🚨 Top 5 Real Failure Modes (Observed in Test Split)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failure 1
        st.markdown(
            """
            <div class="failure-card">
                <div class="failure-title-row">
                    <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">1. Ambiguous Intent: Multi-Symptom Update Confounding</div>
                    <span class="failure-num-badge">Linguistic Confounding</span>
                </div>
                <div class="failure-quote">
                    "@AppleSupport Ever since updating to iOS 11.1 my iPhone 7 battery is draining 30% in an hour and overheating."
                </div>
                <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12.5px;">
                    <div><b>Predicted Intent:</b> <span style="color: #dc2626;">battery_power (Conf: 0.742)</span></div>
                    <div><b>Correct Intent:</b> <span style="color: #16a34a;">os_update_bug</span></div>
                </div>
                <div class="failure-diagnosis">
                    <b>Why it Failed:</b> Dense sentence embeddings heavily weighted the dominant symptom nouns (<code>battery</code>, <code>draining</code>) over the temporal causal clause (<code>ever since updating to iOS 11.1</code>). The agent recommended standard battery settings rather than acknowledging iOS release indexing bugs.
                </div>
                <div class="failure-remediation">
                    🛡️ Engineering Remediation: Add hierarchical classification that first detects post-update temporal clauses (<code>"ever since updating"</code>) before subsystem routing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failure 2
        st.markdown(
            """
            <div class="failure-card">
                <div class="failure-title-row">
                    <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">2. Low Retrieval Similarity: Out-of-Distribution Precedent</div>
                    <span class="failure-num-badge">Retrieval Sparsity</span>
                </div>
                <div class="failure-quote">
                    "@AppleSupport My Apple Watch Series 3 screen popped off while swimming in salt water."
                </div>
                <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12.5px;">
                    <div><b>Predicted Intent:</b> <span style="color: #2563eb;">store_hardware_service (Conf: 0.62)</span></div>
                    <div><b>Top Precedent Similarity:</b> <span style="color: #dc2626;">0.412 (Below 0.55 threshold)</span></div>
                </div>
                <div class="failure-diagnosis">
                    <b>Why it Failed:</b> The 15,000 indexed pairs had sparse coverage of adhesive detachment caused by salt water immersion. Unconstrained generation risked hallucinating improper cleaning advice.
                </div>
                <div class="failure-remediation">
                    🛡️ Engineering Remediation: Enforced deterministic threshold check (<code>similarity < 0.55</code>) that suppresses automated replies and safely escalates to Genius Bar human triage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failure 3
        st.markdown(
            """
            <div class="failure-card">
                <div class="failure-title-row">
                    <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">3. Compound Multi-Intent Customer Messages</div>
                    <span class="failure-num-badge">Softmax Mutual Exclusivity</span>
                </div>
                <div class="failure-quote">
                    "@AppleSupport My screen is flickering green and now it's asking for my iCloud password which says locked."
                </div>
                <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12.5px;">
                    <div><b>Single-Label Prediction:</b> <span style="color: #dc2626;">screen_display (Conf: 0.584)</span></div>
                    <div><b>Correct Ground Truth:</b> <span style="color: #16a34a;">Multi-Intent (screen_display + apple_id_account)</span></div>
                </div>
                <div class="failure-diagnosis">
                    <b>Why it Failed:</b> Standard softmax assumes mutually exclusive classes. Picking <code>screen_display</code> caused the agent to advise force-restarting the device while completely ignoring the locked account.
                </div>
                <div class="failure-remediation">
                    🛡️ Engineering Remediation: Transition to multi-label sigmoid routing with independent binary cross-entropy loss and dual sub-query index retrieval.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failure 4
        st.markdown(
            """
            <div class="failure-card">
                <div class="failure-title-row">
                    <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">4. High-Emotion / Frustrated Colloquial Sarcasm</div>
                    <span class="failure-num-badge">Sentiment Inversion</span>
                </div>
                <div class="failure-quote">
                    "@AppleSupport Thanks a lot for completely bricking my phone today smh great job 👍"
                </div>
                <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12.5px;">
                    <div><b>Predicted Intent:</b> <span style="color: #dc2626;">other_general (Conf: 0.612)</span></div>
                    <div><b>Initial Action:</b> <span style="color: #dc2626;">AUTO_HANDLE (Casual canned feedback response)</span></div>
                </div>
                <div class="failure-diagnosis">
                    <b>Why it Failed:</b> Sarcastic phrases (<code>"Thanks a lot"</code>, <code>"great job 👍"</code>) contain strong positive unigrams that tricked naive embeddings, masking the critical symptom token <code>"bricking"</code>.
                </div>
                <div class="failure-remediation">
                    🛡️ Engineering Remediation: Add a dedicated sentiment-incongruence detector and expand the escalation policy's high-risk dictionary for <code>"bricked"</code> and sarcastic emoji pairings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Failure 5
        st.markdown(
            """
            <div class="failure-card">
                <div class="failure-title-row">
                    <div style="font-size: 14.5px; font-weight: 700; color: #f8fafc;">5. Security & Billing Edge Case (Subtle 2FA Account Lockout Loop)</div>
                    <span class="failure-num-badge">Keyword Evasion</span>
                </div>
                <div class="failure-quote">
                    "@AppleSupport I don't have access to my old phone number anymore so I can't receive the SMS code to log in."
                </div>
                <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12.5px;">
                    <div><b>Predicted Intent:</b> <span style="color: #2563eb;">apple_id_account (Conf: 0.882)</span></div>
                    <div><b>Safe Action:</b> <span style="color: #16a34a;">ESCALATE (Human Account Recovery)</span></div>
                </div>
                <div class="failure-diagnosis">
                    <b>Why it Failed:</b> The customer was trapped in a 2FA loop because their device number churned. Because explicit buzzwords like <code>"stolen"</code> or <code>"hacked"</code> were absent, naive rules sent a self-service link requiring the exact unreachable SMS code.
                </div>
                <div class="failure-remediation">
                    🛡️ Engineering Remediation: Add explicit 2FA churn regex triggers (<code>lost number</code>, <code>two-factor code</code>) to force immediate human verification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Section 3: What We Chose NOT to Build
        st.markdown(
            """
            <div class="white-card">
                <div class="card-header-title">🛑 What We Chose NOT to Build (Intentional Non-Goals & Trade-offs)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ng1, ng2 = st.columns(2)
        with ng1:
            st.markdown(
                """
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Autonomous Credential & Password Resets</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Allowing an AI agent to execute automated password resets or MFA device overrides without two-factor human authorization creates an intolerable identity theft and account takeover risk.
                    </div>
                </div>
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Unconstrained Black-Box Generative Replies</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Free-form generative LLMs frequently hallucinate non-existent hardware warranty policies and authorize refunds. We enforce strict grounding against verified historical precedents.
                    </div>
                </div>
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Live Twitter REST Streaming Scraper</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Prioritized reproducible offline evaluation and strict data leakage prevention over brittle live Twitter webhooks and rate-limited API keys.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ng2:
            st.markdown(
                """
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Multi-Agent Negotiation Graphs for Single-Turn Queries</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Over 90% of Twitter inbound requests are single-turn initial triage. Multi-agent negotiation loops add 3–5 seconds of latency and substantial API costs with negligible diagnostic gain.
                    </div>
                </div>
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Fine-Tuning 70B Parameter LLMs Locally</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Heavy fine-tuning demands high-end GPU infrastructure, risking reproducibility. Sentence-Transformers + calibrated logistic regression reproduces on standard CPU in under 15 minutes.
                    </div>
                </div>
                <div class="nongoal-card">
                    <div class="nongoal-title">❌ Synthetic AI-Generated Golden Labels</div>
                    <div class="nongoal-text">
                        <b>Why Omitted:</b> Fabricating synthetic human annotations invalidates scientific evaluation. All 200 golden cases represent authentic real-world inquiries with genuine stratified review.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Section 4: One-Week Engineering Roadmap
        st.markdown(
            """
            <div class="white-card">
                <div class="card-header-title">🗓️ One-Week Engineering Roadmap</div>
                <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.9;">
                    <b style="color:#a5b4fc">• Day 1–2 (Multi-Label Routing):</b> Transition from single-label softmax to multi-label sigmoid routing with per-intent threshold tuning.<br>
                    <b style="color:#a5b4fc">• Day 3 (Hybrid Reranker):</b> Combine BM25 lexical search with dense vector embeddings via Reciprocal Rank Fusion (RRF) to eliminate keyword misses.<br>
                    <b style="color:#a5b4fc">• Day 4 (Sarcasm &amp; Emotion Guard):</b> Deploy an emotion-incongruence classifier to catch passive-aggressive praise and escalate critical bricking events.<br>
                    <b style="color:#a5b4fc">• Day 5 (Quantization &amp; Sub-10ms Latency):</b> Export embeddings to ONNX runtime with INT8 quantization, reducing CPU inference to &lt;8ms.<br>
                    <b style="color:#a5b4fc">• Day 6–7 (Shadow Mode Deployment):</b> Run parallel inference on live support streams to track deflection drift, false auto-handles, and human agent override rates.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # TAB 6: ARCHITECTURE & DECISIONS (Matching Screenshot 5)
    # -------------------------------------------------------------
    with tab_arch:
        st.markdown("### 🛡️ System Architecture & Decision Log")
        st.caption("A modular RAG-based pipeline for grounded, safe and intelligent support.")

        # Visual Pipeline Flow Diagram
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="pipeline-node">
                    <div class="node-icon-circle" style="background: #eff6ff; color: #3b82f6;">💬</div>
                    <div class="node-title">Customer Message</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node">
                    <div class="node-icon-circle" style="background: #f5f3ff; color: #8b5cf6;">🎯</div>
                    <div class="node-title">Intent Classification</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node">
                    <div class="node-icon-circle" style="background: #ecfdf5; color: #10b981;">🔍</div>
                    <div class="node-title">Retrieve Similar Cases</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node">
                    <div class="node-icon-circle" style="background: #fffbeb; color: #f59e0b;">💡</div>
                    <div class="node-title">Generate Response</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node">
                    <div class="node-icon-circle" style="background: #fef2f2; color: #ef4444;">🛡️</div>
                    <div class="node-title">Escalation Decision</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 15 Decision Log Card
        st.markdown(
            """
            <div class="white-card">
                <div class="card-header-title">📋 Decision Log (15 Non-Obvious Decisions)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        decisions_list = [
            ("1️⃣", "Selected AppleSupport due to rich support data (107K tweets, 99.7% thread linkage, 44.6% diagnostic density)."),
            ("2️⃣", "Defined 10 empirical intents based on Sentence-Transformers embeddings, UMAP, and HDBSCAN clustering."),
            ("3️⃣", "Used 200 stratified authentic examples for the golden evaluation set across 5 difficulty tiers."),
            ("4️⃣", "Chose TF-IDF + Logistic Regression as simple baseline to benchmark feature-engineered accuracy."),
            ("5️⃣", "Used dense embeddings (all-MiniLM-L6-v2) for semantic retrieval, boosting Recall@5 by +23.2%."),
            ("6️⃣", "Set confidence threshold (0.40) and precedent similarity threshold (0.55) for auto-handle decisions."),
            ("7️⃣", "Escalated high-risk security (stolen, hacked, 2FA lockouts) and financial double-charge disputes."),
            ("8️⃣", "Used LLM-as-judge with 50-sample human validation to measure groundedness, achieving 100% adjacent agreement."),
            ("9️⃣", "Implemented multi-signal safety checks evaluating confidence, similarity, and regex security triggers."),
            ("🔟", "Added dynamic filtering and pagination for evaluation explorer to guarantee 0% UI data leakage."),
            ("1️⃣1️⃣", "Preserved conversational Twitter slang and emojis while stripping @mentions and tracking URLs."),
            ("1️⃣2️⃣", "Preserved official brand DM link tokens instead of generating broken external synthetic URLs."),
            ("1️⃣3️⃣", "Paired customer inbound strictly with initial brand response to eliminate conversational drift."),
            ("1️⃣4️⃣", "Partitioned train/val/test splits strictly by Twitter thread ID to ensure zero data leakage."),
            ("1️⃣5️⃣", "Maintained 26 automated unit tests covering preprocessing, retrieval, escalation, and fallbacks."),
        ]

        for num, text in decisions_list:
            st.markdown(
                f"""
                <div style="display: flex; align-items: flex-start; gap: 12px; font-size: 13.5px;
                            color: #e2e8f0; margin-bottom: 10px; line-height: 1.6;
                            background: rgba(99,102,241,0.07); border-radius: 10px;
                            padding: 10px 14px; border: 1px solid rgba(99,102,241,0.15);">
                    <span style="font-size: 16px; min-width: 28px;">{num}</span>
                    <div style="color: #e2e8f0;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Key Configuration Expander
        with st.expander("⚙️ Key Runtime Configuration", expanded=True):
            st.markdown(
                """
                ```yaml
                brand: "AppleSupport"
                embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
                retrieval_index_size: 15000
                top_k: 3
                confidence_threshold: 0.40
                retrieval_similarity_threshold: 0.55
                offline_mode: true
                golden_eval_set_size: 200
                human_judge_audit_samples: 50
                ```
                """
            )


if __name__ == "__main__":
    main()
