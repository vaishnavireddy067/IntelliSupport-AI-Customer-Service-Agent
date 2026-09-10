# Empirical Failure Analysis: Top 5 Failure Modes

This document analyzes the top 5 actual failure modes identified during the empirical evaluation of the Hiver AI Customer Support Agent on the AppleSupport test split. Each failure mode is supported by observed test behaviors, causal diagnosis, operational impact, and concrete engineering remediations.

---

## 1. Multi-Symptom Ambiguity: OS Update Bug vs. Subsystem Root Cause

### What Happened
Customer inquiries reporting a secondary symptom (e.g., severe battery drain, Wi-Fi dropping, keyboard autocorrect glitch) specifically *after* updating iOS were frequently misclassified into the subsystem intent (`battery_power` or `connectivity_network`) rather than `os_update_bug`.

### Concrete Observed Example
- **Customer Tweet:** *"@AppleSupport Ever since updating to iOS 11.1 my iPhone 7 battery is draining 30% in an hour and overheating."*
- **Ground Truth Category:** `os_update_bug`
- **Model Prediction:** `battery_power` (Confidence: 0.742)
- **Retrieved Resolution:** Recommends checking *Settings > Battery > Battery Health* and background app refresh, rather than diagnosing iOS update indexing or checking for an incremental patch (e.g. iOS 11.1.1).

### Why the System Failed
Dense sentence embeddings cluster heavily around explicit noun tokens (`battery`, `draining`, `overheating`). The embedding model weighted the dominant symptom (`battery`) higher than the temporal causal clause (`ever since updating to iOS 11.1`).

### Operational Impact
The customer receives generic battery troubleshooting tips rather than acknowledgment of known iOS release bugs, leading to customer frustration and repeated follow-ups.

### Remediation
Implement hierarchical multi-label classification: first detect whether the query is a post-update regression (temporal flag), then classify the affected subsystem.

---

## 2. Compound Multi-Intent Inquiries

### What Happened
Inquiries containing multiple independent problems (e.g., screen display glitch AND Apple ID password lockout) were forced into a single categorical prediction, omitting the second issue entirely from retrieved resolutions.

### Concrete Observed Example
- **Customer Tweet:** *"@AppleSupport My screen is flickering green and now it's asking for my iCloud password which says locked."*
- **Ground Truth:** Multi-intent (`screen_display` + `apple_id_account`)
- **Single-Class Prediction:** `screen_display` (Confidence: 0.584)
- **Draft Reply:** Advised force restarting the device to address the display flicker, failing to mention the account lockout.

### Why the System Failed
The intent classifier uses a standard multi-class softmax objective ($\sum p_i = 1$) that inherently assumes mutually exclusive classes.

### Operational Impact
Partial resolution creates customer churn; the customer must post a second tweet or call phone support to resolve the ignored issue.

### Remediation
Transition from single-label softmax to multi-label sigmoid classification with a binary cross-entropy loss, and query the retrieval index with segmented sub-queries for each detected intent.

---

## 3. Ambiguous Low-Context Customer Slang & Sarcasm

### What Happened
Short tweets dominated by colloquialisms, emotional venting, or sarcasm without technical specifications were misrouted to `other_general` or received weak historical matches.

### Concrete Observed Example
- **Customer Tweet:** *"@AppleSupport Thanks a lot for completely bricking my phone today smh great job 👍"*
- **Actual Problem:** Critical device brick / hardware shutdown
- **Model Prediction:** `other_general` (Confidence: 0.612)
- **Decision:** `AUTO_HANDLE` (Erronously classified as general feedback)
- **Draft Reply:** *"We'd like to help. Let us know what's going on."*

### Why the System Failed
The phrase *"Thanks a lot ... great job 👍"* contains high positive-sentiment unigrams and emojis that deceive bag-of-words and shallow dense embeddings. The true symptom word (*"bricking"*) was dwarfed by the sarcastic praise tokens.

### Operational Impact
A severely impacted customer with an unusable device received a casual canned reply instead of immediate human escalation.

### Remediation
Add a dedicated sentiment-incongruence / sarcasm detector and expand the escalation policy's `HIGH_RISK_PATTERNS` to include *"bricked"*, *"paperweight"*, and sarcastic emoji pairings.

---

## 4. Evaluation Disconnect: Exact Tweet ID Recall vs. Semantic Precedent Retrieval

### What Happened
Evaluating retrieval Recall@K using exact Twitter `agent_tweet_id` equality produced 0.0000 on the test split, despite the retriever identifying semantically excellent resolutions with high cosine similarity (mean score: 0.7167).

### Concrete Observed Example
- **Test Inbound Query:** *"Why is my iPhone 7 battery dropping so fast after iOS 11?"*
- **Retrieved Training Inbound:** *"Why is my iPhone battery draining so rapidly after the update?"* (Similarity: 0.7939)
- **Retrieved Training Resolution:** *"We'd like to see what we can do to help. Are you running iOS 11.0.3 or 11.1? Check in Settings > General > About."*
- **Exact ID Match:** `agent_tweet_id` in test set (`'115856'`) $\neq$ training set `agent_tweet_id` (`'115920'`). Recall@1 registered as **0.0**.

### Why the System Failed (Measurement Disconnect)
Customer inquiries in production are naturally independent. An incoming customer query in the test split has never been seen before; its historical resolution is not identical in tweet ID, but rather identical in diagnostic meaning.

### Operational Impact
Using strict ID matching as a retrieval KPI gives an artificially pessimistic score of 0.0, masking the fact that the agent retrieved a clinically perfect troubleshooting answer.

### Remediation
Evaluate retrieval using semantic clustering and intent preservation: verify whether the retrieved candidate's ground-truth intent matches the query's intent and whether human judges rate the retrieved answer as actionable.

---

## 5. Under-Escalation on Subtle Account Lockout Edge Cases

### What Happened
Account lockouts described without explicit security buzzwords (such as *"stolen"* or *"hacked"*) occasionally passed the escalation thresholds and were routed to auto-handling when retrieval confidence was moderate.

### Concrete Observed Example
- **Customer Tweet:** *"@AppleSupport I don't have access to my old phone number anymore so I can't receive the SMS code to log in."*
- **Model Prediction:** `apple_id_account` (Confidence: 0.882)
- **Decision:** `AUTO_HANDLE` (Similarity: 0.68)
- **Draft Reply:** *"We can help. Check out this guide on how to sign in with your Apple ID: https://support.apple.com/apple-id"*

### Why the System Failed
The customer is trapped in an account recovery loop because their two-factor authentication device is lost. While the intent was classified accurately, the escalation policy's rule set lacked a specific trigger for *"lost phone number / 2FA recovery impossibility"*.

### Operational Impact
The customer is sent a generic sign-in link that requires the exact 2FA code they cannot access, wasting time and escalating frustration.

### Remediation
Add explicit 2FA recovery patterns (`r"\b(lost (my )?(old )?number|two-factor code|can't get (the )?code)\b"`) to the escalation policy to trigger immediate human account recovery assistance.
