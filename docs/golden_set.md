# Golden Evaluation Set Documentation

**File Path:** `data/golden_eval.csv`  
**Total Curated Examples:** 200  
**Target Brand:** AppleSupport  

---

## 1. Sampling Methodology

The golden evaluation set is designed to evaluate real-world production performance under diverse linguistic, emotional, and technical conditions. It is **not** a uniform random sample of the training set.

### Stratification Dimensions:
1. **Balanced Intent Distribution:** Each of the 10 empirical intents is represented by 18–22 carefully audited queries.
2. **Linguistic Difficulty Tiers:**
   - **Short / Low-Context Inquiries (< 30 characters):** e.g., *"why is it laggy"*, *"fix this pls"*
   - **Noisy Messages with Slang & Typos:** e.g., *"phone wont charge smh"*, *"idk what happened"*
   - **High-Emotion & Frustration:** e.g., *"tried everything and still broken"*, *"unacceptable service"*
   - **Compound / Multi-Intent Expressions:** e.g., customer reporting both rapid battery drain and Bluetooth dropouts after an iOS update.
   - **Clean Standard Inquiries:** Detailed hardware and OS symptom reports.

---

## 2. Dataset Schema

The golden set is stored at `data/golden_eval.csv` with the following 9 columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `example_id` | Integer | Unique identifier (1 to 200). |
| `brand` | String | Target brand (`AppleSupport`). |
| `conversation_id` | String | Original Twitter root tweet ID linking back to raw data. |
| `customer_message` | String | Cleaned customer tweet (mentions stripped, slang/emojis preserved). |
| `context` | String | Inbound interaction metadata and linguistic difficulty tier. |
| `gold_intent` | String | Canonical ground-truth intent from the 10-class taxonomy. |
| `gold_action` | String | Operational routing decision: `AUTO_HANDLE` vs. `ESCALATE`. |
| `gold_reason` | String | Defensible operational justification for the routing decision. |
| `gold_reply_quality_notes` | String | Ground-truth criteria for evaluating draft reply quality. |

---

## 3. Labeling Guidelines & Ambiguity Handling

### A. Intent Resolution Rules
- **Multi-Intent Messages:** If a customer mentions an iOS update and battery drain, the label is assigned to `battery_power` if the primary question seeks battery troubleshooting, or `os_update_bug` if the complaint is primarily about system stability.
- **Hardware vs. Store Appointment:** Direct questions asking how to book Genius Bar appointments or repair costs are classified as `store_hardware_service`; physical symptom troubleshooting without appointment requests is classified as the specific hardware intent (e.g. `screen_display`).
- **Account Credentials vs. Charges:** Login, password, and 2FA SMS lockouts are classified as `apple_id_account`; credit card debits and subscription renewals are classified as `billing_subscription`.

### B. Escalation Decision Criteria
1. **Automatic Escalation (`ESCALATE`):**
   - Stolen, lost, or compromised devices.
   - Security lockouts, Apple ID account recovery disputes, and two-factor authentication failures.
   - Financial billing disputes, credit card double charges, or unapproved minor purchases.
   - Customer expressing severe frustration, repeated failed self-troubleshooting ("tried everything", "third time contacting you"), or explicit requests for a manager/human.
   - Ambiguous queries with zero actionable diagnostic symptoms where automated advice could mislead the user.
2. **Automated Handling (`AUTO_HANDLE`):**
   - Routine, self-serviceable software and hardware inquiries with high-confidence historical troubleshooting precedents (e.g., standard battery calibration, network reset paths, app reinstallation, iOS update checks).

---

## 4. Class Distribution

| Intent Name | Intent ID | Count | Percentage | Default Policy |
| :--- | :--- | :---: | :---: | :---: |
| Network & Connectivity | `connectivity_network` | 22 | 11.0% | AUTO_HANDLE |
| Media, Audio & Camera | `media_services` | 22 | 11.0% | AUTO_HANDLE |
| App Store & Apps | `app_store_issues` | 22 | 11.0% | AUTO_HANDLE |
| General Inquiries | `other_general` | 20 | 10.0% | AUTO_HANDLE |
| Apple ID & Security | `apple_id_account` | 20 | 10.0% | ESCALATE |
| Battery & Power | `battery_power` | 20 | 10.0% | AUTO_HANDLE |
| OS & Updates | `os_update_bug` | 19 | 9.5% | AUTO_HANDLE |
| Screen & Display | `screen_display` | 19 | 9.5% | AUTO_HANDLE |
| Store & Repairs | `store_hardware_service` | 18 | 9.0% | AUTO_HANDLE |
| Billing & Subscriptions | `billing_subscription` | 18 | 9.0% | ESCALATE |
| **Total** | | **200** | **100.0%** | |

---

## 5. Limitations

1. **Static Snapshot:** Tweets reflect the iOS 11 / iPhone 8 / iPhone X era present in Kaggle's historical Twitter support dataset.
2. **Context Horizon:** Single-turn customer starter tweets lack the extended conversational history of multi-day support tickets.
3. **Subjective Boundary Edge Cases:** In compound complaints (e.g. *"iOS 11 broke my Bluetooth and battery"*), assigning a single discrete intent label involves an inherent prioritization tradeoff.
