# IntelliSupport AI — Autonomous Customer Support Agent

> **Hiver SDE Intern Assignment:** An end-to-end, empirically grounded AI Customer Support Agent built on Kaggle's *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`), specialized for **AppleSupport**.
> **GitHub Repository:** [vaishnavireddy067/IntelliSupport-AI-Customer-Service-Agent](https://github.com/vaishnavireddy067/IntelliSupport-AI-Customer-Service-Agent)

IntelliSupport AI provides reproducible data extraction, empirical intent discovery across 10 classes, verified historical precedent retrieval, grounded draft resolution, multi-signal safety escalation, and comprehensive benchmark evaluation with zero fabricated numbers.

---

## 1. Problem

Autonomous AI support agents in high-volume customer channels must resolve routine inquiries quickly without hallucinating non-existent policies, making unauthorized refund promises, or bungling sensitive account security lockouts.

This system addresses this challenge through a three-tier architecture:
1. **Accurate Intent Classification:** Categorizing noisy customer queries into an empirical 10-class taxonomy.
2. **Grounded Resolution Retrieval:** Finding verified historical AppleSupport troubleshooting precedents via dense vector embeddings (`all-MiniLM-L6-v2`).
3. **Deterministic Safety Escalation:** Using an explicit, auditable rule-based policy to route security, fraud, and high-risk queries to human agents (`AUTO_HANDLE` vs. `ESCALATE_TO_HUMAN`).

---

## 2. Approach & Architecture

```
[ Customer Tweet ] (Twitter noisy text, typos, slang)
        │
        ▼
[ Preprocessing ] (Preserves slang, emojis; strips leading @mentions)
        │
        ▼
[ Intent Classifier ] (Dense Sentence Embeddings + Calibrated Classifier)
        │
        ▼
[ Vector Retrieval ] (Top-K search over 15,000 historical Apple resolutions)
        │
        ▼
[ Grounded Generator ] (Strict evidence-grounded prompt or deterministic fallback)
        │
        ▼
[ Escalation Policy ] (Confidence threshold, retrieval score, security keywords)
        │
        ▼
[ Output ] -> Decision: AUTO_HANDLE / ESCALATE_TO_HUMAN + Grounded Draft Reply
```

---

## 3. Dataset Scope: AppleSupport

- **Source:** Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`, 2,811,774 rows).
- **Brand:** `AppleSupport` (2nd largest brand with 106,860 outbound replies, 97,138 direct mentions).
- **Linkage Rate:** 99.87% of AppleSupport outbound replies link to preceding customer tweets.
- **Repository Hygiene:** The raw 516 MB archive is streamed directly; no bloated CSVs are committed to git.
- **Sample Cohort:** 25,000 deduplicated pairs (`random_seed=42`) split into Train (17,500), Val (3,750), and Test (3,750).

---

## 4. Setup & Installation

### Prerequisites
- Python 3.12 (or Python $\ge$ 3.10)
- Git

### Windows Setup
```powershell
# 1. Clone repository & enter directory
cd "Hiver- Assignment"

# 2. Create virtual environment
py -3.12 -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Linux / macOS Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Environment Variables

Copy the template file to configure optional API credentials:
```powershell
copy .env.example .env
```
*(Optional)* Add an `OPENAI_API_KEY` for LLM-based reply generation and LLM judge evaluation.  
**Note:** The system is **100% operational offline with zero cost** using its built-in deterministic historical retrieval fallback.

---

## 6. Reproducibility & Execution Commands

Every step can be reproduced with a single command using fixed random seeds (`seed=42`):

## 6. Reproducibility & Execution Commands (Under 15 Minutes)

Every headline result, baseline, and evaluation metric can be reproduced in minutes from the command line:

### Step 1: Run Automated Test Suite
```powershell
.venv\Scripts\pytest -v
```
*(All 26 unit tests pass in ~3 seconds verifying data loader, leakage prevention, retrieval ranking, intent classifiers, baselines, and escalation policies).*

### Step 2: Extract & Sample Data (Phase 1)
```powershell
.venv\Scripts\python scripts/sample_data.py --sample_size 25000 --seed 42
```
*(Streams `archive.zip`, extracts AppleSupport dialogue pairs, and splits into train/val/test with 0% query leakage).*

### Step 3: Run Baseline Comparison (Phase 5)
```powershell
.venv\Scripts\python scripts/run_baselines.py
```
*(Trains and evaluates Majority Baseline and TF-IDF Baseline on identical test data).*

### Step 4: Build Vector Retrieval Index (Phase 6)
```powershell
.venv\Scripts\python scripts/build_index.py --data_path data/processed/train.csv --max_records 15000
```
*(Indexes 15,000 AppleSupport precedents into dense vector space using `all-MiniLM-L6-v2`).*

### Step 5: Generate Curated Golden Evaluation Set (Phase 10)
```powershell
.venv\Scripts\python scripts/build_golden_set.py --n_samples 200 --seed 42
```
*(Samples 200 diverse queries across difficulty tiers at `data/golden_eval.csv` and documented in `docs/golden_set.md`).*

### Step 6: Run Full Headline Comparative Evaluation (Phase 11)
```powershell
.venv\Scripts\python -m evaluation.evaluate
```
*(Generates the final Headline Metrics table across all three systems and updates `evaluation/results.json` & `evaluation/results.csv`).*

### Step 7: Run LLM-as-a-Judge vs. Human Agreement Audit (Phase 12)
```powershell
.venv\Scripts\python -m evaluation.judge_agreement
```
*(Evaluates 50 human-audited support replies against the LLM judge rubric and generates `evaluation/judge_agreement.csv`).*

---

## 7. Interactive Agent Demos

### CLI Demo (Zero API Key Needed):
```powershell
.venv\Scripts\python -m src.agent "@AppleSupport my iPhone 8 battery drops from 80% to 15% in one hour after iOS 11 update"
```

### Sample CLI Output:
```json
{
  "intent": "battery_power",
  "confidence": 0.9907,
  "draft_reply": "We're here to help. We’d love to help with your battery. To start, please send us a DM to continue. https://t.co/GDrqU22YpT",
  "decision": "AUTO_HANDLE",
  "escalation_reason": "High intent confidence (0.991) and strong historical match (0.799) for routine self-service resolution.",
  "evidence": [
    {
      "conversation_id": "1072237",
      "similarity": 0.799,
      "historical_query": "i have an issue regarding my iPhone 8 battery. I unplugged it like an hour ago and it’s already down to 66%.",
      "historical_reply": "We’d love to help with your battery. To start, please send us a DM to continue. https://t.co/GDrqU22YpT"
    },
    {
      "conversation_id": "854612",
      "similarity": 0.798,
      "historical_query": "What the fucc is up with my battery after @115858 iOS 11 update?? Dropped from 50% to 7% in 5 minutes. Battery lasted 2.30 hours today.",
      "historical_reply": "Thanks for reaching out to us. Let's talk more in DM. https://t.co/GDrqU22YpT"
    },
    {
      "conversation_id": "679936",
      "similarity": 0.7561,
      "historical_query": "fix IOS 11 and the battery issues. My phones on power saver and it’s dropped 20% in 20 minutes",
      "historical_reply": "We absolutely understand the importance of battery life and we're here to help. Are you on iOS 11.0.2?"
    }
  ]
}
```

### Lightweight Streamlit Web UI:
```powershell
.venv\Scripts\streamlit run demo_app.py
```

---

## 8. Headline Evaluation Results

Computed directly on the AppleSupport test split (`seed=42`) and serialized to [`evaluation/results.json`](evaluation/results.json) & [`evaluation/results.csv`](evaluation/results.csv):

| System | Intent Accuracy | Macro F1 | Retrieval R@5 | Reply Score | Escalation F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority baseline** | 0.6700 | 0.0802 | 0.0000 | 3.00 | 0.0000 |
| **TF-IDF baseline** | 0.8600 | 0.6687 | 0.4520 | 3.80 | 0.6210 |
| **Proposed Agent** | **0.8540** | **0.6241** | **0.6840** | **4.38** | **0.8142** |

### LLM Judge vs. Human Agreement Audit (50 Samples)
- **Adjacent Score Agreement (+/- 1.0):** **100.0%**
- **Mean Absolute Error (MAE):** **0.3520**
- **Root Mean Squared Error (RMSE):** **0.3914**
- **Audit File:** [`evaluation/judge_agreement.csv`](evaluation/judge_agreement.csv)

---

## 9. Failure Analysis (Top 5 Modes)

Documented in depth with real examples in [`reports/failure_analysis.md`](reports/failure_analysis.md):
1. **Multi-Symptom Update Confounding:** Post-update temporal clauses confused with battery symptoms.
2. **Compound Multi-Intent Inquiries:** Single-label softmax forcing a single choice on multi-problem complaints.
3. **Sarcasm / Sarcastic Praise:** Colloquial sarcasm (*"Thanks for bricking my phone 👍"*) tricking dense embeddings.
4. **Measurement Disconnect on Strict ID Recall:** Test queries retrieve semantically valid solutions that have different historical tweet IDs.
5. **Subtle 2FA Account Lockouts:** Non-buzzword account disputes slipping past keyword rules.

---

## 10. Mandatory Section: "What is misleading about my headline number?"

Our headline intent accuracy of **85.40%** and deflection rate look impressive at first glance. However, this headline number is misleading for three critical reasons:

1. **Class Imbalance Conceals Rare Intent Failures:** Over 65% of customer inquiries are general complaints (`other_general`). A zero-skill majority baseline achieves 67.00% accuracy simply by guessing the majority class every time, while obtaining a near-zero Macro F1 (0.0802). Our proposed model achieves a high overall accuracy, but its Macro F1 is 0.6241, meaning performance on rare intents like `store_hardware_service` and `os_update_bug` is lower than headline accuracy implies.
2. **High Auto-Handling (95%) Hides Critical False-Auto-Handle Risks:** A false auto-handle (routing an account compromise or stolen phone to automated troubleshooting) carries immense brand and legal liability. A deflection rate is only as good as its safety boundary calibration.
3. **Retrieval Proximity $\neq$ Diagnostic Entailment:** Cosine similarity measures linguistic likeness, not physical causation. A customer inquiry about water damage can linguistically match a precedent about software restore, risking inappropriate advice if not guarded by strict escalation policies.

See the full technical discussion in [`reports/report.md`](reports/report.md).

```
hiver-ai-support-agent/
├── README.md                          # Project guide & reproduction instructions
├── requirements.txt                   # Minimal pinned dependencies
├── .gitignore                         # Python, environment, and dataset ignores
├── .env.example                       # API key configuration template
├── pytest.ini                         # Pytest path configuration
│
├── data/
│   ├── raw/                           # .gitkeep (external archive.zip)
│   ├── processed/                     # train.csv, val.csv, test.csv, retrieval_index/
│   └── golden/
│       ├── golden_set.csv             # 200 stratified candidate queries
│       └── human_ratings_template.csv # 50 samples for judge-human rating audit
│
├── src/
│   ├── agent.py                       # End-to-end AppleSupport agent
│   ├── data/                          # Data loading & noise-preserving preprocessing
│   ├── intents/                       # Taxonomy, Majority baseline, TF-IDF, MiniLM
│   ├── retrieval/                     # Vector index & Top-K retrieval
│   ├── generation/                    # Grounded reply drafting with zero-cost fallback
│   ├── escalation/                    # Rule-based escalation policy
│   └── evaluation/                    # Metrics, LLM judge rubric, evaluation pipeline
│
├── scripts/
│   ├── prepare_data.py                # Extracts & splits 25,000 pairs from archive.zip
│   ├── build_index.py                 # Builds 15,000-vector retrieval index
│   ├── create_golden_template.py      # Creates 200-sample human golden evaluation template
│   └── run_evaluation.py              # Executes benchmarks & outputs baseline_results.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Message lengths, linkages, noise analysis
│   └── 02_intent_analysis.ipynb       # Intent distributions & confusion analysis
│
├── reports/
│   ├── data_inspection.md             # Dataset dimensions, columns, AppleSupport stats
│   ├── baseline_results.json          # Machine-readable benchmark results
│   ├── confusion_matrix.png           # Confusion matrix heatmap for final classifier
│   ├── decision_log.md                # 14 non-obvious engineering decisions
│   ├── failure_analysis.md            # Top 5 empirical failure modes
│   └── report.md                      # 6-page comprehensive technical report
│
└── tests/                             # Unit tests for preprocessing, models, retrieval, policy
    ├── test_preprocess.py
    ├── test_classifier.py
    ├── test_retrieval.py
    └── test_escalation.py
```
