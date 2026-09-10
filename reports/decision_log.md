# Engineering Decision Log: Hiver AI Support Agent (AppleSupport)

This document records key non-obvious architectural, algorithmic, and methodological engineering decisions made during the development of the AppleSupport Customer Support AI Agent.

---

### Decision 1: Focus Exclusively on AppleSupport
- **Decision:** Restrict the dataset scope to a single corporate handle (`AppleSupport`) rather than mixing multi-brand conversations.
- **Reason:** Customer support domain vocabularies, resolution workflows, and escalation thresholds diverge drastically across industries (e.g. airline flight cancellations vs. consumer electronics hardware troubleshooting). Training on a single brand yields coherent semantic embeddings and consistent troubleshooting knowledge.
- **Alternative Considered:** Training a universal multi-brand agent across all 2.8M tweets.
- **Why Alternative Was Rejected:** Dilutes brand tone, introduces contradictory brand policies, and drastically increases computational and memory overhead without improving domain-specific resolution quality.
- **Impact:** Clean, domain-coherent retrieval and highly relevant historical resolution suggestions.

---

### Decision 2: Stream Data Directly from `archive.zip` without Duplicating the 516 MB CSV into Git
- **Decision:** Implement a zip-streaming reader in `src/data/load_data.py` to extract AppleSupport pairs without copying or committing the 516 MB raw dataset into the local git repository.
- **Reason:** Large raw files bloat version control, slow down git clones, and violate best practices for repository hygiene.
- **Alternative Considered:** Unzipping the 516 MB `twcs.csv` into `data/raw/` and committing it.
- **Why Alternative Was Rejected:** Git is inefficient with half-gigabyte text files and GitHub imposes strict 100 MB file upload limits.
- **Impact:** Kept the repository lean, fast, and GitHub-ready while preserving 100% reproducible data extraction.

---

### Decision 3: Reconstruct Two-Turn Conversation Pairs via `in_response_to_tweet_id`
- **Decision:** Reconstruct explicit Customer Inbound $\rightarrow$ AppleSupport Outbound conversation pairs by matching AppleSupport tweets (`author_id == 'AppleSupport'`) back to their parent customer tweets (`in_response_to_tweet_id == tweet_id`).
- **Reason:** Outbound support tweets alone lack customer problem context, while customer inbound tweets alone lack verified brand resolution steps. Grounding response generation requires paired problem-resolution data.
- **Alternative Considered:** Treating all tweets as independent text documents or using entire multi-turn thread trees.
- **Why Alternative Was Rejected:** Unpaired documents provide no supervision for what response resolves what problem. Full thread trees add immense graph traversal complexity and contain customer "thank you" sign-offs that add little diagnostic value.
- **Impact:** 48,754 high-quality unique problem-solution pairs readily available for retrieval and evaluation.

---

### Decision 4: Preserve Authentic Customer Noise (Typos, Slang, Punctuation, Emojis)
- **Decision:** Only decode HTML entities and strip leading Twitter handle mentions (`@AppleSupport`), strictly preserving typos, slang, casing, punctuation, and emojis in customer messages.
- **Reason:** Real customer support tickets contain emotional punctuation (e.g., `"????"`, `"smh"`, `"pls fix"`). Aggressive lemmatization or spelling correction strips urgency and emotion signals crucial for intent classification and escalation.
- **Alternative Considered:** Aggressive text normalization (lowercasing, spell checking, regex stopword stripping, emoji deletion).
- **Why Alternative Was Rejected:** Eliminates emotional features necessary for detecting customer frustration and risk escalation; creates an unrealistic benchmark that fails in real-world deployments.
- **Impact:** Models trained and tested on realistic noisy inputs reflecting actual production traffic.

---

### Decision 5: Compact 10-Class Empirical Intent Taxonomy
- **Decision:** Establish an empirical 10-class intent taxonomy (`battery_power`, `os_update_bug`, `screen_display`, `apple_id_account`, `connectivity_network`, `app_store_issues`, `billing_subscription`, `media_services`, `store_hardware_service`, `other_general`).
- **Reason:** Direct n-gram and keyword frequency analysis of 10,000 AppleSupport inquiries revealed these specific issue clusters. A 10-class taxonomy provides granular resolution routing while maintaining sufficient sample support per class.
- **Alternative Considered:** Using generic categories (e.g. "technical issue", "billing", "other") or creating 30+ fine-grained classes (e.g. separating iPhone 7 battery from iPhone X battery).
- **Why Alternative Was Rejected:** 3 generic classes fail to provide actionable routing, while 30+ classes create severe data sparsity and high classification error rates.
- **Impact:** High inter-class distinctiveness and high practical utility for automated routing.

---

### Decision 6: Mandatory Majority-Class Baseline
- **Decision:** Implement and evaluate a trivial majority-class classifier (`MajorityIntentClassifier`) as Baseline 1.
- **Reason:** A majority baseline provides an empirical zero-skill floor. Without it, reported accuracy numbers (e.g., 65%) lack context on whether the model is learning meaningful patterns or merely exploiting class imbalance.
- **Alternative Considered:** Skipping the trivial baseline and only comparing TF-IDF against Neural Embeddings.
- **Why Alternative Was Rejected:** Reviewers cannot assess relative performance gain without a proper zero-skill baseline.
- **Impact:** Demonstrates that ML and embedding models learn genuine semantic distinctions rather than majority priors.

---

### Decision 7: TF-IDF + Logistic Regression as Baseline 2
- **Decision:** Use Sublinear TF-IDF with unigrams and bigrams + Logistic Regression as the classic statistical ML baseline.
- **Reason:** TF-IDF is computationally lightweight, interpretable, deterministic, and serves as an industry gold standard benchmark for text classification.
- **Alternative Considered:** Naive Bayes or Random Forest.
- **Why Alternative Was Rejected:** Naive Bayes makes unrealistic feature independence assumptions that hurt performance on n-grams; Random Forest on high-dimensional sparse text matrices is slow to train and prone to overfitting without boosting accuracy.
- **Impact:** Established a strong 75%+ accuracy baseline with sub-second inference latency.

---

### Decision 8: Final Intent Classifier Using Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Decision:** Adopt `all-MiniLM-L6-v2` dense sentence embeddings feeding into a calibrated Logistic Regression classifier.
- **Reason:** MiniLM maps semantically equivalent paraphrases (e.g., *"battery dying fast"* vs. *"drops from 80% to 10% in an hour"*) to neighboring regions in 384-dimensional space, overcoming TF-IDF's out-of-vocabulary and lexical mismatch limitations.
- **Alternative Considered:** Fine-tuning an entire BERT/RoBERTa model end-to-end.
- **Why Alternative Was Rejected:** Full neural fine-tuning requires heavy GPU infrastructure, incurs high deployment costs, increases latency, and risks catastrophic forgetting.
- **Impact:** Superior generalization on noisy, paraphrased customer inputs with fast CPU inference.

---

### Decision 9: Dual Vector Search Engine with Automatic FAISS and Numpy Fallback
- **Decision:** Architect `VectorIndex` to leverage FAISS (`IndexFlatIP`) when available, while providing an automatic fallback to normalized cosine matrix multiplication using NumPy.
- **Reason:** Windows environments frequently face binary wheel incompatibilities with FAISS CPU. A seamless fallback guarantees 100% runnable code across any reviewer's operating system without sacrificing mathematical correctness.
- **Alternative Considered:** Requiring FAISS as a strict mandatory dependency.
- **Why Alternative Was Rejected:** Would cause build and test crashes on Windows systems lacking C++ OpenMP runtimes.
- **Impact:** Zero installation friction, zero platform lock-in, and identical mathematical ranking.

---

### Decision 10: Grounded Response Generation with Deterministic Zero-Cost Retrieval Fallback
- **Decision:** Implement a dual-mode response generator that uses an LLM API when keys are configured in `.env`, but automatically defaults to an adapted historical AppleSupport response when no API key exists.
- **Reason:** Ensures the entire project is 100% executable and evaluable out-of-the-box by any reviewer without requiring paid API credits.
- **Alternative Considered:** Mocking LLM API calls with random hardcoded responses, or failing with an error if no API key is set.
- **Why Alternative Was Rejected:** Fabricating mock outputs violates assignment integrity, while hard errors break end-to-end pipeline execution.
- **Impact:** Fully operational, reproducible offline demo with zero cost.

---

### Decision 11: Explicit Rule-Based Escalation Policy Over LLM Self-Decision
- **Decision:** Use an explicit, multi-signal rule-based engine (`src/escalation/policy.py`) for the `AUTO_HANDLE` vs. `ESCALATE_TO_HUMAN` decision, rather than asking the LLM to decide.
- **Reason:** LLMs exhibit unpredictable hallucinations, sycophancy, and variable calibration. Customer safety, legal risk, fraud detection, and refund governance require strict deterministic guarantees and auditable logic.
- **Alternative Considered:** Adding `"Should we escalate? (yes/no)"` to the LLM generation prompt.
- **Why Alternative Was Rejected:** Unauditable, non-deterministic, vulnerable to prompt injection, and difficult to calibrate safety thresholds against.
- **Impact:** Completely transparent, deterministic escalation decisions with clear causal reasons.

---

### Decision 12: Stratified Golden Evaluation Set with Unfabricated Labels
- **Decision:** Sample 200 authentic customer tweets across Easy, Medium, and Hard linguistic categories, leaving gold fields for genuine human annotation rather than pre-filling synthetic labels.
- **Reason:** Fabricating AI-generated annotations and presenting them as "human gold labels" invalidates scientific evaluation integrity.
- **Alternative Considered:** Using GPT-4 to generate pseudo-human labels and claiming 100% human agreement.
- **Why Alternative Was Rejected:** Strictly forbidden by assignment instructions ("DO NOT fabricate human labels... Every reported number must come from an actual experiment").
- **Impact:** High-integrity evaluation template that clearly distinguishes verified empirical data from pending human audits.

---

### Decision 13: Reporting Both Macro F1 and Weighted F1 Alongside Accuracy
- **Decision:** Mandate reporting Macro F1 and Weighted F1 alongside Accuracy across all intent classification models.
- **Reason:** Support intent distributions in production are naturally imbalanced (e.g., OS update bugs and battery queries dominate billing disputes). Accuracy can appear deceptively high even if rare intents have near-zero recall.
- **Alternative Considered:** Reporting only Accuracy as the single headline number.
- **Why Alternative Was Rejected:** Hides severe performance degradation on critical minority classes (e.g. billing fraud or account recovery).
- **Impact:** Provides an honest, multi-dimensional assessment of classifier capability.

---

### Decision 14: No Production Twitter Bot or Frontend UI
- **Decision:** Exclude Twitter REST API streaming clients, webhooks, and React frontends, focusing all engineering effort on data modeling, retrieval, evaluation, and documentation.
- **Reason:** The assignment objective is to build a robust, reproducible ML customer support core. Live Twitter bots require developer portal approvals, rate limits, and authentication ceremonies that distract from core ML evaluation.
- **Alternative Considered:** Building a React/Streamlit chat UI or connecting live Twitter stream webhooks.
- **Why Alternative Was Rejected:** Explicitly discouraged by prompt ("DO NOT overbuild this project... DO NOT create a fancy frontend... DO NOT create a production Twitter bot").
- **Impact:** A clean, focused, maintainable ML codebase with maximum reproducibility.

---

### Decision 15: Offline Sentence-Transformer Initialization via `HF_HUB_OFFLINE`
- **Decision:** Configure `os.environ.setdefault("HF_HUB_OFFLINE", "1")` across inference scripts to load cached Sentence-Transformers without outbound network calls.
- **Reason:** Default Hugging Face Hub behavior sends synchronous HTTP HEAD requests to verify remote model commits on every invocation. On developer laptops or firewalled evaluation environments, this causes 15–30 second latency spikes and network timeout errors.
- **Alternative Considered:** Allowing online Hugging Face Hub calls on every execution.
- **Why Alternative Was Rejected:** Introduces non-deterministic network failure risks and violates the requirement that Headline results reproduce in under 15 minutes.
- **Impact:** Sub-second model initialization and 100% reliable offline CLI inference.
