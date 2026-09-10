# Dataset Inspection Report: Customer Support on Twitter (AppleSupport Scope)

**Date of Inspection:** 2026-09-09  
**Inspector:** ML Engineering Agent  
**Target Brand:** AppleSupport  

---

## 1. Dataset Source & Archive Details

- **Found Location:** `C:\Users\anugu vaishnavi\Downloads\archive.zip`
- **Archive Size:** 176,772,673 bytes (~168.58 MB)
- **Archive Contents:**
  - `sample.csv` (17,357 bytes)
  - `twcs/twcs.csv` (516,508,641 bytes ~ 492.58 MB uncompressed)
- **Primary Data File:** `twcs/twcs.csv` (Customer Support on Twitter dataset by thoughtvector on Kaggle)

*Note: Per instructions, the 516 MB raw CSV is NOT copied in full into the Git repository to keep repository size lean. Instead, data extraction and preprocessing scripts stream or extract the required filtered brand subset directly into `data/processed/`.*

---

## 2. Dataset Dimensions & Schema

- **Total Rows across All Brands:** 2,811,774 rows
- **Columns (7 fields):**
  1. `tweet_id`: Unique identifier for each tweet (string/integer).
  2. `author_id`: Masked ID for customers (e.g., `115854`) or handle for company support (e.g., `AppleSupport`, `AmazonHelp`).
  3. `inbound`: Boolean (`True` if the tweet was sent by an end customer, `False` if sent by a corporate support agent).
  4. `created_at`: Timestamp string (e.g., `Tue Oct 31 22:10:47 +0000 2017`).
  5. `text`: Raw text content of the tweet.
  6. `response_tweet_id`: Comma-delimited list or single tweet_id responding to this tweet.
  7. `in_response_to_tweet_id`: The parent tweet_id this tweet is replying to.

---

## 3. Missing Value Analysis

Across the 2,811,774 records:

| Column | Missing Count | Percentage Missing | Meaning |
| :--- | :--- | :--- | :--- |
| `tweet_id` | 0 | 0.0% | Fully populated identifier |
| `author_id` | 0 | 0.0% | Fully populated |
| `inbound` | 0 | 0.0% | Fully populated boolean flag |
| `created_at` | 0 | 0.0% | Fully populated timestamp |
| `text` | 0 | 0.0% | Fully populated message body |
| `response_tweet_id` | 1,040,629 | 37.01% | Terminal tweets or unreplied customer tweets |
| `in_response_to_tweet_id` | 794,335 | 28.25% | Conversation starter tweets (root queries) |

No missing textual values or corrupted rows were detected in the core identification fields.

---

## 4. Brand Distribution & AppleSupport Representation

Top corporate support handles in the dataset:
1. `AmazonHelp`: 169,840 replies
2. **`AppleSupport`: 106,860 replies** (Second largest brand in the entire dataset)
3. `Uber_Support`: 56,270 replies
4. `SpotifyCares`: 43,265 replies
5. `Delta`: 42,253 replies

### AppleSupport-Specific Row Counts:
- **Outbound Support Replies (`author_id == 'AppleSupport'`):** 106,860
- **Outbound Replies with `in_response_to_tweet_id`:** 106,719 (99.87% of AppleSupport replies reference the preceding customer tweet)
- **Direct Inbound Customer Tweets mentioning `@AppleSupport`:** 97,138

---

## 5. Conversation Linking & Structure

AppleSupport conversations follow an explicit thread structure:
1. **Initial Customer Inbound:**
   - `author_id`: Anonymous customer ID (e.g., `115855`)
   - `inbound`: `True`
   - `text`: Starts with or mentions `@AppleSupport` (e.g., `"@AppleSupport Tried resetting my settings .. restarting my phone .. all that"`)
   - `in_response_to_tweet_id`: empty or previous tweet
2. **AppleSupport Outbound Reply:**
   - `author_id`: `AppleSupport`
   - `inbound`: `False`
   - `in_response_to_tweet_id`: Matches customer's `tweet_id`
   - `text`: Support advice, troubleshooting instructions, iOS version checks, or DM links (e.g., `"@115855 Any steps tried since it started last night?"`)

### Key Observations for Project Design:
1. **High Conversation Pair Density:** Over 106,000 direct customer-to-agent reply pairs are cleanly linkable via `in_response_to_tweet_id`.
2. **Realistic Customer Language:** Tweets contain emojis, typos, truncated URLs, iOS version mentions, frustration signals, and short fragmented sentences. As instructed, we will preserve this authentic noise.
3. **Common Support Patterns:** AppleSupport frequently diagnoses hardware/battery issues, iCloud/Apple ID lockouts, iOS update bugs, audio/Bluetooth glitches, and store/repair inquiries.