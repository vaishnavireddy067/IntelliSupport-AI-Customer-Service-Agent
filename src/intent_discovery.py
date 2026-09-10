"""Intent Discovery and Empirical Topic Extraction Module.

Performs exploratory data analysis on real AppleSupport customer inquiries to derive
a data-driven, non-arbitrary customer support taxonomy.
"""

import os
import re
import yaml
from collections import Counter
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def discover_top_ngrams(
    texts: List[str],
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, int]]]:
    """Extract most frequent unigrams and bigrams from raw customer queries."""
    unigram_counts = Counter()
    bigram_counts = Counter()

    for text in texts:
        if not text or not isinstance(text, str):
            continue
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        unigram_counts.update(tokens)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigram_counts[bigram] += 1

    return {
        "unigrams": unigram_counts.most_common(top_n),
        "bigrams": bigram_counts.most_common(top_n),
    }


def discover_tfidf_keywords(
    texts: List[str],
    max_features: int = 30,
) -> List[Tuple[str, float]]:
    """Discover highest-scoring salient TF-IDF terms excluding common stopwords."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
    )
    cleaned_texts = [t for t in texts if t and isinstance(t, str)]
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    term_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0])
    return sorted(term_scores, key=lambda x: x[1], reverse=True)


def load_intent_taxonomy(yaml_path: str = "configs/intents.yaml") -> Dict[str, Any]:
    """Load intent taxonomy configuration from YAML."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Taxonomy configuration not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_discovery_report(
    data_path: str = "data/processed/train.csv",
    taxonomy_path: str = "configs/intents.yaml",
) -> None:
    """Run discovery and display empirical topic distribution and taxonomy alignment."""
    print(f"\n=======================================================")
    print(f"      AppleSupport Intent Discovery & Taxonomy Report   ")
    print(f"=======================================================\n")

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run scripts/sample_data.py first.")
        return

    df = pd.read_csv(data_path)
    texts = df["customer_text"].dropna().tolist()
    print(f"Loaded {len(texts):,} customer inquiries for empirical topic modeling.")

    # 1. Top N-grams
    ngrams = discover_top_ngrams(texts, top_n=10)
    print("\n--- Top Customer Unigrams ---")
    for word, count in ngrams["unigrams"]:
        print(f"  {word:15s}: {count:,}")

    print("\n--- Top Customer Bigrams ---")
    for phrase, count in ngrams["bigrams"]:
        print(f"  {phrase:20s}: {count:,}")

    # 2. TF-IDF Salient Terms
    tfidf_terms = discover_tfidf_keywords(texts, max_features=15)
    print("\n--- Salient TF-IDF Diagnostic Terms ---")
    for term, score in tfidf_terms:
        print(f"  {term:20s}: {score:.1f}")

    # 3. Mapped Taxonomy
    taxonomy = load_intent_taxonomy(taxonomy_path)
    print(f"\n--- Defined Empirical Taxonomy ({len(taxonomy['intents'])} Intents) ---")
    print("| Intent ID | Intent Name | Escalation Default | Positive Sample |")
    print("| :--- | :--- | :--- | :--- |")
    for item in taxonomy["intents"]:
        sample = item["positive_examples"][0] if item["positive_examples"] else "N/A"
        default_act = item.get("escalation_considerations", {}).get("default_action", "AUTO_HANDLE")
        print(f"| `{item['intent_id']}` | {item['intent_name']} | `{default_act}` | \"{sample[:45]}...\" |")


if __name__ == "__main__":
    print_discovery_report()
