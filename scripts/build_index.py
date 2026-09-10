"""Script to build and save the historical resolution retrieval index.

Computes sentence embeddings for customer inquiries in the training set and builds
a vector index referencing the ground-truth AppleSupport responses.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.index import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_retrieval_index(
    data_path: str = "data/processed/train.csv",
    index_dir: str = "data/processed/retrieval_index",
    model_name: str = "all-MiniLM-L6-v2",
    max_records: int = 15000,
    batch_size: int = 128,
):
    """Build and save vector index from training conversation pairs."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}. Run scripts/prepare_data.py first.")

    logger.info("Loading conversation data from %s...", data_path)
    df = pd.read_csv(data_path)

    if max_records and len(df) > max_records:
        logger.info("Subsetting index records to %d for optimal latency and memory.", max_records)
        df = df.iloc[:max_records].copy()

    customer_texts = df["customer_text"].fillna("").tolist()
    metadata = df.to_dict(orient="records")

    logger.info("Loading embedding model %s...", model_name)
    encoder = SentenceTransformer(model_name)

    logger.info("Generating embeddings for %d historical customer inquiries (batch_size=%d)...", len(customer_texts), batch_size)
    embeddings = encoder.encode(
        customer_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info("Building vector index...")
    index = VectorIndex(use_faiss=True)
    index.build(embeddings, metadata)

    logger.info("Saving index to %s...", index_dir)
    index.save(index_dir)
    logger.info("Vector index successfully created and saved.")

    # Run quick sanity verification
    test_query = "My battery drops rapidly after the new iOS update"
    q_vec = encoder.encode([test_query], normalize_embeddings=True)
    results = index.search(q_vec, top_k=2)
    logger.info("Sanity check test query: '%s'", test_query)
    for r in results:
        logger.info(
            "  Rank %d (Score: %.4f): Inbound: '%s' -> Outbound: '%s'",
            r["rank"],
            r["similarity_score"],
            r["customer_text"][:60],
            r["agent_text"][:60],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector retrieval index for AppleSupport responses")
    parser.add_argument("--data_path", type=str, default="data/processed/train.csv", help="Input CSV path")
    parser.add_argument("--index_dir", type=str, default="data/processed/retrieval_index", help="Output directory")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--max_records", type=int, default=15000, help="Max historical records to index")

    args = parser.parse_args()
    build_retrieval_index(
        data_path=args.data_path,
        index_dir=args.index_dir,
        model_name=args.model,
        max_records=args.max_records,
    )
