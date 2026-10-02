#!/usr/bin/env python
"""Standalone script to build and serialize Content-Based TF-IDF Recommender artifacts.

Usage:
    python scripts/train_content_recommender.py
"""

import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.content_recommender import ContentRecommender


def main() -> int:
    print("=" * 70)
    print("BOOK RECOMMENDER SYSTEM - CONTENT MODEL TRAINING (PHASE 3)")
    print("=" * 70)

    metadata_path = os.path.join(REPO_ROOT, "artifacts", "book_metadata_cosine.pkl")
    if not os.path.isfile(metadata_path):
        print(f"[-] ERROR: Source metadata file not found at: {metadata_path}")
        return 1

    print(f"[*] Loading book metadata from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        meta_df = pickle.load(f)

    print(f"  [+] Books in metadata: {len(meta_df)}")
    print(f"  [+] Metadata columns:  {list(meta_df.columns)}")

    print("\n[*] Initializing and fitting ContentRecommender (TF-IDF)...")
    recommender = ContentRecommender(ngram_range=(1, 2), min_df=1, author_weight=2)
    recommender.fit(meta_df)

    vocab_size = len(recommender.vectorizer.vocabulary_)
    matrix_shape = recommender.tfidf_matrix.shape
    print(f"  [+] TF-IDF Matrix Shape: {matrix_shape}")
    print(f"  [+] Vocabulary Size:     {vocab_size} unique n-grams")

    artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
    vec_path = os.path.join(artifacts_dir, "content_tfidf_vectorizer.pkl")
    mat_path = os.path.join(artifacts_dir, "content_tfidf_matrix.pkl")
    meta_out_path = os.path.join(artifacts_dir, "content_book_metadata.pkl")

    print(f"\n[*] Serializing Content artifacts to: {artifacts_dir}")
    with open(vec_path, "wb") as f:
        pickle.dump(recommender.vectorizer, f)
    with open(mat_path, "wb") as f:
        pickle.dump(recommender.tfidf_matrix, f)
    with open(meta_out_path, "wb") as f:
        pickle.dump(recommender.metadata_df, f)

    print(f"  [+] Saved {os.path.basename(vec_path)}:      {os.path.getsize(vec_path) / 1024:.1f} KB")
    print(f"  [+] Saved {os.path.basename(mat_path)}:          {os.path.getsize(mat_path) / 1024:.1f} KB")
    print(f"  [+] Saved {os.path.basename(meta_out_path)}:   {os.path.getsize(meta_out_path) / 1024:.1f} KB")

    # Sanity sample query
    sample_title = "Harry Potter and the Sorcerer's Stone (Book 1)"
    print(f"\n--- Content Sanity Query for: '{sample_title}' ---")
    recs = recommender.recommend(sample_title, n_recommendations=5)
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['title']} (Author: {r.get('author')}) | Similarity: {r['content_similarity']}")

    print("\n[SUCCESS] Content model training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
