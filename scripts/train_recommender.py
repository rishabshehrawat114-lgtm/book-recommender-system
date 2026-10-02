#!/usr/bin/env python
"""Standalone training pipeline script for Book Recommender System.

Usage:
    # Train using raw Book-Crossing CSV directory:
    python scripts/train_recommender.py --data-dir path/to/raw_csvs

    # Train using existing cleaned ratings artifact in the repo:
    python scripts/train_recommender.py --from-cleaned artifacts/final_rating.pkl

    # Or default (checks data/ directory, then falls back to artifacts/final_rating.pkl):
    python scripts/train_recommender.py
"""

import argparse
import os
import pickle
import sys
import pandas as pd

# Add repo root to sys.path so 'src' can be imported cleanly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocessing import prepare_dataset
from src.models.recommender import BookRecommender
from src.utils.artifacts import save_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the item-based collaborative filtering book recommender engine."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing raw CSVs (BX-Books.csv, BX-Users.csv, BX-Book-Ratings.csv).",
    )
    parser.add_argument(
        "--from-cleaned",
        type=str,
        default=None,
        help="Path to pre-cleaned ratings DataFrame pickle (e.g. artifacts/final_rating.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save new model artifacts (default: artifacts).",
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=200,
        help="Minimum ratings per active user threshold (default: 200).",
    )
    parser.add_argument(
        "--min-book-ratings",
        type=int,
        default=50,
        help="Minimum ratings per popular book threshold (default: 50).",
    )
    parser.add_argument(
        "--rating-strategy",
        type=str,
        choices=["explicit_only", "impute_implicit"],
        default="explicit_only",
        help="Rating handling strategy: 'explicit_only' (recommended) or 'impute_implicit'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("BOOK RECOMMENDER SYSTEM - TRAINING PIPELINE (PHASE 1)")
    print("=" * 70)

    # Determine input data source
    data_dir = args.data_dir
    cleaned_path = args.from_cleaned

    if data_dir is None and cleaned_path is None:
        # Check standard local data/ directory
        candidate_data_dir = os.path.join(REPO_ROOT, "data")
        candidate_cleaned = os.path.join(REPO_ROOT, "artifacts", "final_rating.pkl")

        if os.path.isdir(candidate_data_dir) and os.path.isfile(os.path.join(candidate_data_dir, "BX-Books.csv")):
            data_dir = candidate_data_dir
            print(f"[+] Found local raw data directory: {data_dir}")
        elif os.path.isfile(candidate_cleaned):
            cleaned_path = candidate_cleaned
            print(f"[+] Found existing cleaned ratings artifact: {cleaned_path}")
        else:
            print("[-] ERROR: No dataset found.")
            print("    Please provide either:")
            print("      --data-dir <path_to_raw_csv_directory>")
            print("      --from-cleaned <path_to_cleaned_ratings_pickle>")
            return 1

    # Load and prepare dataset
    cleaned_df = None
    if cleaned_path is not None:
        if not os.path.isfile(cleaned_path):
            print(f"[-] ERROR: Cleaned file not found: {cleaned_path}")
            return 1
        print(f"[*] Loading pre-cleaned ratings from: {cleaned_path}")
        with open(cleaned_path, "rb") as f:
            cleaned_df = pickle.load(f)
        input_location = cleaned_path
    else:
        print(f"[*] Processing raw Book-Crossing CSVs from: {data_dir}")
        input_location = data_dir

    print(f"[*] Applying rating strategy: '{args.rating_strategy}'")
    prep_result = prepare_dataset(
        data_dir=data_dir,
        cleaned_df=cleaned_df,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        rating_strategy=args.rating_strategy,
    )

    sparse_matrix = prep_result["sparse_matrix"]
    item_index = prep_result["item_index"]
    user_index = prep_result["user_index"]
    metadata_df = prep_result["metadata_df"]
    n_ratings = prep_result["n_ratings"]
    n_books = prep_result["n_books"]
    n_users = prep_result["n_users"]
    non_zero = prep_result["non_zero_interactions"]
    sparsity = prep_result["sparsity"]

    print("\n--- Pipeline Execution Metrics ---")
    print(f"  Input Dataset Location:     {input_location}")
    print(f"  Total Processed Ratings:    {n_ratings}")
    print(f"  Books in Matrix:            {n_books}")
    print(f"  Users in Matrix:            {n_users}")
    print(f"  Interaction Matrix Shape:   ({n_books}, {n_users})")
    print(f"  Observed Interactions:      {non_zero}")
    print(f"  Matrix Density:             {100.0 - sparsity:.2f}%")
    print(f"  Matrix Sparsity:            {sparsity:.2f}%")

    # Fit Cosine Recommender Model
    print("\n--- Fitting Model ---")
    print("  Algorithm:  brute")
    print("  Metric:     cosine")
    recommender = BookRecommender(metric="cosine", algorithm="brute")
    recommender.fit(sparse_matrix, item_index)
    print("  Status:     Fitted successfully.")

    # Save artifacts
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    print(f"\n--- Serializing Artifacts to '{output_dir}' ---")
    saved_paths = save_artifacts(
        recommender=recommender,
        sparse_matrix=sparse_matrix,
        item_index=item_index,
        metadata_df=metadata_df,
        output_dir=output_dir,
        model_name="model_cosine.pkl",
        matrix_name="book_pivot_sparse.pkl",
        books_name_file="books_name_cosine.pkl",
        metadata_name="book_metadata_cosine.pkl",
    )

    for key, path in saved_paths.items():
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  [+] Saved {key:<12}: {path} ({size_kb:.1f} KB)")

    # Sample Sanity Check
    print("\n--- Sample Sanity Query ---")
    sample_candidates = [b for b in item_index if "Harry Potter" in b]
    sample_title = sample_candidates[0] if sample_candidates else item_index[0]
    print(f"  Query Book: '{sample_title}'")
    recs = recommender.recommend(sample_title, n_recommendations=5)
    for i, r in enumerate(recs, 1):
        print(f"    {i}. {r['title']} | Cosine Dist: {r['cosine_distance']} | Similarity: {r['cosine_similarity']}")

    print("\n[SUCCESS] Training pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
