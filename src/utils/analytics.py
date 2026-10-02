"""Dataset analytics and descriptive insights utilities (Phase 5).

Computes verified statistics from the project interaction dataset, including
explicit rating distributions, user activity profiles, and matrix sparsity.
"""

from typing import Any, Dict, List
import pandas as pd


def compute_dataset_analytics(
    ratings_df: pd.DataFrame,
    n_catalog_books: int = 742,
) -> Dict[str, Any]:
    """Compute descriptive statistics and distributions from ratings DataFrame.

    Args:
        ratings_df: DataFrame containing 'user_id', 'title', 'rating', 'author', 'publisher'.
        n_catalog_books: Total number of catalog books in matrix.

    Returns:
        Dictionary of dataset metrics and summary tables.
    """
    total_rows = len(ratings_df)
    total_users = ratings_df["user_id"].nunique()
    total_books = ratings_df["title"].nunique()

    explicit_df = ratings_df[ratings_df["rating"] > 0]
    explicit_count = len(explicit_df)
    explicit_users_count = explicit_df["user_id"].nunique()

    # Matrix Sparsity calculation
    theoretical_cells = n_catalog_books * explicit_users_count
    sparsity_pct = (1.0 - (explicit_count / theoretical_cells)) * 100.0 if theoretical_cells > 0 else 0.0

    # User activity stats
    user_counts = explicit_df.groupby("user_id")["rating"].count()
    user_stats = {
        "mean": round(float(user_counts.mean()), 2) if not user_counts.empty else 0.0,
        "median": float(user_counts.median()) if not user_counts.empty else 0.0,
        "min": int(user_counts.min()) if not user_counts.empty else 0,
        "max": int(user_counts.max()) if not user_counts.empty else 0,
    }

    # Book rating stats
    book_counts = explicit_df.groupby("title")["rating"].count()
    book_stats = {
        "mean": round(float(book_counts.mean()), 2) if not book_counts.empty else 0.0,
        "median": float(book_counts.median()) if not book_counts.empty else 0.0,
        "min": int(book_counts.min()) if not book_counts.empty else 0,
        "max": int(book_counts.max()) if not book_counts.empty else 0,
    }

    # Rating value distribution (1 to 10)
    rating_dist = explicit_df["rating"].value_counts().sort_index().to_dict()

    # Most-rated books with author and publisher
    top_books_df = (
        explicit_df.groupby("title")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            author=("author", "first"),
            publisher=("publisher", "first"),
        )
        .reset_index()
        .sort_values(by="rating_count", ascending=False)
        .head(10)
    )
    top_books_df["avg_rating"] = top_books_df["avg_rating"].round(2)

    return {
        "total_rows": total_rows,
        "total_catalog_books": n_catalog_books,
        "total_unique_books": total_books,
        "total_unique_users": total_users,
        "explicit_ratings_count": explicit_count,
        "explicit_users_count": explicit_users_count,
        "sparsity_percentage": round(sparsity_pct, 2),
        "user_ratings_stats": user_stats,
        "book_ratings_stats": book_stats,
        "rating_distribution": rating_dist,
        "top_rated_books_df": top_books_df,
    }
