"""Data loading, cleaning, and matrix preparation for Collaborative Filtering.

This module implements the complete data transformation pipeline for the
Book-Crossing dataset:
    Raw Data -> Cleaning -> User Thresholding -> Item Thresholding ->
    Deduplication -> Rating Handling -> Sparse Interaction Matrix
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Book-Crossing CSV files from a directory.

    Expected files:
        - BX-Books.csv
        - BX-Users.csv
        - BX-Book-Ratings.csv

    Args:
        data_dir: Directory path containing the CSV files.

    Returns:
        Tuple of (books_df, users_df, ratings_df).

    Raises:
        FileNotFoundError: If any of the required CSV files are missing.
    """
    books_path = os.path.join(data_dir, "BX-Books.csv")
    users_path = os.path.join(data_dir, "BX-Users.csv")
    ratings_path = os.path.join(data_dir, "BX-Book-Ratings.csv")

    for path, name in [
        (books_path, "BX-Books.csv"),
        (users_path, "BX-Users.csv"),
        (ratings_path, "BX-Book-Ratings.csv"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required dataset file '{name}' was not found in: {data_dir}"
            )

    books = pd.read_csv(
        books_path,
        sep=";",
        on_bad_lines="skip",
        encoding="latin-1",
        low_memory=False,
    )
    users = pd.read_csv(
        users_path,
        sep=";",
        on_bad_lines="skip",
        encoding="latin-1",
        low_memory=False,
    )
    ratings = pd.read_csv(
        ratings_path,
        sep=";",
        on_bad_lines="skip",
        encoding="latin-1",
        low_memory=False,
    )

    # Standardize column names
    books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]].copy()
    books.rename(
        columns={
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
            "Image-URL-L": "img_url",
        },
        inplace=True,
    )

    ratings.rename(
        columns={
            "User-ID": "user_id",
            "Book-Rating": "rating",
        },
        inplace=True,
    )

    return books, users, ratings


def filter_active_users(ratings: pd.DataFrame, min_ratings: int = 200) -> pd.DataFrame:
    """Filter ratings to keep only active users with at least `min_ratings`.

    Args:
        ratings: DataFrame containing 'user_id' and 'rating'.
        min_ratings: Minimum number of ratings required per user.

    Returns:
        Filtered ratings DataFrame.
    """
    user_counts = ratings["user_id"].value_counts()
    active_user_ids = user_counts[user_counts > min_ratings].index
    return ratings[ratings["user_id"].isin(active_user_ids)].copy()


def merge_ratings_and_books(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Merge filtered ratings with book metadata on ISBN.

    Args:
        ratings: Filtered ratings DataFrame.
        books: Cleaned books metadata DataFrame.

    Returns:
        Merged DataFrame containing rating and book details.
    """
    return ratings.merge(books, on="ISBN")


def filter_popular_books(merged_df: pd.DataFrame, min_ratings: int = 50) -> pd.DataFrame:
    """Filter books having at least `min_ratings` total ratings across active users.

    Args:
        merged_df: Merged ratings and books DataFrame.
        min_ratings: Minimum number of ratings required per book title.

    Returns:
        DataFrame filtered to popular books with an added 'num_of_rating' column.
    """
    num_rating = merged_df.groupby("title")["rating"].count().reset_index()
    num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)
    merged_with_counts = merged_df.merge(num_rating, on="title")
    return merged_with_counts[merged_with_counts["num_of_rating"] >= min_ratings].copy()


def deduplicate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate user-book rating pairs, retaining the first occurrence.

    Args:
        df: DataFrame containing 'user_id' and 'title'.

    Returns:
        Deduplicated DataFrame.
    """
    return df.drop_duplicates(subset=["user_id", "title"]).copy()


def handle_ratings(
    df: pd.DataFrame,
    strategy: str = "explicit_only",
    implicit_rating_value: Optional[float] = None,
) -> pd.DataFrame:
    """Handle explicit vs. implicit ratings in the Book-Crossing dataset.

    Technical Justification:
    In the Book-Crossing dataset:
      - Ratings 1–10: Explicit ratings expressing user sentiment.
      - Rating 0: Implicit interaction (e.g. read, shelf addition) without explicit score.
      - Unrated pairs: Truly missing entries (unobserved).

    Treating rating 0 as a 0-star quality rating is mathematically flawed because:
    1. Rating 0 denotes interaction, not extreme dissatisfaction.
    2. In standard sparse matrices, 0-value ratings are indistinguishable from unobserved cells.
    3. In cosine similarity, unobserved interactions contribute 0 to the dot product.

    Strategies:
      - 'explicit_only' (recommended default): Retains only explicit ratings (1–10).
        Implicit interactions are excluded from explicit collaborative similarity so
        they are not conflated with negative sentiment or unobserved items.
      - 'impute_implicit': Preserves explicit ratings (1–10) and assigns implicit interactions
        (rating == 0) a designated positive interaction value (default: median explicit rating,
        typically 8.0 or a specified confidence score).

    Args:
        df: DataFrame with 'rating' column.
        strategy: 'explicit_only' or 'impute_implicit'.
        implicit_rating_value: Value to assign if strategy is 'impute_implicit'.

    Returns:
        Processed DataFrame with appropriately handled ratings.
    """
    processed = df.copy()

    if strategy == "explicit_only":
        # Keep only positive explicit ratings
        processed = processed[processed["rating"] > 0].copy()
    elif strategy == "impute_implicit":
        if implicit_rating_value is None:
            # Use median of explicit ratings as a neutral engagement prior
            explicit_ratings = processed[processed["rating"] > 0]["rating"]
            implicit_rating_value = float(explicit_ratings.median()) if len(explicit_ratings) > 0 else 5.0

        processed.loc[processed["rating"] == 0, "rating"] = implicit_rating_value
    else:
        raise ValueError(f"Unknown rating strategy '{strategy}'. Use 'explicit_only' or 'impute_implicit'.")

    return processed


def create_interaction_matrix(
    df: pd.DataFrame,
    item_col: str = "title",
    user_col: str = "user_id",
    rating_col: str = "rating",
) -> Tuple[csr_matrix, pd.Index, pd.Index, pd.DataFrame]:
    """Construct an item-user pivot table and return its sparse representation.

    Args:
        df: Cleaned ratings DataFrame.
        item_col: Column name representing items (rows).
        user_col: Column name representing users (columns).
        rating_col: Column name representing rating values.

    Returns:
        Tuple of:
          - csr_matrix: Compressed Sparse Row matrix of shape (n_items, n_users).
          - item_index: Index of item names (length n_items).
          - user_index: Index of user IDs (length n_users).
          - pivot_df: Dense DataFrame representation (with fillna(0)) for compatibility.
    """
    pivot_df = df.pivot_table(index=item_col, columns=user_col, values=rating_col)
    pivot_df.fillna(0, inplace=True)

    item_index = pivot_df.index
    user_index = pivot_df.columns
    sparse_mat = csr_matrix(pivot_df.values, dtype=np.float32)

    return sparse_mat, item_index, user_index, pivot_df


def prepare_dataset(
    data_dir: Optional[str] = None,
    cleaned_df: Optional[pd.DataFrame] = None,
    min_user_ratings: int = 200,
    min_book_ratings: int = 50,
    rating_strategy: str = "explicit_only",
    implicit_rating_value: Optional[float] = None,
) -> Dict[str, Union[csr_matrix, pd.Index, pd.DataFrame, int, float]]:
    """Execute the end-to-end data preparation pipeline.

    Supports either raw CSV input (via `data_dir`) or pre-cleaned DataFrame (`cleaned_df`).

    Returns:
        Dictionary containing:
          - 'sparse_matrix': csr_matrix for model training
          - 'item_index': pd.Index of book titles
          - 'user_index': pd.Index of user IDs
          - 'pivot_df': Dense DataFrame of pivot table
          - 'final_df': Processed ratings DataFrame
          - 'n_ratings': Total ratings processed
          - 'n_books': Number of unique books
          - 'n_users': Number of unique users
          - 'sparsity': Percentage sparsity of the matrix
    """
    if data_dir is not None:
        books, users, ratings = load_raw_data(data_dir)
        filtered_ratings = filter_active_users(ratings, min_ratings=min_user_ratings)
        merged = merge_ratings_and_books(filtered_ratings, books)
        popular = filter_popular_books(merged, min_ratings=min_book_ratings)
        deduped = deduplicate_ratings(popular)
    elif cleaned_df is not None:
        deduped = deduplicate_ratings(cleaned_df)
    else:
        raise ValueError("Either 'data_dir' or 'cleaned_df' must be provided.")

    processed = handle_ratings(
        deduped,
        strategy=rating_strategy,
        implicit_rating_value=implicit_rating_value,
    )

    sparse_mat, item_index, user_index, pivot_df = create_interaction_matrix(
        processed,
        item_col="title",
        user_col="user_id",
        rating_col="rating",
    )

    total_cells = sparse_mat.shape[0] * sparse_mat.shape[1]
    non_zero = sparse_mat.nnz
    sparsity = (1.0 - (non_zero / total_cells)) * 100.0 if total_cells > 0 else 0.0

    return {
        "sparse_matrix": sparse_mat,
        "item_index": item_index,
        "user_index": user_index,
        "pivot_df": pivot_df,
        "final_df": processed,
        "metadata_df": deduped,  # Retains all book metadata and poster URLs
        "n_ratings": len(processed),
        "n_books": sparse_mat.shape[0],
        "n_users": sparse_mat.shape[1],
        "non_zero_interactions": non_zero,
        "sparsity": sparsity,
    }
