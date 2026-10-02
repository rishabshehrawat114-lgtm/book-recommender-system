"""Artifact serialization and deserialization utilities.

Provides functions to save and load recommendation models, sparse matrices,
and metadata while preserving the original baseline artifacts.
"""

import os
import pickle
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.recommender import BookRecommender


def save_artifacts(
    recommender: BookRecommender,
    sparse_matrix: csr_matrix,
    item_index: pd.Index,
    metadata_df: Optional[pd.DataFrame] = None,
    output_dir: str = "artifacts",
    model_name: str = "model_cosine.pkl",
    matrix_name: str = "book_pivot_sparse.pkl",
    books_name_file: str = "books_name_cosine.pkl",
    metadata_name: str = "book_metadata_cosine.pkl",
) -> Dict[str, str]:
    """Serialize the new recommendation artifacts without overwriting original files.

    Args:
        recommender: Fitted BookRecommender instance.
        sparse_matrix: Fitted interaction CSR matrix.
        item_index: Index of book titles.
        metadata_df: Cleaned metadata DataFrame.
        output_dir: Directory to save serialized files.
        model_name: Filename for the cosine model artifact.
        matrix_name: Filename for the sparse matrix artifact.
        books_name_file: Filename for the book titles index.
        metadata_name: Filename for book metadata.

    Returns:
        Dictionary of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {
        "model": os.path.join(output_dir, model_name),
        "matrix": os.path.join(output_dir, matrix_name),
        "books_name": os.path.join(output_dir, books_name_file),
    }

    # 1. Save recommender model
    with open(paths["model"], "wb") as f:
        pickle.dump(recommender, f)

    # 2. Save sparse matrix payload (matrix + indices)
    sparse_payload = {
        "sparse_matrix": sparse_matrix,
        "item_index": item_index,
    }
    with open(paths["matrix"], "wb") as f:
        pickle.dump(sparse_payload, f)

    # 3. Save books name index
    with open(paths["books_name"], "wb") as f:
        pickle.dump(item_index, f)

    # 4. Save metadata if provided (deduplicated title -> details mapping)
    if metadata_df is not None:
        paths["metadata"] = os.path.join(output_dir, metadata_name)
        # Create lightweight mapping to replace heavy 3.8MB DataFrame
        meta_dedup = metadata_df.drop_duplicates("title")[
            [c for c in ["title", "author", "year", "publisher", "img_url"] if c in metadata_df.columns]
        ]
        with open(paths["metadata"], "wb") as f:
            pickle.dump(meta_dedup, f)

    return paths


def load_artifacts(
    artifacts_dir: str = "artifacts",
    model_name: str = "model_cosine.pkl",
) -> Tuple[BookRecommender, Optional[pd.DataFrame]]:
    """Load the cosine recommendation model and associated metadata.

    Args:
        artifacts_dir: Path to directory containing the artifacts.
        model_name: Name of the model file to load.

    Returns:
        Tuple of (BookRecommender instance, metadata DataFrame or None).

    Raises:
        FileNotFoundError: If the model artifact cannot be found.
    """
    model_path = os.path.join(artifacts_dir, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    with open(model_path, "rb") as f:
        recommender: BookRecommender = pickle.load(f)

    metadata_path = os.path.join(artifacts_dir, "book_metadata_cosine.pkl")
    metadata_df = None
    if os.path.isfile(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata_df = pickle.load(f)

    return recommender, metadata_df


def load_recommendation_service(
    artifacts_dir: str = "artifacts",
    default_alpha: float = 0.5,
) -> Tuple[Any, pd.DataFrame]:
    """Load all necessary model artifacts and construct a RecommendationService instance.

    Args:
        artifacts_dir: Path to directory containing the serialized artifacts.
        default_alpha: Default collaborative weight for the service.

    Returns:
        Tuple of (RecommendationService instance, ratings DataFrame).

    Raises:
        FileNotFoundError: If any required artifact file is missing.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from src.models.content_recommender import ContentRecommender
    from src.models.recommendation_service import RecommendationService

    required_files = [
        "model_cosine.pkl",
        "content_tfidf_vectorizer.pkl",
        "content_tfidf_matrix.pkl",
        "content_book_metadata.pkl",
        "final_rating.pkl",
    ]

    for fname in required_files:
        fpath = os.path.join(artifacts_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required artifact '{fname}' not found in '{artifacts_dir}'. "
                f"Please ensure Phase 1 and Phase 3 models have been trained."
            )

    # 1. Collaborative Recommender
    with open(os.path.join(artifacts_dir, "model_cosine.pkl"), "rb") as f:
        collab_rec: BookRecommender = pickle.load(f)

    # 2. Content Recommender
    with open(os.path.join(artifacts_dir, "content_tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(artifacts_dir, "content_tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(artifacts_dir, "content_book_metadata.pkl"), "rb") as f:
        content_meta = pickle.load(f)

    content_rec = ContentRecommender()
    content_rec.vectorizer = vectorizer
    content_rec.tfidf_matrix = tfidf_matrix
    content_rec.metadata_df = content_meta
    content_rec.item_names = list(content_meta["title"])
    content_rec._title_to_idx = {t: i for i, t in enumerate(content_rec.item_names)}
    content_rec._dense_sim_matrix = cosine_similarity(tfidf_matrix, dense_output=True)
    np.fill_diagonal(content_rec._dense_sim_matrix, 0.0)
    content_rec._is_fitted = True

    # 3. User Ratings & Popularity Baseline
    with open(os.path.join(artifacts_dir, "final_rating.pkl"), "rb") as f:
        ratings_df: pd.DataFrame = pickle.load(f)

    explicit_ratings = ratings_df[ratings_df["rating"] > 0]
    popularity_ranking = list(explicit_ratings["title"].value_counts().index)

    service = RecommendationService(
        collab_recommender=collab_rec,
        content_recommender=content_rec,
        popularity_ranking=popularity_ranking,
        warm_user_threshold=3,
        default_alpha=default_alpha,
        cold_user_alpha=0.25,
    )

    return service, ratings_df
