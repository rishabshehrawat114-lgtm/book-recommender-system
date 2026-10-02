"""Hybrid Recommender combining Collaborative Filtering and Content-Based Filtering.

This module implements a weighted hybrid recommender that merges collaborative
rating patterns with content-based TF-IDF textual similarities.
"""

from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
from src.models.content_recommender import ContentRecommender
from src.models.recommender import BookRecommender


def min_max_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Apply min-max normalization to project scores into [0.0, 1.0].

    Args:
        scores: 1D array of scores.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized score array in [0.0, 1.0].
    """
    valid_mask = np.isfinite(scores)
    if not np.any(valid_mask):
        return np.zeros_like(scores)

    min_val = np.min(scores[valid_mask])
    max_val = np.max(scores[valid_mask])

    rng = max_val - min_val
    if rng <= eps:
        # All finite scores are identical
        out = np.zeros_like(scores)
        out[valid_mask] = 1.0
        return out

    out = (scores - min_val) / rng
    out[~valid_mask] = -np.inf
    return out


class HybridRecommender:
    """Weighted hybrid recommender system combining collaborative and content signals.

    Formula:
        hybrid_score = alpha * norm(collab_score) + (1.0 - alpha) * norm(content_score)

    Attributes:
        collab_recommender (BookRecommender): Fitted collaborative filtering model.
        content_recommender (ContentRecommender): Fitted content-based model.
        alpha (float): Weight assigned to collaborative filtering (between 0.0 and 1.0).
    """

    def __init__(
        self,
        collab_recommender: BookRecommender,
        content_recommender: ContentRecommender,
        default_alpha: float = 0.7,
    ) -> None:
        """Initialize HybridRecommender.

        Args:
            collab_recommender: Fitted BookRecommender instance.
            content_recommender: Fitted ContentRecommender instance.
            default_alpha: Default collaborative weight in [0.0, 1.0] (default: 0.7).
        """
        self.collab_recommender = collab_recommender
        self.content_recommender = content_recommender
        self.default_alpha = self._validate_alpha(default_alpha)

        # Confirm alignment between collaborative and content item vocabularies
        self.collab_books = set(collab_recommender.get_supported_books())
        self.content_books = set(content_recommender.get_supported_books())
        self.common_books = list(collab_recommender.get_supported_books())
        self._title_to_idx = {t: i for i, t in enumerate(self.common_books)}

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        """Ensure alpha lies strictly in [0.0, 1.0]."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        return float(alpha)

    def recommend_for_book(
        self,
        book_title: str,
        n_recommendations: int = 5,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate hybrid item-to-item recommendations for a single book.

        Args:
            book_title: Query book title.
            n_recommendations: Number of recommendations to return.
            alpha: Optional override for collaborative weight alpha.

        Returns:
            List of recommendation dictionaries with hybrid, collaborative, and content scores.
        """
        effective_alpha = self._validate_alpha(alpha) if alpha is not None else self.default_alpha

        if book_title not in self._title_to_idx:
            # Cold-start book scenario: if book is not in collaborative catalog, delegate to content
            if self.content_recommender.is_fitted and book_title in self.content_books:
                content_recs = self.content_recommender.recommend(book_title, n_recommendations)
                for r in content_recs:
                    r["hybrid_score"] = r["content_similarity"]
                    r["collab_score"] = 0.0
                    r["content_score"] = r["content_similarity"]
                return content_recs
            raise ValueError(f"Book '{book_title}' is not recognized in the recommender catalog.")

        n_catalog = len(self.common_books)
        collab_scores = np.zeros(n_catalog, dtype=np.float32)
        content_scores = np.zeros(n_catalog, dtype=np.float32)

        # 1. Collaborative item-item scores
        if effective_alpha > 0.0:
            raw_collab = self.collab_recommender.recommend(
                book_title,
                n_recommendations=n_catalog - 1,
            )
            for item in raw_collab:
                t = item["title"]
                if t in self._title_to_idx:
                    collab_scores[self._title_to_idx[t]] = float(item["cosine_similarity"])

        # 2. Content item-item scores
        if effective_alpha < 1.0:
            raw_content = self.content_recommender.recommend(
                book_title,
                n_recommendations=n_catalog - 1,
            )
            for item in raw_content:
                t = item["title"]
                if t in self._title_to_idx:
                    content_scores[self._title_to_idx[t]] = float(item["content_similarity"])

        # 3. Min-Max normalization
        norm_collab = min_max_normalize(collab_scores)
        norm_content = min_max_normalize(content_scores)

        # 4. Weighted combination
        hybrid_scores = effective_alpha * norm_collab + (1.0 - effective_alpha) * norm_content

        # 5. Mask query book itself
        query_idx = self._title_to_idx[book_title]
        hybrid_scores[query_idx] = -np.inf

        # 6. Rank top-N candidates
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]

        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            candidate_title = self.common_books[idx]
            rec_item = {
                "title": candidate_title,
                "hybrid_score": round(float(hybrid_scores[idx]), 6),
                "collab_score": round(float(collab_scores[idx]), 6),
                "content_score": round(float(content_scores[idx]), 6),
                "norm_collab_score": round(float(norm_collab[idx]), 6),
                "norm_content_score": round(float(norm_content[idx]), 6),
            }

            if self.content_recommender.metadata_df is not None:
                meta = self.content_recommender.metadata_df
                match = meta[meta["title"] == candidate_title]
                if not match.empty:
                    row = match.iloc[0]
                    if "author" in row:
                        rec_item["author"] = str(row["author"])
                    if "year" in row:
                        rec_item["year"] = str(row["year"])
                    if "publisher" in row:
                        rec_item["publisher"] = str(row["publisher"])
                    if "img_url" in row:
                        rec_item["img_url"] = str(row["img_url"])

            recommendations.append(rec_item)

        return recommendations

    def recommend_for_user_interactions(
        self,
        user_ratings_df: pd.DataFrame,
        n_recommendations: int = 5,
        alpha: Optional[float] = None,
        positive_threshold: float = 7.0,
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations for a user given their explicit training interactions.

        Args:
            user_ratings_df: DataFrame with 'title' and 'rating' columns for the user.
            n_recommendations: Number of recommendations to return.
            alpha: Optional override for collaborative weight alpha.
            positive_threshold: Rating threshold to build content preference profile.

        Returns:
            List of recommendation dictionaries.
        """
        effective_alpha = self._validate_alpha(alpha) if alpha is not None else self.default_alpha
        n_catalog = len(self.common_books)

        user_titles = set(user_ratings_df["title"])
        user_indices = [self._title_to_idx[t] for t in user_titles if t in self._title_to_idx]

        collab_scores = np.zeros(n_catalog, dtype=np.float32)
        content_scores = np.zeros(n_catalog, dtype=np.float32)

        # 1. Collaborative user profile scoring
        if effective_alpha > 0.0 and self.collab_recommender.sparse_matrix is not None:
            # Compute user vector against item matrix
            user_vec = np.zeros(n_catalog, dtype=np.float32)
            for _, row in user_ratings_df.iterrows():
                t = row["title"]
                if t in self._title_to_idx:
                    user_vec[self._title_to_idx[t]] = float(row["rating"])

            # Compute candidate collaborative scores via pairwise similarity dot product
            # matrix @ user_vec
            sim_mat = 1.0 - self.collab_recommender.model._fit_X.dot(self.collab_recommender.model._fit_X.T).toarray()
            # If using sparse matrix dot product:
            collab_scores = self.collab_recommender.sparse_matrix.dot(
                self.collab_recommender.sparse_matrix.T.dot(user_vec)
            )

        # 2. Content user profile scoring
        if effective_alpha < 1.0:
            pos_df = user_ratings_df[user_ratings_df["rating"] >= positive_threshold]
            if pos_df.empty:
                pos_df = user_ratings_df

            liked_titles = [t for t in pos_df["title"] if t in self._title_to_idx]
            liked_ratings = [float(r) for t, r in zip(pos_df["title"], pos_df["rating"]) if t in self._title_to_idx]

            content_scores = self.content_recommender.predict_user_scores(
                liked_titles=liked_titles,
                weights=liked_ratings,
            )

        # 3. Min-Max normalize
        norm_collab = min_max_normalize(collab_scores)
        norm_content = min_max_normalize(content_scores)

        # 4. Weighted combination
        hybrid_scores = effective_alpha * norm_collab + (1.0 - effective_alpha) * norm_content

        # 5. Mask out all books already interacted with by the user
        for idx in user_indices:
            hybrid_scores[idx] = -np.inf

        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]

        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            candidate_title = self.common_books[idx]
            rec_item = {
                "title": candidate_title,
                "hybrid_score": round(float(hybrid_scores[idx]), 6),
                "collab_score": round(float(collab_scores[idx]), 6),
                "content_score": round(float(content_scores[idx]), 6),
                "norm_collab_score": round(float(norm_collab[idx]), 6),
                "norm_content_score": round(float(norm_content[idx]), 6),
            }

            if self.content_recommender.metadata_df is not None:
                meta = self.content_recommender.metadata_df
                match = meta[meta["title"] == candidate_title]
                if not match.empty:
                    row = match.iloc[0]
                    if "author" in row:
                        rec_item["author"] = str(row["author"])
                    if "year" in row:
                        rec_item["year"] = str(row["year"])
                    if "publisher" in row:
                        rec_item["publisher"] = str(row["publisher"])
                    if "img_url" in row:
                        rec_item["img_url"] = str(row["img_url"])

            recommendations.append(rec_item)

        return recommendations
