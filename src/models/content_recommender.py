"""Content-Based Recommender System using TF-IDF on Book Metadata.

This module implements a content-based recommendation engine that builds
textual representations from book titles, authors, and publishers, computes
TF-IDF vectors, and calculates cosine similarities between books and user profiles.
"""

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_metadata_text(text: Any) -> str:
    """Clean and normalize metadata string."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)
    # Replace HTML entities
    text_str = text_str.replace("&amp;", "and").replace("&lt;", "<").replace("&gt;", ">")
    # Clean redundant whitespace
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


class ContentRecommender:
    """Content-based recommender using TF-IDF on book metadata.

    Attributes:
        vectorizer (TfidfVectorizer): Scikit-learn TF-IDF vectorizer.
        tfidf_matrix (csr_matrix): L2-normalized TF-IDF document matrix.
        item_names (List[str]): List of book titles aligned with matrix rows.
        metadata_df (pd.DataFrame): DataFrame of book metadata.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        author_weight: int = 2,
    ) -> None:
        """Initialize ContentRecommender.

        Args:
            ngram_range: N-gram range for TfidfVectorizer (default: (1, 2)).
            min_df: Minimum document frequency for terms (default: 1).
            author_weight: Number of times to repeat author name in text soup
                           to weight author similarity appropriately (default: 2).
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.author_weight = author_weight

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self.tfidf_matrix: Optional[Any] = None
        self.item_names: List[str] = []
        self.metadata_df: Optional[pd.DataFrame] = None
        self._title_to_idx: Dict[str, int] = {}
        self._is_fitted: bool = False
        self._dense_sim_matrix: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def _build_soup(self, row: pd.Series) -> str:
        """Combine metadata fields into a normalized text representation."""
        title = clean_metadata_text(row.get("title", ""))
        author = clean_metadata_text(row.get("author", ""))
        publisher = clean_metadata_text(row.get("publisher", ""))

        # Repeat author for appropriate domain weighting
        author_tokens = " ".join([author] * self.author_weight) if author else ""
        return f"{title} {author_tokens} {publisher}".strip()

    def fit(self, metadata_df: pd.DataFrame) -> "ContentRecommender":
        """Fit TF-IDF vectorizer on book metadata DataFrame.

        Args:
            metadata_df: DataFrame with 'title', 'author', 'publisher' columns.

        Returns:
            self: The fitted ContentRecommender.
        """
        if "title" not in metadata_df.columns:
            raise ValueError("metadata_df must contain a 'title' column.")

        # Deduplicate titles to ensure 1-to-1 index mapping
        clean_df = metadata_df.drop_duplicates(subset=["title"]).copy().reset_index(drop=True)
        self.metadata_df = clean_df
        self.item_names = list(clean_df["title"])
        self._title_to_idx = {title: idx for idx, title in enumerate(self.item_names)}

        # Build text soup for each book
        soups = clean_df.apply(self._build_soup, axis=1)

        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)

        # Precompute dense pairwise cosine similarities for fast catalog queries (742 x 742 is ~4.4 MB)
        self._dense_sim_matrix = cosine_similarity(self.tfidf_matrix, dense_output=True)
        np.fill_diagonal(self._dense_sim_matrix, 0.0)

        self._is_fitted = True
        return self

    def recommend(
        self,
        book_title: str,
        n_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend books similar to a given book based on content metadata.

        Args:
            book_title: Title of query book in the catalog.
            n_recommendations: Number of recommendations to return.

        Returns:
            List of recommendation dictionaries with title, content_similarity, author, etc.
        """
        if not self._is_fitted or self._dense_sim_matrix is None:
            raise RuntimeError("ContentRecommender must be fitted before recommend().")

        if book_title not in self._title_to_idx:
            raise ValueError(
                f"Book '{book_title}' is not present in the catalog. "
                f"Total available books: {len(self.item_names)}."
            )

        idx = self._title_to_idx[book_title]
        sim_scores = self._dense_sim_matrix[idx]

        # Top-K candidate indices (self-match diagonal is already 0.0)
        top_indices = np.argsort(sim_scores)[::-1][:n_recommendations]

        recommendations: List[Dict[str, Any]] = []
        for rank_idx in top_indices:
            candidate_title = self.item_names[rank_idx]
            if candidate_title == book_title:
                continue

            sim_val = float(sim_scores[rank_idx])
            rec_item: Dict[str, Any] = {
                "title": candidate_title,
                "content_similarity": round(sim_val, 6),
            }

            if self.metadata_df is not None:
                row = self.metadata_df.iloc[rank_idx]
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

    def predict_user_scores(
        self,
        liked_titles: List[str],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compute content similarity score vector across all catalog books for a user profile.

        Args:
            liked_titles: List of book titles user liked (e.g. rating >= 7.0).
            weights: Optional rating weights for each liked book.

        Returns:
            1D numpy array of content affinity scores for every book in self.item_names.
        """
        if not self._is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("ContentRecommender must be fitted.")

        if not liked_titles:
            return np.zeros(len(self.item_names), dtype=np.float32)

        indices = [self._title_to_idx[t] for t in liked_titles if t in self._title_to_idx]
        if not indices:
            return np.zeros(len(self.item_names), dtype=np.float32)

        if weights is None or len(weights) != len(liked_titles):
            w = np.ones(len(indices), dtype=np.float32)
        else:
            w = np.array([weights[i] for i, t in enumerate(liked_titles) if t in self._title_to_idx], dtype=np.float32)

        w_sum = float(np.sum(w))
        if w_sum > 0:
            w = w / w_sum

        # Extract profile as weighted combination of TF-IDF vectors
        user_profile_vec = np.zeros(self.tfidf_matrix.shape[1], dtype=np.float32)
        for idx, weight in zip(indices, w):
            user_profile_vec += weight * self.tfidf_matrix.getrow(idx).toarray().ravel()

        norm = np.linalg.norm(user_profile_vec)
        if norm > 0:
            user_profile_vec = user_profile_vec / norm

        # Compute cosine similarity between all catalog books and user profile vector
        scores = self.tfidf_matrix.dot(user_profile_vec).ravel()
        return scores.astype(np.float32)

    def recommend_for_new_book(
        self,
        title: str,
        author: str = "",
        publisher: str = "",
        n_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend existing catalog books similar to a newly provided, unseen book.

        Args:
            title: Title of the new book.
            author: Author of the new book.
            publisher: Publisher of the new book.
            n_recommendations: Number of recommendations.

        Returns:
            List of recommendations from the catalog.
        """
        if not self._is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("ContentRecommender must be fitted.")

        title_clean = clean_metadata_text(title)
        author_clean = clean_metadata_text(author)
        publisher_clean = clean_metadata_text(publisher)

        # Handle empty/insufficient metadata gracefully
        if not title_clean and not author_clean:
            return []

        author_tokens = " ".join([author_clean] * self.author_weight) if author_clean else ""
        soup = f"{title_clean} {author_tokens} {publisher_clean}".strip()

        # Vectorize new book text using the fitted vocabulary
        new_vec = self.vectorizer.transform([soup])
        sim_scores = cosine_similarity(new_vec, self.tfidf_matrix).ravel()

        top_indices = np.argsort(sim_scores)[::-1][:n_recommendations]

        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            candidate_title = self.item_names[idx]
            sim_val = float(sim_scores[idx])

            rec_item: Dict[str, Any] = {
                "title": candidate_title,
                "content_similarity": round(sim_val, 6),
            }

            if self.metadata_df is not None:
                row = self.metadata_df.iloc[idx]
                if "author" in row:
                    rec_item["author"] = str(row["author"])
                if "year" in row:
                    rec_item["year"] = str(row["year"])
                if "img_url" in row:
                    rec_item["img_url"] = str(row["img_url"])

            recommendations.append(rec_item)

        return recommendations

    def get_supported_books(self) -> List[str]:
        """Return all book titles in the catalog."""
        return list(self.item_names)
