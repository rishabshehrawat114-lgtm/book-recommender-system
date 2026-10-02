"""Cosine-similarity item-based collaborative filtering recommender engine.

This module provides the core `BookRecommender` class using Scikit-Learn's
NearestNeighbors algorithm with brute-force search and cosine distance metric.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


class BookRecommender:
    """Item-based Collaborative Filtering Recommender using Cosine Distance.

    Attributes:
        metric (str): Distance metric ('cosine').
        algorithm (str): Search algorithm ('brute').
        model (NearestNeighbors): Underlying fitted Scikit-Learn estimator.
        item_names (pd.Index): Index of book titles corresponding to matrix rows.
        sparse_matrix (csr_matrix): Fitted item-user interaction matrix.
    """

    def __init__(
        self,
        metric: str = "cosine",
        algorithm: str = "brute",
        n_neighbors: int = 6,
    ) -> None:
        """Initialize the recommender model.

        Args:
            metric: Distance metric. Default is 'cosine'.
            algorithm: Neighbor search algorithm. Default is 'brute'.
            n_neighbors: Default number of neighbors (including the query book).
        """
        self.metric = metric
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(
            metric=self.metric,
            algorithm=self.algorithm,
        )
        self.item_names: Optional[pd.Index] = None
        self.sparse_matrix: Optional[csr_matrix] = None
        self._is_fitted: bool = False
        self._title_to_index: Dict[str, int] = {}

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def fit(
        self,
        sparse_matrix: Union[csr_matrix, np.ndarray],
        item_names: Union[pd.Index, List[str]],
    ) -> "BookRecommender":
        """Fit the NearestNeighbors model on an item-user interaction matrix.

        Args:
            sparse_matrix: Interaction matrix of shape (n_items, n_users).
                           Rows are items (books) and columns are users.
            item_names: Index or list of item titles corresponding to matrix rows.

        Returns:
            self: The fitted BookRecommender instance.

        Raises:
            ValueError: If matrix rows do not match item_names length.
        """
        if not issparse(sparse_matrix):
            sparse_matrix = csr_matrix(sparse_matrix)

        if sparse_matrix.shape[0] != len(item_names):
            raise ValueError(
                f"Matrix rows ({sparse_matrix.shape[0]}) must match item count ({len(item_names)})."
            )

        self.sparse_matrix = sparse_matrix
        self.item_names = pd.Index(item_names) if not isinstance(item_names, pd.Index) else item_names
        self._title_to_index = {title: idx for idx, title in enumerate(self.item_names)}

        self.model.fit(self.sparse_matrix)
        self._is_fitted = True
        return self

    def recommend(
        self,
        book_title: str,
        n_recommendations: int = 5,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Generate top-N book recommendations for a given book title.

        Formula:
            Cosine Distance d in [0, 2]
            Cosine Similarity s = 1.0 - d in [-1, 1]

        Args:
            book_title: Exact title of the book to find recommendations for.
            n_recommendations: Number of distinct recommended books to return.
            metadata_df: Optional DataFrame with 'title' and metadata columns
                         (e.g., 'author', 'year', 'img_url') to enrich results.

        Returns:
            List of recommendation dictionaries, each containing:
              - 'title': Recommended book title
              - 'cosine_distance': Float distance (lower = more similar)
              - 'cosine_similarity': Float similarity (higher = more similar)
              - Optional metadata fields ('author', 'year', 'img_url') if provided.

        Raises:
            RuntimeError: If called before model is fitted.
            ValueError: If `book_title` is not found in the model vocabulary.
        """
        if not self._is_fitted or self.sparse_matrix is None or self.item_names is None:
            raise RuntimeError("Model must be fitted before calling recommend().")

        if book_title not in self._title_to_index:
            raise ValueError(
                f"Book '{book_title}' is not present in the model index. "
                f"Total available books: {len(self.item_names)}."
            )

        book_idx = self._title_to_index[book_title]

        # Request n_recommendations + 1 to account for the query book itself
        n_query = min(n_recommendations + 1, self.sparse_matrix.shape[0])
        row_vec = self.sparse_matrix.getrow(book_idx)

        distances, indices = self.model.kneighbors(row_vec, n_neighbors=n_query)

        recommendations: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            candidate_title = self.item_names[idx]
            # Exclude the query book itself (self-match)
            if candidate_title == book_title:
                continue

            cos_dist = float(dist)
            cos_sim = float(1.0 - cos_dist)

            rec_item: Dict[str, Any] = {
                "title": candidate_title,
                "cosine_distance": round(cos_dist, 6),
                "cosine_similarity": round(cos_sim, 6),
            }

            # Enrich with metadata if available
            if metadata_df is not None and "title" in metadata_df.columns:
                match = metadata_df[metadata_df["title"] == candidate_title]
                if not match.empty:
                    first_row = match.iloc[0]
                    if "author" in first_row:
                        rec_item["author"] = str(first_row["author"])
                    if "year" in first_row:
                        rec_item["year"] = str(first_row["year"])
                    if "img_url" in first_row:
                        rec_item["img_url"] = str(first_row["img_url"])

            recommendations.append(rec_item)

            if len(recommendations) >= n_recommendations:
                break

        return recommendations

    def get_supported_books(self) -> List[str]:
        """Return the list of all book titles supported by the model."""
        if self.item_names is None:
            return []
        return list(self.item_names)

    def get_model_params(self) -> Dict[str, Any]:
        """Return configuration and fitting parameters."""
        return {
            "metric": self.metric,
            "algorithm": self.algorithm,
            "is_fitted": self._is_fitted,
            "n_items": len(self.item_names) if self.item_names is not None else 0,
            "n_features": self.sparse_matrix.shape[1] if self.sparse_matrix is not None else 0,
            "underlying_params": self.model.get_params(),
        }
