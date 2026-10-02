"""Unified Recommendation Service with Cold-Start Handling.

Orchestrates Collaborative, Content, Hybrid, and Cold-Start policies:
  - Scenario A: New user (no history) -> Popularity baseline or Seed-book Content recommendations.
  - Scenario B: Limited-history user (1-2 ratings) -> Content-weighted hybrid (alpha=0.25).
  - Scenario C: Warm user (>=3 ratings) -> Standard hybrid recommender (alpha=0.7).
  - Scenario D: New/unseen book -> Content inference from text metadata, with popular fallback.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.models.content_recommender import ContentRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.models.recommender import BookRecommender


class RecommendationService:
    """High-level service interface for book recommendations and cold-start handling."""

    def __init__(
        self,
        collab_recommender: BookRecommender,
        content_recommender: ContentRecommender,
        popularity_ranking: Optional[List[str]] = None,
        warm_user_threshold: int = 3,
        default_alpha: float = 0.7,
        cold_user_alpha: float = 0.25,
    ) -> None:
        """Initialize RecommendationService.

        Args:
            collab_recommender: Fitted collaborative recommender.
            content_recommender: Fitted content recommender.
            popularity_ranking: List of book titles sorted by training popularity.
            warm_user_threshold: Minimum ratings to be considered a warm user (default: 3).
            default_alpha: Collaborative weight for warm users (default: 0.7).
            cold_user_alpha: Collaborative weight for limited-history users (default: 0.25).
        """
        self.collab_recommender = collab_recommender
        self.content_recommender = content_recommender
        self.hybrid_recommender = HybridRecommender(
            collab_recommender=collab_recommender,
            content_recommender=content_recommender,
            default_alpha=default_alpha,
        )
        self.popularity_ranking = popularity_ranking or collab_recommender.get_supported_books()[:50]
        self.warm_user_threshold = warm_user_threshold
        self.default_alpha = default_alpha
        self.cold_user_alpha = cold_user_alpha

    def recommend_for_book(
        self,
        book_title: str,
        n_recommendations: int = 5,
        method: str = "hybrid",
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend books similar to a given title.

        Args:
            book_title: Title of query book.
            n_recommendations: Number of recommendations.
            method: 'hybrid', 'collaborative', or 'content'.
            alpha: Custom alpha for hybrid method.

        Returns:
            List of recommendations.
        """
        if method == "collaborative":
            return self.collab_recommender.recommend(book_title, n_recommendations)
        elif method == "content":
            return self.content_recommender.recommend(book_title, n_recommendations)
        elif method == "hybrid":
            return self.hybrid_recommender.recommend_for_book(
                book_title,
                n_recommendations=n_recommendations,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'hybrid', 'collaborative', or 'content'.")

    def recommend_for_new_user(
        self,
        seed_books: Optional[List[str]] = None,
        n_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Scenario A: Recommend for a brand new user with 0 interaction history.

        Policy:
          - If user selects seed book(s): return content recommendations based on seeds.
          - If no seeds: return top popular catalog books.

        Args:
            seed_books: Optional list of books the new user likes.
            n_recommendations: Number of recommendations.

        Returns:
            List of recommendation dictionaries with metadata.
        """
        if seed_books:
            valid_seeds = [b for b in seed_books if b in self.content_recommender.get_supported_books()]
            if valid_seeds:
                scores = self.content_recommender.predict_user_scores(valid_seeds)
                # Mask seed books themselves
                catalog = self.content_recommender.get_supported_books()
                title_to_idx = {t: i for i, t in enumerate(catalog)}
                for s in valid_seeds:
                    scores[title_to_idx[s]] = -np.inf

                top_indices = np.argsort(scores)[::-1][:n_recommendations]
                recs: List[Dict[str, Any]] = []
                for idx in top_indices:
                    title = catalog[idx]
                    item = {
                        "title": title,
                        "strategy": "seed_content_fallback",
                        "score": round(float(scores[idx]), 6),
                    }
                    self._enrich_item_metadata(item, title)
                    recs.append(item)
                return recs

        # Fallback to popularity baseline
        recs = []
        for title in self.popularity_ranking[:n_recommendations]:
            item = {
                "title": title,
                "strategy": "popularity_fallback",
                "score": 1.0,
            }
            self._enrich_item_metadata(item, title)
            recs.append(item)
        return recs

    def _enrich_item_metadata(self, item: Dict[str, Any], title: str) -> None:
        """Enrich a recommendation dictionary with catalog metadata if available."""
        if self.content_recommender.metadata_df is not None:
            meta = self.content_recommender.metadata_df
            match = meta[meta["title"] == title]
            if not match.empty:
                row = match.iloc[0]
                if "author" in row and "author" not in item:
                    item["author"] = str(row["author"])
                if "year" in row and "year" not in item:
                    item["year"] = str(row["year"])
                if "publisher" in row and "publisher" not in item:
                    item["publisher"] = str(row["publisher"])
                if "img_url" in row and "img_url" not in item:
                    item["img_url"] = str(row["img_url"])

    def recommend_for_user(
        self,
        user_ratings_df: pd.DataFrame,
        n_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Adaptive recommendation based on user history depth.

        Policy:
          - 0 ratings: Scenario A (New User -> Popularity).
          - 1-2 ratings: Scenario B (Limited History -> Content-heavy hybrid alpha=0.25).
          - >=3 ratings: Scenario C (Warm User -> Full hybrid alpha=0.70).
        """
        n_ratings = len(user_ratings_df)

        if n_ratings == 0:
            return self.recommend_for_new_user(n_recommendations=n_recommendations)

        if n_ratings < self.warm_user_threshold:
            # Limited history: use cold_user_alpha (0.25)
            alpha_to_use = self.cold_user_alpha
            strategy_label = "limited_history_hybrid"
        else:
            # Warm user: use default_alpha (0.70)
            alpha_to_use = self.default_alpha
            strategy_label = "warm_user_hybrid"

        recs = self.hybrid_recommender.recommend_for_user_interactions(
            user_ratings_df=user_ratings_df,
            n_recommendations=n_recommendations,
            alpha=alpha_to_use,
        )

        for r in recs:
            r["strategy"] = strategy_label

        return recs

    def recommend_for_new_book(
        self,
        title: str,
        author: str = "",
        publisher: str = "",
        n_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Scenario D: Recommend catalog books similar to an unseen book.

        Policy:
          - If metadata is valid: use TF-IDF inference.
          - If metadata is insufficient: graceful fallback to popular books.
        """
        content_recs = self.content_recommender.recommend_for_new_book(
            title=title,
            author=author,
            publisher=publisher,
            n_recommendations=n_recommendations,
        )

        if content_recs:
            for r in content_recs:
                r["strategy"] = "new_book_content_inference"
            return content_recs

        # Fallback if metadata was empty
        recs = []
        for t in self.popularity_ranking[:n_recommendations]:
            item = {
                "title": t,
                "strategy": "new_book_metadata_fallback_popularity",
                "score": 1.0,
            }
            self._enrich_item_metadata(item, t)
            recs.append(item)
        return recs
