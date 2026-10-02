"""Unit tests for HybridRecommender and RecommendationService (Phase 3)."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.content_recommender import ContentRecommender
from src.models.recommender import BookRecommender
from src.models.hybrid_recommender import HybridRecommender, min_max_normalize
from src.models.recommendation_service import RecommendationService


class TestHybridRecommender(unittest.TestCase):
    """Test suite for HybridRecommender and cold-start RecommendationService."""

    def setUp(self):
        self.item_names = ["Book 1", "Book 2", "Book 3", "Book 4", "Book 5"]
        self.metadata = pd.DataFrame([
            {"title": "Book 1", "author": "Author Alpha", "publisher": "Pub Xenon"},
            {"title": "Book 2", "author": "Author Alpha", "publisher": "Pub Xenon"},  # Content similar to Book 1
            {"title": "Book 3", "author": "Author Beta", "publisher": "Pub Yankee"},
            {"title": "Book 4", "author": "Author Beta", "publisher": "Pub Yankee"},  # Content similar to Book 3
            {"title": "Book 5", "author": "Author Gamma", "publisher": "Pub Zulu"},
        ])

        # Collaborative matrix: Book 1 and Book 3 co-rated by users
        dense_collab = np.array([
            [10.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 9.0],
            [9.0, 8.0, 0.0, 0.0],  # Collab similar to Book 1
            [0.0, 0.0, 7.0, 8.0],
            [5.0, 5.0, 5.0, 5.0],
        ])
        sparse_mat = csr_matrix(dense_collab)

        self.collab_rec = BookRecommender(metric="cosine", algorithm="brute").fit(sparse_mat, self.item_names)
        self.content_rec = ContentRecommender().fit(self.metadata)
        self.hybrid = HybridRecommender(
            collab_recommender=self.collab_rec,
            content_recommender=self.content_rec,
            default_alpha=0.7,
        )

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            self.hybrid._validate_alpha(-0.1)
        with self.assertRaises(ValueError):
            self.hybrid._validate_alpha(1.1)
        self.assertEqual(self.hybrid._validate_alpha(0.5), 0.5)

    def test_min_max_normalize(self):
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        norm = min_max_normalize(arr)
        self.assertAlmostEqual(float(norm[0]), 0.0)
        self.assertAlmostEqual(float(norm[1]), 0.5)
        self.assertAlmostEqual(float(norm[2]), 1.0)

    def test_alpha_0_pure_content(self):
        # Query Book 1 with alpha=0 (pure content) -> should recommend Book 2 (same author)
        recs = self.hybrid.recommend_for_book("Book 1", n_recommendations=2, alpha=0.0)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["title"], "Book 2")
        self.assertNotIn("Book 1", [r["title"] for r in recs])

    def test_alpha_1_pure_collab(self):
        # Query Book 1 with alpha=1 (pure collab) -> should recommend Book 3 (same user ratings)
        recs = self.hybrid.recommend_for_book("Book 1", n_recommendations=2, alpha=1.0)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["title"], "Book 3")
        self.assertNotIn("Book 1", [r["title"] for r in recs])

    def test_intermediate_alpha(self):
        recs = self.hybrid.recommend_for_book("Book 1", n_recommendations=3, alpha=0.5)
        self.assertEqual(len(recs), 3)
        self.assertIn("hybrid_score", recs[0])
        self.assertIn("collab_score", recs[0])
        self.assertIn("content_score", recs[0])

    def test_cold_start_service(self):
        service = RecommendationService(
            collab_recommender=self.collab_rec,
            content_recommender=self.content_rec,
            popularity_ranking=["Book 1", "Book 2", "Book 3", "Book 4", "Book 5"],
            warm_user_threshold=3,
        )

        # Scenario A: New user with no seeds -> popularity fallback
        new_u_recs = service.recommend_for_new_user(n_recommendations=3)
        self.assertEqual(len(new_u_recs), 3)
        self.assertEqual(new_u_recs[0]["strategy"], "popularity_fallback")

        # Scenario A2: New user with seed book -> seed content fallback
        seed_recs = service.recommend_for_new_user(seed_books=["Book 1"], n_recommendations=2)
        self.assertEqual(len(seed_recs), 2)
        self.assertEqual(seed_recs[0]["strategy"], "seed_content_fallback")
        self.assertEqual(seed_recs[0]["title"], "Book 2")

        # Scenario B: Limited history user (1 rating) -> limited_history_hybrid
        lim_user_df = pd.DataFrame([{"title": "Book 1", "rating": 9.0}])
        lim_recs = service.recommend_for_user(lim_user_df, n_recommendations=2)
        self.assertEqual(len(lim_recs), 2)
        self.assertEqual(lim_recs[0]["strategy"], "limited_history_hybrid")
        self.assertNotIn("Book 1", [r["title"] for r in lim_recs])

        # Scenario D: Unseen book
        new_book_recs = service.recommend_for_new_book(
            title="Book X",
            author="Author Alpha",
            publisher="Pub Xenon",
            n_recommendations=2,
        )
        self.assertEqual(len(new_book_recs), 2)
        self.assertEqual(new_book_recs[0]["strategy"], "new_book_content_inference")


if __name__ == "__main__":
    unittest.main()
