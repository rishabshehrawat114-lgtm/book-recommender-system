"""Unit tests for RecommendationService and artifact loading integration (Phase 4)."""

import os
import sys
import unittest
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.artifacts import load_recommendation_service


class TestRecommendationServiceIntegration(unittest.TestCase):
    """Integration test suite for RecommendationService loaded from production artifacts."""

    @classmethod
    def setUpClass(cls):
        artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
        cls.service, cls.ratings_df = load_recommendation_service(artifacts_dir)
        cls.catalog = cls.service.content_recommender.get_supported_books()

    def test_01_service_initialization(self):
        """Verify service loads with expected catalog size and non-empty ratings."""
        self.assertIsNotNone(self.service)
        self.assertEqual(len(self.catalog), 742)
        self.assertGreater(len(self.ratings_df), 10000)

    def test_02_known_book_recommendation(self):
        """Verify known book produces recommendations excluding the query book."""
        query = "1984"
        self.assertIn(query, self.catalog)
        recs = self.service.recommend_for_book(query, n_recommendations=5, method="hybrid")
        self.assertEqual(len(recs), 5)
        titles = [r["title"] for r in recs]
        self.assertNotIn(query, titles)
        self.assertIn("hybrid_score", recs[0])
        self.assertIn("author", recs[0])

    def test_03_invalid_book_raises_error(self):
        """Verify invalid book title raises ValueError."""
        with self.assertRaises(ValueError):
            self.service.recommend_for_book("Nonexistent Book Title XYZ 123", n_recommendations=5)

    def test_04_known_user_recommendation(self):
        """Verify known user recommendations exclude books the user already rated."""
        uid = 277427
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == uid][["title", "rating"]]
        self.assertGreater(len(user_ratings), 0)
        user_rated_titles = set(user_ratings["title"])

        recs = self.service.recommend_for_user(user_ratings, n_recommendations=5)
        self.assertEqual(len(recs), 5)
        for r in recs:
            self.assertNotIn(r["title"], user_rated_titles)
            self.assertIn("hybrid_score", r)
            self.assertIn("author", r)

    def test_05_new_user_cold_start_popularity(self):
        """Verify new user with no seed books receives popularity fallback."""
        recs = self.service.recommend_for_new_user(seed_books=None, n_recommendations=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs[0]["strategy"], "popularity_fallback")
        self.assertIn("title", recs[0])
        self.assertIn("author", recs[0])

    def test_06_new_user_cold_start_seeds(self):
        """Verify new user with seed book receives content fallback excluding seed."""
        seed = "Animal Farm"
        recs = self.service.recommend_for_new_user(seed_books=[seed], n_recommendations=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs[0]["strategy"], "seed_content_fallback")
        titles = [r["title"] for r in recs]
        self.assertNotIn(seed, titles)

    def test_07_unseen_book_recommendation(self):
        """Verify unseen book produces content-based recommendations from catalog."""
        recs = self.service.recommend_for_new_book(
            title="The Hobbit",
            author="J.R.R. Tolkien",
            publisher="Houghton Mifflin",
            n_recommendations=5,
        )
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs[0]["strategy"], "new_book_content_inference")
        self.assertIn("title", recs[0])

    def test_08_missing_artifact_raises_filenotfound(self):
        """Verify loading from non-existent directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_recommendation_service("non_existent_artifacts_dir_xyz")


if __name__ == "__main__":
    unittest.main()
