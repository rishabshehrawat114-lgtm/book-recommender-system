"""Unit tests for the Book Recommender Engine (Phase 1).

Tests core functionality:
  1. Model loads successfully.
  2. Model metric is cosine.
  3. Matrix dimensions are valid.
  4. Known book generates recommendations.
  5. Selected book is excluded from recommendations.
  6. Requested number of recommendations is respected.
  7. Invalid book titles handled gracefully.
  8. Similarity and distance values are numerically valid and properly labeled.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Add project root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.recommender import BookRecommender
from src.utils.artifacts import load_artifacts, save_artifacts


class TestBookRecommender(unittest.TestCase):
    """Test suite for the BookRecommender class and artifact loading."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures or load trained artifacts if present."""
        cls.artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
        cls.model_path = os.path.join(cls.artifacts_dir, "model_cosine.pkl")

        # Create a synthetic sparse matrix fixture for isolated unit testing
        cls.synthetic_items = ["Book A", "Book B", "Book C", "Book D", "Book E", "Book F"]
        # 6 books x 4 users
        cls.synthetic_dense = np.array([
            [10.0, 9.0, 0.0, 0.0],
            [9.0, 8.0, 0.0, 0.0],  # Very similar to Book A
            [0.0, 0.0, 8.0, 9.0],
            [0.0, 0.0, 7.0, 8.0],  # Very similar to Book C
            [5.0, 0.0, 5.0, 0.0],
            [0.0, 6.0, 0.0, 6.0],
        ])
        cls.synthetic_sparse = csr_matrix(cls.synthetic_dense)
        cls.synthetic_recommender = BookRecommender(metric="cosine", algorithm="brute")
        cls.synthetic_recommender.fit(cls.synthetic_sparse, cls.synthetic_items)

    def test_01_model_initialization_and_fit(self):
        """Test 1: Verify model initializes and fits correctly."""
        rec = BookRecommender(metric="cosine", algorithm="brute")
        self.assertFalse(rec.is_fitted)
        rec.fit(self.synthetic_sparse, self.synthetic_items)
        self.assertTrue(rec.is_fitted)

    def test_02_model_metric_is_cosine(self):
        """Test 2: Verify the model metric is explicitly 'cosine'."""
        self.assertEqual(self.synthetic_recommender.metric, "cosine")
        params = self.synthetic_recommender.get_model_params()
        self.assertEqual(params["metric"], "cosine")
        self.assertEqual(params["underlying_params"]["metric"], "cosine")

    def test_03_expected_matrix_dimensions(self):
        """Test 3: Verify matrix dimensions match the number of item names."""
        rec = self.synthetic_recommender
        self.assertEqual(rec.sparse_matrix.shape[0], len(self.synthetic_items))
        self.assertEqual(rec.sparse_matrix.shape[1], 4)

    def test_04_known_book_generates_recommendations(self):
        """Test 4: Verify a known book generates non-empty recommendations."""
        recs = self.synthetic_recommender.recommend("Book A", n_recommendations=3)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        # Book B has similar user ratings to Book A, so it should be the top neighbor
        self.assertEqual(recs[0]["title"], "Book B")

    def test_05_selected_book_excluded_from_recommendations(self):
        """Test 5: Verify the query book itself is NEVER returned in the recommendations."""
        for book in self.synthetic_items:
            recs = self.synthetic_recommender.recommend(book, n_recommendations=4)
            rec_titles = [r["title"] for r in recs]
            self.assertNotIn(
                book,
                rec_titles,
                f"Query book '{book}' was unexpectedly returned in its own recommendations!",
            )

    def test_06_requested_number_of_recommendations_respected(self):
        """Test 6: Verify the requested n_recommendations count is strictly respected."""
        for n in [1, 2, 3, 5]:
            recs = self.synthetic_recommender.recommend("Book A", n_recommendations=n)
            self.assertEqual(len(recs), n)

    def test_07_invalid_book_handled_gracefully(self):
        """Test 7: Verify invalid book titles raise ValueError with clear message."""
        invalid_title = "Non-Existent Book XYZ 12345"
        with self.assertRaises(ValueError) as context:
            self.synthetic_recommender.recommend(invalid_title, n_recommendations=3)
        self.assertIn("not present in the model index", str(context.exception))

    def test_08_similarity_and_distance_numerically_valid(self):
        """Test 8: Verify distance and similarity are numerically sound and correctly labeled.

        - Cosine distance must be in [0.0, 2.0]
        - Cosine similarity must be in [-1.0, 1.0]
        - Similarity must equal 1.0 - distance
        """
        recs = self.synthetic_recommender.recommend("Book A", n_recommendations=4)
        for r in recs:
            dist = r["cosine_distance"]
            sim = r["cosine_similarity"]

            self.assertGreaterEqual(dist, 0.0)
            self.assertLessEqual(dist, 2.0)

            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)

            # Check that similarity is 1.0 - distance (within rounding tolerance)
            self.assertAlmostEqual(sim, 1.0 - dist, places=4)

    def test_09_production_artifact_loading_if_trained(self):
        """Test 9: If model_cosine.pkl exists in artifacts, verify it loads and functions."""
        if os.path.isfile(self.model_path):
            recommender, meta = load_artifacts(self.artifacts_dir, "model_cosine.pkl")
            self.assertTrue(recommender.is_fitted)
            self.assertEqual(recommender.metric, "cosine")
            books = recommender.get_supported_books()
            self.assertGreater(len(books), 0)

            # Test query
            sample_book = books[0]
            recs = recommender.recommend(sample_book, n_recommendations=3)
            self.assertEqual(len(recs), 3)
            rec_titles = [r["title"] for r in recs]
            self.assertNotIn(sample_book, rec_titles)


if __name__ == "__main__":
    unittest.main()
