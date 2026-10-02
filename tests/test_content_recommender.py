"""Unit tests for ContentRecommender (Phase 3)."""

import os
import sys
import unittest
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.content_recommender import ContentRecommender, clean_metadata_text


class TestContentRecommender(unittest.TestCase):
    """Test suite for ContentRecommender."""

    def setUp(self):
        self.synthetic_metadata = pd.DataFrame([
            {"title": "The Hobbit", "author": "J.R.R. Tolkien", "publisher": "George Allen &amp; Unwin"},
            {"title": "The Fellowship of the Ring", "author": "J.R.R. Tolkien", "publisher": "George Allen & Unwin"},
            {"title": "The Two Towers", "author": "J.R.R. Tolkien", "publisher": "George Allen & Unwin"},
            {"title": "Dune", "author": "Frank Herbert", "publisher": "Chilton Books"},
            {"title": "Dune Messiah", "author": "Frank Herbert", "publisher": "Putnam"},
            {"title": "Foundation", "author": "Isaac Asimov", "publisher": "Gnome Press"},
        ])
        self.rec = ContentRecommender(author_weight=2)
        self.rec.fit(self.synthetic_metadata)

    def test_text_cleaning(self):
        cleaned = clean_metadata_text("John Wiley &amp; Sons Inc   ")
        self.assertEqual(cleaned, "John Wiley and Sons Inc")
        self.assertEqual(clean_metadata_text(None), "")
        self.assertEqual(clean_metadata_text(float("nan")), "")

    def test_fit_and_dimensions(self):
        self.assertTrue(self.rec.is_fitted)
        self.assertEqual(len(self.rec.item_names), 6)
        self.assertEqual(self.rec.tfidf_matrix.shape[0], 6)
        self.assertGreater(self.rec.tfidf_matrix.shape[1], 0)

    def test_known_book_recommendation(self):
        recs = self.rec.recommend("The Hobbit", n_recommendations=2)
        self.assertEqual(len(recs), 2)
        # Should recommend Tolkien's other books first due to author emphasis
        rec_titles = [r["title"] for r in recs]
        self.assertIn("The Fellowship of the Ring", rec_titles)
        self.assertNotIn("The Hobbit", rec_titles)  # Excludes self-match

    def test_self_match_excluded(self):
        for book in self.rec.item_names:
            recs = self.rec.recommend(book, n_recommendations=3)
            rec_titles = [r["title"] for r in recs]
            self.assertNotIn(book, rec_titles)

    def test_invalid_title_raises_error(self):
        with self.assertRaises(ValueError):
            self.rec.recommend("Non-Existent Book Title 999", n_recommendations=2)

    def test_predict_user_scores(self):
        # User liked Tolkien books
        scores = self.rec.predict_user_scores(["The Hobbit", "The Fellowship of the Ring"])
        self.assertEqual(len(scores), 6)
        # The Two Towers should have highest remaining score among other books
        two_towers_idx = self.rec._title_to_idx["The Two Towers"]
        dune_idx = self.rec._title_to_idx["Dune"]
        self.assertGreater(scores[two_towers_idx], scores[dune_idx])

    def test_recommend_for_new_book(self):
        # Query unseen book by Frank Herbert
        recs = self.rec.recommend_for_new_book(
            title="Children of Dune",
            author="Frank Herbert",
            publisher="Putnam",
            n_recommendations=2,
        )
        self.assertEqual(len(recs), 2)
        # Top recommendations should be Herbert's Dune books
        rec_titles = [r["title"] for r in recs]
        self.assertTrue("Dune" in rec_titles or "Dune Messiah" in rec_titles)

    def test_new_book_empty_metadata_graceful(self):
        recs = self.rec.recommend_for_new_book(title="", author="", publisher="")
        self.assertEqual(recs, [])

    def test_deterministic_results(self):
        recs1 = self.rec.recommend("Dune", n_recommendations=2)
        recs2 = self.rec.recommend("Dune", n_recommendations=2)
        self.assertEqual(recs1, recs2)


if __name__ == "__main__":
    unittest.main()
