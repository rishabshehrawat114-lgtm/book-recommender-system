"""Unit tests for Phase 5 Production Readiness, Analytics, and Explainability."""

import json
import os
import sys
import unittest
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.analytics import compute_dataset_analytics
from src.utils.artifacts import load_recommendation_service
from src.utils.explainability import (
    generate_recommendation_explanation,
    get_recommendation_summary,
)


class TestPhase5Readiness(unittest.TestCase):
    """Test suite covering dataset analytics, explainability, and deployment readiness."""

    @classmethod
    def setUpClass(cls):
        artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
        cls.service, cls.ratings_df = load_recommendation_service(artifacts_dir)

    def test_01_dataset_analytics_integrity(self):
        """Verify empirical dataset statistics match verified Phase 1-3 invariants."""
        analytics = compute_dataset_analytics(self.ratings_df, n_catalog_books=742)

        self.assertEqual(analytics["total_catalog_books"], 742)
        self.assertEqual(analytics["explicit_ratings_count"], 14961)
        self.assertEqual(analytics["explicit_users_count"], 839)
        self.assertAlmostEqual(analytics["sparsity_percentage"], 97.60, places=1)

        # Check rating distribution has keys 1 to 10
        dist = analytics["rating_distribution"]
        self.assertEqual(sorted(list(dist.keys())), list(range(1, 11)))
        self.assertEqual(sum(dist.values()), 14961)

        # Check top-rated books table has 10 items
        top_df = analytics["top_rated_books_df"]
        self.assertEqual(len(top_df), 10)
        self.assertIn("The Lovely Bones: A Novel", list(top_df["title"]))

    def test_02_explainability_all_scenarios(self):
        """Verify evidence-based explanations for all system recommendation policies."""
        scenarios = [
            {"item": {"title": "B1", "strategy": "popularity_fallback"}, "method": "popularity"},
            {"item": {"title": "B2", "strategy": "seed_content_fallback", "score": 0.8}, "method": "content"},
            {"item": {"title": "B3", "strategy": "new_book_content_inference", "content_similarity": 0.65}, "method": "content"},
            {"item": {"title": "B4", "strategy": "limited_history_hybrid", "hybrid_score": 0.5, "norm_collab_score": 0.4, "norm_content_score": 0.6}, "method": "hybrid"},
            {"item": {"title": "B5", "strategy": "warm_user_hybrid", "hybrid_score": 0.7, "norm_collab_score": 0.8, "norm_content_score": 0.5}, "method": "hybrid"},
            {"item": {"title": "B6", "cosine_similarity": 0.75}, "method": "collaborative"},
            {"item": {"title": "B7", "content_similarity": 0.85}, "method": "content"},
        ]

        for sc in scenarios:
            exp = generate_recommendation_explanation(sc["item"], method=sc["method"], alpha=0.5)
            self.assertIn("source", exp)
            self.assertIn("reason", exp)
            self.assertIn("signals", exp)
            self.assertGreater(len(exp["source"]), 0)
            self.assertGreater(len(exp["reason"]), 0)
            self.assertGreater(len(exp["signals"]), 0)

            # Ensure no unsupported claims
            full_text = (exp["source"] + " " + exp["reason"] + " " + exp["signals"]).lower()
            self.assertNotIn("emotion", full_text)
            self.assertNotIn("plot twist", full_text)
            self.assertNotIn("guaranteed match", full_text)

    def test_03_recommendation_summary_generation(self):
        """Verify recommendation operational summary dictionary format."""
        summary = get_recommendation_summary(
            mode="Similar Books",
            strategy_label="Hybrid",
            alpha=0.5,
            n_returned=6,
            total_candidates=741,
        )
        self.assertEqual(summary["mode"], "Similar Books")
        self.assertEqual(summary["recommendations_returned"], 6)
        self.assertEqual(summary["total_catalog_candidates"], 741)
        self.assertTrue(summary["self_match_excluded"])

    def test_04_evaluation_report_file_integrity(self):
        """Verify Phase 3 evaluation report file exists, parses, and has valid ranges."""
        report_path = os.path.join(REPO_ROOT, "reports", "phase3_hybrid_evaluation.json")
        self.assertTrue(os.path.isfile(report_path))

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("alpha_sweep", data)
        self.assertIn("cold_start_simulation", data)
        self.assertIn("popularity_baseline_reference", data)

        for alpha_key, metrics in data["alpha_sweep"].items():
            for m_name, val in metrics.items():
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_05_path_portability_audit(self):
        """Audit Python source files to ensure no hardcoded local user machine paths exist."""
        subdirs_to_check = ["src", "scripts", "tests"]
        forbidden_substrings = ["C:\\Users\\", "C:/Users/"]

        checked_files = []
        for sdir in subdirs_to_check:
            dir_path = os.path.join(REPO_ROOT, sdir)
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if fname.endswith(".py") and os.path.abspath(os.path.join(root, fname)) != os.path.abspath(__file__):
                        fpath = os.path.join(root, fname)
                        checked_files.append(fpath)
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            for forb in forbidden_substrings:
                                self.assertNotIn(
                                    forb,
                                    content,
                                    f"Hardcoded path '{forb}' found in {fpath}",
                                )

        # Also check app.py
        app_path = os.path.join(REPO_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for forb in forbidden_substrings:
                self.assertNotIn(forb, content, f"Hardcoded path '{forb}' found in {app_path}")

        self.assertGreater(len(checked_files), 5)


if __name__ == "__main__":
    unittest.main()
