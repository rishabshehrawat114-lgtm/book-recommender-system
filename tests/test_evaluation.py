"""Unit tests for Recommender System Evaluation Module (Phase 2).

Tests ranking metrics, edge cases, split reproducibility, and leakage prevention.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.evaluation.evaluator import create_leave_one_out_split, RecommenderEvaluator


class TestEvaluationMetrics(unittest.TestCase):
    """Deterministic synthetic unit tests for Precision@K, Recall@K, and NDCG@K."""

    def test_precision_at_k_basic(self):
        actual = ["Book A"]
        # Relevant item at rank 1, K=5 -> Precision = 1/5 = 0.2
        self.assertAlmostEqual(precision_at_k(actual, ["Book A", "Book B", "Book C"], k=5), 0.2)
        # Relevant item at rank 3, K=5 -> Precision = 1/5 = 0.2
        self.assertAlmostEqual(precision_at_k(actual, ["Book X", "Book Y", "Book A"], k=5), 0.2)
        # Relevant item at rank 6, K=5 -> Precision = 0/5 = 0.0
        self.assertAlmostEqual(
            precision_at_k(actual, ["B1", "B2", "B3", "B4", "B5", "Book A"], k=5),
            0.0,
        )

    def test_recall_at_k_basic(self):
        actual = ["Book A"]
        # Relevant item in top 5 -> Recall = 1/1 = 1.0
        self.assertAlmostEqual(recall_at_k(actual, ["Book A", "Book B"], k=5), 1.0)
        self.assertAlmostEqual(recall_at_k(actual, ["Book X", "Book A"], k=5), 1.0)
        # Relevant item not in top 5 -> Recall = 0.0
        self.assertAlmostEqual(
            recall_at_k(actual, ["B1", "B2", "B3", "B4", "B5", "Book A"], k=5),
            0.0,
        )

    def test_ndcg_at_k_discounting(self):
        actual = ["Book A"]
        # Rank 1: DCG = 1 / log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k(actual, ["Book A", "Book B"], k=5), 1.0)

        # Rank 2: DCG = 1 / log2(3), IDCG = 1.0 -> NDCG = 1 / log2(3) = 0.63092975
        expected_rank2 = 1.0 / np.log2(3)
        self.assertAlmostEqual(ndcg_at_k(actual, ["Book B", "Book A"], k=5), expected_rank2, places=6)

        # Rank 3: DCG = 1 / log2(4) = 0.5, IDCG = 1.0 -> NDCG = 0.5
        self.assertAlmostEqual(ndcg_at_k(actual, ["Book B", "Book C", "Book A"], k=5), 0.5, places=6)

        # Absent from top 5: NDCG = 0.0
        self.assertAlmostEqual(
            ndcg_at_k(actual, ["B1", "B2", "B3", "B4", "B5", "Book A"], k=5),
            0.0,
        )

    def test_empty_predictions_and_actuals(self):
        self.assertEqual(precision_at_k(["Book A"], [], k=5), 0.0)
        self.assertEqual(recall_at_k(["Book A"], [], k=5), 0.0)
        self.assertEqual(ndcg_at_k(["Book A"], [], k=5), 0.0)

        self.assertEqual(precision_at_k([], ["Book A"], k=5), 0.0)
        self.assertEqual(recall_at_k([], ["Book A"], k=5), 0.0)
        self.assertEqual(ndcg_at_k([], ["Book A"], k=5), 0.0)

    def test_k_larger_than_prediction_list(self):
        actual = ["Book A"]
        predicted = ["Book A", "Book B"]
        # List has 2 items, k=10 -> precision = 1 / 10 = 0.1
        self.assertAlmostEqual(precision_at_k(actual, predicted, k=10), 0.1)
        self.assertAlmostEqual(recall_at_k(actual, predicted, k=10), 1.0)
        self.assertAlmostEqual(ndcg_at_k(actual, predicted, k=10), 1.0)

    def test_invalid_k_raises_error(self):
        with self.assertRaises(ValueError):
            precision_at_k(["A"], ["A"], k=0)
        with self.assertRaises(ValueError):
            recall_at_k(["A"], ["A"], k=-1)
        with self.assertRaises(ValueError):
            ndcg_at_k(["A"], ["A"], k=-5)


class TestLeaveOneOutSplitAndLeakage(unittest.TestCase):
    """Test suite for split logic, reproducibility, and leakage audit."""

    def setUp(self):
        # Create deterministic synthetic DataFrame
        # 3 users, 5 books
        self.synthetic_df = pd.DataFrame([
            # User 1: 3 ratings, 2 positive (8, 9), 1 mediocre (5)
            {"user_id": 1, "title": "Book A", "rating": 8},
            {"user_id": 1, "title": "Book B", "rating": 9},
            {"user_id": 1, "title": "Book C", "rating": 5},
            # User 2: 2 ratings, 1 positive (10), 1 mediocre (4)
            {"user_id": 2, "title": "Book B", "rating": 10},
            {"user_id": 2, "title": "Book D", "rating": 4},
            # User 3: 1 rating (ineligible - single rating)
            {"user_id": 3, "title": "Book E", "rating": 9},
            # User 4: 2 ratings, none >= 7 (ineligible - no positive rating)
            {"user_id": 4, "title": "Book A", "rating": 6},
            {"user_id": 4, "title": "Book C", "rating": 5},
        ])

    def test_split_eligibility_and_reasons(self):
        split = create_leave_one_out_split(self.synthetic_df, positive_threshold=7.0, min_ratings=2, seed=42)

        self.assertEqual(sorted(split.eligible_users), [1, 2])
        self.assertIn(3, split.excluded_users)
        self.assertEqual(split.excluded_users[3], "single_rating_positive")
        self.assertIn(4, split.excluded_users)
        self.assertEqual(split.excluded_users[4], "multiple_ratings_no_positive")

        # Check test size matches eligible users
        self.assertEqual(len(split.test_df), 2)
        # Check train size is total (8) - test (2) = 6
        self.assertEqual(len(split.train_df), 6)

    def test_split_reproducibility_with_seed(self):
        split1 = create_leave_one_out_split(self.synthetic_df, positive_threshold=7.0, min_ratings=2, seed=42)
        split2 = create_leave_one_out_split(self.synthetic_df, positive_threshold=7.0, min_ratings=2, seed=42)

        pd.testing.assert_frame_equal(split1.train_df, split2.train_df)
        pd.testing.assert_frame_equal(split1.test_df, split2.test_df)

    def test_held_out_interaction_absent_from_train(self):
        split = create_leave_one_out_split(self.synthetic_df, positive_threshold=7.0, min_ratings=2, seed=42)

        train_pairs = set(zip(split.train_df["user_id"], split.train_df["title"]))
        for _, row in split.test_df.iterrows():
            test_pair = (row["user_id"], row["title"])
            self.assertNotIn(
                test_pair,
                train_pairs,
                f"Data leakage detected! Test pair {test_pair} was found in training interactions!",
            )

    def test_evaluator_leakage_audit_passes(self):
        split = create_leave_one_out_split(self.synthetic_df, positive_threshold=7.0, min_ratings=2, seed=42)
        evaluator = RecommenderEvaluator(split)
        evaluator.fit()

        audit = evaluator.verify_leakage()
        self.assertTrue(audit["all_leakage_checks_passed"])
        self.assertTrue(audit["test_pairs_absent_from_train"])
        self.assertTrue(audit["popularity_strictly_from_train"])
        self.assertTrue(audit["training_matrix_free_of_test_data"])


if __name__ == "__main__":
    unittest.main()
