"""Offline evaluation module for the Book Recommender System.

Provides ranking metrics (Precision@K, Recall@K, NDCG@K), leakage-free
leave-one-out train/test splitters, and popularity baseline evaluators.
"""

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.evaluation.evaluator import (
    HoldoutSplit,
    create_leave_one_out_split,
    RecommenderEvaluator,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "HoldoutSplit",
    "create_leave_one_out_split",
    "RecommenderEvaluator",
]
