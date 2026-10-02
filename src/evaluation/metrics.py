"""Evaluation metrics for Recommender Systems.

Provides mathematically correct implementations of:
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
"""

from typing import Any, List, Set, Union
import numpy as np


def precision_at_k(
    actual: Union[List[Any], Set[Any]],
    predicted: List[Any],
    k: int = 5,
) -> float:
    """Calculate Precision@K.

    Definition:
        Precision@K = (Number of relevant items in top-K predictions) / K

    Args:
        actual: Ground-truth relevant items for the user.
        predicted: Ordered list of recommended item titles/IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Precision score in [0.0, 1.0].

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not predicted or not actual:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]

    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / k)


def recall_at_k(
    actual: Union[List[Any], Set[Any]],
    predicted: List[Any],
    k: int = 5,
) -> float:
    """Calculate Recall@K.

    Definition:
        Recall@K = (Number of relevant items in top-K predictions) / |Actual|

    Args:
        actual: Ground-truth relevant items for the user.
        predicted: Ordered list of recommended item titles/IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        Recall score in [0.0, 1.0].

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not predicted or not actual:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]

    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / len(actual_set))


def ndcg_at_k(
    actual: Union[List[Any], Set[Any]],
    predicted: List[Any],
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

    Formula:
        DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
        IDCG@K = sum_{i=1}^{min(K, |Actual|)} (1 / log2(i + 1))
        NDCG@K = DCG@K / IDCG@K

    Args:
        actual: Ground-truth relevant items for the user.
        predicted: Ordered list of recommended item titles/IDs.
        k: Cutoff rank (must be > 0).

    Returns:
        NDCG score in [0.0, 1.0].

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not predicted or not actual:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in actual_set:
            # Rank is 1-indexed (i + 1), denominator is log2(rank + 1) = log2(i + 2)
            dcg += 1.0 / np.log2(i + 2)

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    n_rel = min(k, len(actual_set))
    if n_rel == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))

    return float(dcg / idcg) if idcg > 0 else 0.0
