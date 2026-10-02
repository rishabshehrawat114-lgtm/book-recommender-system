"""Recommendation explainability utilities (Phase 5).

Generates evidence-based, transparent explanations for book recommendations
using only verified model signals (collaborative co-occurrence, TF-IDF metadata
similarity, and cold-start fallback policies).
"""

from typing import Any, Dict, Optional


def generate_recommendation_explanation(
    item: Dict[str, Any],
    method: str = "hybrid",
    alpha: float = 0.5,
) -> Dict[str, str]:
    """Generate factual, evidence-based explanation for a recommendation item.

    Args:
        item: Recommendation dictionary containing title, scores, and strategy.
        method: Method used ('hybrid', 'collaborative', or 'content').
        alpha: Collaborative weight used for hybrid scoring.

    Returns:
        Dictionary with 'source', 'reason', and 'signals' keys.
    """
    strategy = item.get("strategy", "")

    if strategy == "popularity_fallback":
        return {
            "source": "Catalog Popularity Fallback",
            "reason": (
                "Recommended because no user history was available for collaborative or "
                "content modeling. Ranked by overall reader rating frequency in the catalog."
            ),
            "signals": "Global explicit rating frequency in catalog.",
        }

    if strategy == "seed_content_fallback":
        return {
            "source": "Seed Book Content Similarity",
            "reason": (
                "Recommended based on metadata text similarity (Title, Author, Publisher) "
                "to the seed book(s) you selected."
            ),
            "signals": f"TF-IDF centroid cosine similarity: {item.get('score', 0):.4f}",
        }

    if strategy == "new_book_content_inference":
        sim_val = item.get("content_similarity", item.get("score", 0.0))
        return {
            "source": "Out-of-Catalog Content Inference",
            "reason": (
                "Recommended by projecting your provided Title, Author, and Publisher into "
                "the catalog TF-IDF feature space."
            ),
            "signals": f"TF-IDF cosine similarity: {sim_val:.4f}",
        }

    if strategy == "limited_history_hybrid":
        collab_sig = item.get("norm_collab_score", 0.0)
        content_sig = item.get("norm_content_score", 0.0)
        return {
            "source": "Limited-History Content-Weighted Hybrid (alpha = 0.25)",
            "reason": (
                "Because this user has limited rating history (< 3 ratings), recommendations "
                "prioritize metadata text similarity (75%) combined with available collaborative patterns (25%)."
            ),
            "signals": (
                f"Hybrid Score: {item.get('hybrid_score', 0):.4f} "
                f"[Collab Signal: {collab_sig:.3f}, Content Signal: {content_sig:.3f}]"
            ),
        }

    if strategy == "warm_user_hybrid":
        collab_sig = item.get("norm_collab_score", 0.0)
        content_sig = item.get("norm_content_score", 0.0)
        return {
            "source": f"Personalized Hybrid Recommender (alpha = {alpha:.2f})",
            "reason": (
                "Combines reader rating co-occurrence patterns from the collaborative matrix "
                "with metadata similarity from the user's highly-rated books (ratings >= 7.0)."
            ),
            "signals": (
                f"Hybrid Score: {item.get('hybrid_score', 0):.4f} "
                f"[Collab Signal: {collab_sig:.3f}, Content Signal: {content_sig:.3f}]"
            ),
        }

    if method == "collaborative" or "cosine_similarity" in item:
        sim_val = item.get("cosine_similarity", 0.0)
        return {
            "source": "Collaborative Filtering (Cosine Distance)",
            "reason": (
                "Recommended using reader rating co-occurrence patterns from the user-item "
                "interaction matrix without using book text."
            ),
            "signals": f"Cosine Similarity: {sim_val:.4f}",
        }

    if method == "content" or "content_similarity" in item:
        sim_val = item.get("content_similarity", 0.0)
        return {
            "source": "Content-Based Filtering (TF-IDF)",
            "reason": (
                "Recommended based on textual similarity across verified book metadata "
                "(Title, Author, Publisher)."
            ),
            "signals": f"Content Similarity: {sim_val:.4f}",
        }

    # Default fallback hybrid explanation
    collab_sig = item.get("norm_collab_score", 0.0)
    content_sig = item.get("norm_content_score", 0.0)
    return {
        "source": f"Hybrid Recommendation (alpha = {alpha:.2f})",
        "reason": (
            "Recommendation combines collaborative interaction patterns with metadata "
            "content similarity signals."
        ),
        "signals": (
            f"Hybrid Score: {item.get('hybrid_score', item.get('score', 0)):.4f} "
            f"[Collab Signal: {collab_sig:.3f}, Content Signal: {content_sig:.3f}]"
        ),
    }


def get_recommendation_summary(
    mode: str,
    strategy_label: str,
    alpha: float,
    n_returned: int,
    total_candidates: int = 742,
) -> Dict[str, Any]:
    """Generate concise operational summary for the generated recommendation batch."""
    return {
        "mode": mode,
        "strategy": strategy_label,
        "alpha": alpha,
        "recommendations_returned": n_returned,
        "total_catalog_candidates": total_candidates,
        "total_candidates": total_candidates,
        "self_match_excluded": True,
    }
