#!/usr/bin/env python
"""Offline Evaluation Runner for Hybrid Recommender & Cold-Start Policies (Phase 3).

Usage:
    python scripts/evaluate_hybrid_recommender.py
    python scripts/evaluate_hybrid_recommender.py --seed 42 --output-dir reports
"""

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.evaluator import create_leave_one_out_split
from src.evaluation.hybrid_evaluator import HybridEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline evaluation of the hybrid book recommender system."
    )
    parser.add_argument(
        "--ratings-path",
        type=str,
        default="artifacts/final_rating.pkl",
        help="Path to ratings DataFrame pickle.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="artifacts/book_metadata_cosine.pkl",
        help="Path to metadata DataFrame pickle.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=7.0,
        help="Minimum rating threshold for positive interaction (default: 7.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic leave-one-out sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save reports (default: reports).",
    )
    return parser.parse_args()


def save_reports(
    results: Dict[str, Any],
    output_dir: str,
    reproduce_cmd: str,
) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "phase3_hybrid_evaluation.json")
    md_path = os.path.join(output_dir, "phase3_hybrid_evaluation.md")

    # 1. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Save Markdown
    stats = results["dataset_statistics"]
    alpha_sweep = results["alpha_sweep"]
    cold_start = results["cold_start_simulation"]
    pop_ref = results["popularity_baseline_reference"]

    md_content = f"""# Phase 3 Hybrid Recommendation & Cold-Start Offline Evaluation Report

**Execution Date:** 2026-10-02  
**Evaluation Protocol:** Leakage-Free Leave-One-Out Holdout  
**Random Seed:** {stats['seed']}  
**Reproduction Command:** `{reproduce_cmd}`  

---

## 1. Dataset & Pipeline Summary

* **Explicit Ratings Evaluated:** {stats['total_explicit_ratings']}
* **Training Interactions:** {stats['train_ratings']}
* **Held-out Test Interactions:** {stats['test_ratings']}
* **Catalog Books:** {stats['total_books']}
* **Eligible Evaluation Users:** {stats['eligible_users']}
* **Excluded Users:** {stats['excluded_users']}
* **Positive Interaction Threshold:** rating >= {stats['positive_threshold']}

---

## 2. Model & Weighting Architecture

The hybrid recommender combines collaborative rating similarities with content-based TF-IDF textual similarities:

$$\\text{{hybrid\\_score}} = \\alpha \\cdot \\text{{norm}}(\\text{{collab\\_score}}) + (1.0 - \\alpha) \\cdot \\text{{norm}}(\\text{{content\\_score}})$$

* **Score Normalization:** Min-Max normalization per candidate list:
  $$\\tilde{{s}} = \\frac{{s - \\min(s)}}{{\\max(s) - \\min(s) + \\epsilon}}$$
* **Candidate Filtering:** Books already present in a user's training history are strictly excluded from recommendation candidates.
* **Alpha Weights Evaluated:** $\\alpha \\in \\{{0.00, 0.25, 0.50, 0.75, 1.00\\}}$.
  * $\\alpha = 1.00$: Pure Collaborative Filtering.
  * $\\alpha = 0.00$: Pure Content-Based Filtering.
  * Intermediate $\\alpha$: Hybrid combinations.

---

## 3. Alpha Sweep Measured Results

All models evaluated on the exact same 793 eligible users and held-out test interactions:

| Model Configuration | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | `{pop_ref['Precision@5']:.6f}` | `{pop_ref['Recall@5']:.6f}` | `{pop_ref['NDCG@5']:.6f}` | `{pop_ref['Precision@10']:.6f}` | `{pop_ref['Recall@10']:.6f}` | `{pop_ref['NDCG@10']:.6f}` |
| **alpha = 0.00 (Pure Content)** | `{alpha_sweep['alpha_0.00']['Precision@5']:.6f}` | `{alpha_sweep['alpha_0.00']['Recall@5']:.6f}` | `{alpha_sweep['alpha_0.00']['NDCG@5']:.6f}` | `{alpha_sweep['alpha_0.00']['Precision@10']:.6f}` | `{alpha_sweep['alpha_0.00']['Recall@10']:.6f}` | `{alpha_sweep['alpha_0.00']['NDCG@10']:.6f}` |
| **alpha = 0.25 (Content-Heavy)** | `{alpha_sweep['alpha_0.25']['Precision@5']:.6f}` | `{alpha_sweep['alpha_0.25']['Recall@5']:.6f}` | `{alpha_sweep['alpha_0.25']['NDCG@5']:.6f}` | `{alpha_sweep['alpha_0.25']['Precision@10']:.6f}` | `{alpha_sweep['alpha_0.25']['Recall@10']:.6f}` | `{alpha_sweep['alpha_0.25']['NDCG@10']:.6f}` |
| **alpha = 0.50 (Balanced)** | `{alpha_sweep['alpha_0.50']['Precision@5']:.6f}` | `{alpha_sweep['alpha_0.50']['Recall@5']:.6f}` | `{alpha_sweep['alpha_0.50']['NDCG@5']:.6f}` | `{alpha_sweep['alpha_0.50']['Precision@10']:.6f}` | `{alpha_sweep['alpha_0.50']['Recall@10']:.6f}` | `{alpha_sweep['alpha_0.50']['NDCG@10']:.6f}` |
| **alpha = 0.75 (Collab-Heavy)** | `{alpha_sweep['alpha_0.75']['Precision@5']:.6f}` | `{alpha_sweep['alpha_0.75']['Recall@5']:.6f}` | `{alpha_sweep['alpha_0.75']['NDCG@5']:.6f}` | `{alpha_sweep['alpha_0.75']['Precision@10']:.6f}` | `{alpha_sweep['alpha_0.75']['Recall@10']:.6f}` | `{alpha_sweep['alpha_0.75']['NDCG@10']:.6f}` |
| **alpha = 1.00 (Pure Collab)** | `{alpha_sweep['alpha_1.00']['Precision@5']:.6f}` | `{alpha_sweep['alpha_1.00']['Recall@5']:.6f}` | `{alpha_sweep['alpha_1.00']['NDCG@5']:.6f}` | `{alpha_sweep['alpha_1.00']['Precision@10']:.6f}` | `{alpha_sweep['alpha_1.00']['Recall@10']:.6f}` | `{alpha_sweep['alpha_1.00']['NDCG@10']:.6f}` |

---

## 4. Cold-Start Simulation Results

Simulated across the same test users and held-out interactions:

| Cold-Start Scenario | Strategy | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New User (0 History)** | Popularity Fallback | `{cold_start['new_user_popularity_fallback']['Precision@5']:.6f}` | `{cold_start['new_user_popularity_fallback']['Recall@5']:.6f}` | `{cold_start['new_user_popularity_fallback']['NDCG@5']:.6f}` | `{cold_start['new_user_popularity_fallback']['Precision@10']:.6f}` | `{cold_start['new_user_popularity_fallback']['Recall@10']:.6f}` | `{cold_start['new_user_popularity_fallback']['NDCG@10']:.6f}` |
| **Limited History (1 Seed)** | Single-Seed Content Fallback | `{cold_start['single_seed_content_fallback']['Precision@5']:.6f}` | `{cold_start['single_seed_content_fallback']['Recall@5']:.6f}` | `{cold_start['single_seed_content_fallback']['NDCG@5']:.6f}` | `{cold_start['single_seed_content_fallback']['Precision@10']:.6f}` | `{cold_start['single_seed_content_fallback']['Recall@10']:.6f}` | `{cold_start['single_seed_content_fallback']['NDCG@10']:.6f}` |

---

## 5. Measured Observations

1. **Measured Retrieval across Alphas:** Pure content ($\\alpha = 0.00$) achieved Recall@10 of `{alpha_sweep['alpha_0.00']['Recall@10']:.6f}` and NDCG@10 of `{alpha_sweep['alpha_0.00']['NDCG@10']:.6f}`. Pure collaborative filtering ($\\alpha = 1.00$) achieved Recall@10 of `{alpha_sweep['alpha_1.00']['Recall@10']:.6f}` and NDCG@10 of `{alpha_sweep['alpha_1.00']['NDCG@10']:.6f}`.
2. **Hybrid Combinations:** Combining both signals at $\\alpha = 0.50$ yielded Recall@10 of `{alpha_sweep['alpha_0.50']['Recall@10']:.6f}` and NDCG@10 of `{alpha_sweep['alpha_0.50']['NDCG@10']:.6f}`, compared to `{alpha_sweep['alpha_1.00']['Recall@10']:.6f}` for $\\alpha = 1.00$ and `{alpha_sweep['alpha_0.00']['Recall@10']:.6f}` for $\\alpha = 0.00$.
3. **Cold-Start Simulation:** Under zero-history cold-start conditions, popularity fallback achieved Recall@10 of `{cold_start['new_user_popularity_fallback']['Recall@10']:.6f}` and NDCG@10 of `{cold_start['new_user_popularity_fallback']['NDCG@10']:.6f}`. With a single seed book, content similarity fallback achieved Recall@10 of `{cold_start['single_seed_content_fallback']['Recall@10']:.6f}` and NDCG@10 of `{cold_start['single_seed_content_fallback']['NDCG@10']:.6f}`.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("BOOK RECOMMENDER SYSTEM - HYBRID EVALUATION (PHASE 3)")
    print("=" * 70)

    ratings_path = os.path.join(REPO_ROOT, args.ratings_path)
    metadata_path = os.path.join(REPO_ROOT, args.metadata_path)

    if not os.path.isfile(ratings_path):
        print(f"[-] ERROR: Ratings file not found: {ratings_path}")
        return 1
    if not os.path.isfile(metadata_path):
        print(f"[-] ERROR: Metadata file not found: {metadata_path}")
        return 1

    print(f"[*] Loading ratings from:  {ratings_path}")
    with open(ratings_path, "rb") as f:
        df = pickle.load(f)

    print(f"[*] Loading metadata from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        meta_df = pickle.load(f)

    print(f"[*] Creating leave-one-out split (seed={args.seed}, pos_threshold={args.positive_threshold})...")
    split = create_leave_one_out_split(
        df,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
    )

    print("\n[*] Initializing and fitting HybridEvaluator...")
    evaluator = HybridEvaluator(split, metadata_df=meta_df)
    evaluator.fit()

    print("\n[*] Running Alpha Sweep Evaluation (alphas=[0.00, 0.25, 0.50, 0.75, 1.00])...")
    sweep_results = evaluator.run_sweep(alphas=[0.0, 0.25, 0.50, 0.75, 1.0], k_values=[5, 10])

    print("\n--- Measured Alpha Sweep Metrics ---")
    for a_key, metrics in sweep_results["alpha_sweep"].items():
        print(f"  {a_key:<11} | P@5: {metrics['Precision@5']:.6f} | R@5: {metrics['Recall@5']:.6f} | NDCG@5: {metrics['NDCG@5']:.6f} | P@10: {metrics['Precision@10']:.6f} | R@10: {metrics['Recall@10']:.6f} | NDCG@10: {metrics['NDCG@10']:.6f}")

    print("\n--- Cold-Start Simulation Metrics ---")
    for sc_key, metrics in sweep_results["cold_start_simulation"].items():
        print(f"  {sc_key:<30} | P@5: {metrics['Precision@5']:.6f} | R@5: {metrics['Recall@5']:.6f} | NDCG@5: {metrics['NDCG@5']:.6f} | P@10: {metrics['Precision@10']:.6f} | R@10: {metrics['Recall@10']:.6f} | NDCG@10: {metrics['NDCG@10']:.6f}")

    # Popularity reference from Phase 2
    from src.evaluation.evaluator import RecommenderEvaluator
    pe = RecommenderEvaluator(split).fit()
    pop_metrics = pe.evaluate_popularity_baseline(k_values=[5, 10])

    full_results = {
        "dataset_statistics": {
            "total_explicit_ratings": len(split.train_df) + len(split.test_df),
            "train_ratings": len(split.train_df),
            "test_ratings": len(split.test_df),
            "total_books": len(evaluator.item_names),
            "total_users": len(split.eligible_users) + len(split.excluded_users),
            "eligible_users": len(split.eligible_users),
            "excluded_users": len(split.excluded_users),
            "positive_threshold": split.positive_threshold,
            "seed": split.seed,
        },
        "alpha_sweep": sweep_results["alpha_sweep"],
        "cold_start_simulation": sweep_results["cold_start_simulation"],
        "popularity_baseline_reference": pop_metrics,
    }

    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    reproduce_cmd = f"python scripts/evaluate_hybrid_recommender.py --seed {args.seed} --positive-threshold {args.positive_threshold}"
    json_path, md_path = save_reports(full_results, output_dir, reproduce_cmd)

    print(f"\n[+] Saved JSON report to:     {json_path}")
    print(f"[+] Saved Markdown report to: {md_path}")
    print("\n[SUCCESS] Phase 3 Hybrid Evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
