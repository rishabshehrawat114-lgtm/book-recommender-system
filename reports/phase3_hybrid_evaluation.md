# Phase 3 Hybrid Recommendation & Cold-Start Offline Evaluation Report

**Execution Date:** 2026-10-02  
**Evaluation Protocol:** Leakage-Free Leave-One-Out Holdout  
**Random Seed:** 42  
**Reproduction Command:** `python scripts/evaluate_hybrid_recommender.py --seed 42 --positive-threshold 7.0`  

---

## 1. Dataset & Pipeline Summary

* **Explicit Ratings Evaluated:** 14961
* **Training Interactions:** 14168
* **Held-out Test Interactions:** 793
* **Catalog Books:** 742
* **Eligible Evaluation Users:** 793
* **Excluded Users:** 46
* **Positive Interaction Threshold:** rating >= 7.0

---

## 2. Model & Weighting Architecture

The hybrid recommender combines collaborative rating similarities with content-based TF-IDF textual similarities:

$$\text{hybrid\_score} = \alpha \cdot \text{norm}(\text{collab\_score}) + (1.0 - \alpha) \cdot \text{norm}(\text{content\_score})$$

* **Score Normalization:** Min-Max normalization per candidate list:
  $$\tilde{s} = \frac{s - \min(s)}{\max(s) - \min(s) + \epsilon}$$
* **Candidate Filtering:** Books already present in a user's training history are strictly excluded from recommendation candidates.
* **Alpha Weights Evaluated:** $\alpha \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$.
  * $\alpha = 1.00$: Pure Collaborative Filtering.
  * $\alpha = 0.00$: Pure Content-Based Filtering.
  * Intermediate $\alpha$: Hybrid combinations.

---

## 3. Alpha Sweep Measured Results

All models evaluated on the exact same 793 eligible users and held-out test interactions:

| Model Configuration | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | `0.009332` | `0.046658` | `0.028439` | `0.006305` | `0.063052` | `0.033956` |
| **alpha = 0.00 (Pure Content)** | `0.021438` | `0.107188` | `0.073984` | `0.015511` | `0.155107` | `0.089577` |
| **alpha = 0.25 (Content-Heavy)** | `0.023455` | `0.117276` | `0.081206` | `0.016646` | `0.166456` | `0.096963` |
| **alpha = 0.50 (Balanced)** | `0.024968` | `0.124842` | `0.084199` | `0.016898` | `0.168979` | `0.098594` |
| **alpha = 0.75 (Collab-Heavy)** | `0.022951` | `0.114754` | `0.080729` | `0.015637` | `0.156368` | `0.093998` |
| **alpha = 1.00 (Pure Collab)** | `0.017654` | `0.088272` | `0.064603` | `0.012106` | `0.121059` | `0.074945` |

---

## 4. Cold-Start Simulation Results

Simulated across the same test users and held-out interactions:

| Cold-Start Scenario | Strategy | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New User (0 History)** | Popularity Fallback | `0.008575` | `0.042875` | `0.025396` | `0.006053` | `0.060530` | `0.031350` |
| **Limited History (1 Seed)** | Single-Seed Content Fallback | `0.009584` | `0.047919` | `0.030461` | `0.005675` | `0.056747` | `0.033426` |

---

## 5. Measured Observations

1. **Measured Retrieval across Alphas:** Pure content ($\alpha = 0.00$) achieved Recall@10 of `0.155107` and NDCG@10 of `0.089577`. Pure collaborative filtering ($\alpha = 1.00$) achieved Recall@10 of `0.121059` and NDCG@10 of `0.074945`.
2. **Hybrid Combinations:** Combining both signals at $\alpha = 0.50$ yielded Recall@10 of `0.168979` and NDCG@10 of `0.098594`, compared to `0.121059` for $\alpha = 1.00$ and `0.155107` for $\alpha = 0.00$.
3. **Cold-Start Simulation:** Under zero-history cold-start conditions, popularity fallback achieved Recall@10 of `0.060530` and NDCG@10 of `0.031350`. With a single seed book, content similarity fallback achieved Recall@10 of `0.056747` and NDCG@10 of `0.033426`.
