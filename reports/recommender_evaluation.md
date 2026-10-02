# Offline Evaluation Report: Book Recommender System

**Execution Date:** 2026-10-02  
**Evaluation Protocol:** Leakage-Free Leave-One-Out Holdout  
**Random Seed:** 42  
**Reproduction Command:** `python scripts/evaluate_recommender.py --seed 42 --positive-threshold 7.0`  

---

## 1. Dataset Summary

* **Raw / Cleaned Source:** `artifacts/final_rating.pkl`
* **Total Explicit Ratings (rating > 0):** 14961
* **Unique Books in Evaluation Matrix:** 742
* **Total Explicit Users:** 839
* **Eligible Evaluation Users:** 793 (94.52%)
* **Excluded Users:** 46 (5.48%)

---

## 2. Evaluation Protocol & Methodology

* **Protocol:** User-level leave-one-out cross-validation.
* **Positive Interaction Definition:** Explicit rating >= 7.0 (on a 1–10 scale).
* **Train/Test Splitting:** For each eligible user, exactly one positive interaction was randomly sampled as the test item. All remaining interactions across all users formed the training set.
* **Candidate Set:** For each user, the recommender ranks all books in the catalog excluding the books already present in that user's training history.
* **Hit Criterion:** A recommendation is counted as a hit if the held-out test book appears in the user's top-K recommendation list.

---

## 3. User Coverage & Exclusion Breakdown

| Category | User Count | Percentage | Reason |
| :--- | :---: | :---: | :--- |
| **Eligible Users** | **793** | **94.52%** | Have >= 2 explicit ratings and at least 1 positive rating (>= 7.0). |
| Single Rating (Positive) | 32 | 3.81% | Only 1 rating total; holding it out leaves 0 training history for user profile. |
| Single Rating (Non-Positive) | 8 | 0.95% | Only 1 rating total, and rating is < 7.0. |
| Multiple Ratings (No Positive) | 6 | 0.72% | User has multiple explicit ratings, but none meet the positive threshold >= 7.0. |

---

## 4. Leakage Prevention Audit

* **Test Pairs Absent from Training:** `PASSED` (0 leaked pairs found in train set)
* **Popularity Computed Exclusively on Train:** `PASSED`
* **Training Matrix Free of Test Interactions:** `PASSED`
* **Separate Evaluation Model Used:** `PASSED` (Production artifact was NOT reused)
* **Overall Leakage Audit Status:** **PASSED (Zero Leakage)**

---

## 5. Measured Offline Evaluation Results

### Comparison Table

| Metric | Popularity Baseline | Cosine Collaborative Filtering | Relative Difference |
| :--- | :---: | :---: | :---: |
| **Precision@5** | `0.009332` | `0.017654` | **+89.19%** |
| **Recall@5** | `0.046658` | `0.088272` | **+89.19%** |
| **NDCG@5** | `0.028439` | `0.064603` | **+127.16%** |
| **Precision@10** | `0.006305` | `0.012106` | **+92.00%** |
| **Recall@10** | `0.063052` | `0.121059` | **+92.00%** |
| **NDCG@10** | `0.033956` | `0.074945` | **+120.71%** |

---

## 6. Interpretation of Results

1. **Collaborative Filtering vs. Popularity:** Across all 6 evaluated metrics, the Item-Item Cosine Collaborative Filtering model outperforms the Popularity baseline by approximately **+89% on Recall@5** (0.0883 vs. 0.0467) and **+127% on NDCG@5** (0.0646 vs. 0.0284).
2. **Top-10 Hit Rate (Recall@10):** The Cosine Collaborative Filtering model retrieves the exact held-out test book in the top 10 recommendations for **12.11% of all eligible users** (96 out of 793 users), compared to 6.31% for the Popularity baseline.
3. **Ranking Quality (NDCG):** The significant increase in NDCG@5 (0.0646 vs 0.0284) indicates that when relevant books are retrieved, they are positioned near the top ranks (ranks 1 to 3) more frequently than under non-personalized popularity sorting.

---

## 7. Limitations

1. **Extreme Sparsity (97.6%):** With only 14,168 training ratings across 742 books and 839 users, many co-occurrence pairs have limited overlap.
2. **Cold-Start Users / Items:** The model cannot evaluate users with fewer than 2 explicit ratings or books outside the 742 popular set.
3. **Item Metadata Unused:** The evaluation model uses purely rating patterns. Content features (genre, author, text descriptions) are not incorporated in this phase.
