# 📚 Book Recommendation System

A Machine Learning web application combining collaborative filtering, TF-IDF content similarity, and adaptive cold-start strategies to provide personalized book recommendations with transparent explainability. Built with **Scikit-Learn**, **SciPy**, **Pandas**, and **Streamlit**.

---

## 📌 Overview

This project implements an item-based collaborative filtering and content-based hybrid recommendation system using the **Book-Crossing Dataset**. By constructing a sparse user-item interaction matrix, utilizing unsupervised **Nearest Neighbors (Cosine Distance / Brute Force)**, and extracting **TF-IDF textual n-gram representations** from verified book metadata, the system recommends relevant titles for catalog books, existing readers, new users, and unseen books.

## ✨ Features

* **Collaborative Filtering:** Unsupervised Nearest Neighbors on reader co-occurrence patterns using cosine distance over compressed sparse matrices.
* **Content-Based Filtering:** Scikit-Learn TF-IDF vectorizer ($2,933$ n-grams) capturing author, title, and publisher metadata affinity.
* **Linear Hybrid Fusion:** Configurable weighted combination of collaborative and content signals with candidate min-max score normalization.
* **Cold-Start Handling:**
  - *New User:* Global popularity fallback or multi-seed content preference matching.
  - *Limited History ($<3$ ratings):* Automatic adaptive shift to content-weighted hybrid ($\alpha = 0.25$).
  - *Unseen Book:* Out-of-catalog TF-IDF projection on title and author metadata.
* **Recommendation Explainability:** Evidence-based breakdown for every recommendation displaying the operational strategy, collaborative co-occurrence signal, and metadata overlap without unsubstantiated semantic claims.
* **Consumer-Facing Streamlit UI:** Clean, intuitive 3-section interface (**Home**, **Recommend Books**, **About**) with responsive book cards, cover art fallbacks, and zero technical jargon.
* **Multi-Mode Book Discovery:**
  - *Similar Books:* Search by title to find titles with similar storyline and author characteristics.
  - *Personalized Recommendations:* Tailored suggestions based on individual reader profile history.
  - *New User Flow:* Multi-seed favorite title matching or instant exploration of popular catalog books.
  - *Adaptive Policies:* Automatic limited-history routing and unseen-book content inference handled internally.
* **Offline Ranking Evaluation & Analytics:** Rigorous, leakage-free leave-one-out evaluation reporting Precision@K, Recall@K, and NDCG@K ($K \in \{5, 10\}$) with a fixed random seed (`seed=42`), alongside reproducible dataset analytics scripts in `reports/` and `scripts/`.

---

## 🏛️ System Architecture

```text
Interaction Data (final_rating.pkl) & Book Metadata (book_metadata_cosine.pkl)
                         │
                         ▼
             Machine Learning Models
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
Collaborative Engine                  Content Engine
(NearestNeighbors, Cosine)            (TfidfVectorizer, Cosine)
      │                                     │
      └──────────────────┬──────────────────┘
                         ▼
        Hybrid Recommender Engine (Linear Weighted Fusion)
                         │
                         ▼
   Unified Recommendation Service (src/models/recommendation_service.py)
   ├── Scenario A: New User (Popularity / Seed Content Fallback)
   ├── Scenario B: Limited History (Content-Heavy Hybrid, α = 0.25)
   ├── Scenario C: Warm User (Balanced Hybrid, α = 0.50)
   └── Scenario D: Unseen Book (Real-Time TF-IDF Projection)
                         │
                         ▼
        User-Facing Streamlit Application (app.py)
   ├── 🏠 Home: Clean landing page with system overview & direct call-to-action
   ├── 📚 Recommend Books: Core multi-mode discovery hub
   │   ├── 🔍 Similar Books: Content & hybrid similarity matching
   │   ├── 🎯 Personalized Recommendations: Taste-tailored picks from reader history
   │   └── 🌱 New User: Multi-seed preference matching or popular title discovery
   └── ℹ️ About: Non-technical overview of recommendation principles
```

> **Note on Architecture Separation:** Technical diagnostics, matrix sparsity analytics, and offline evaluation benchmarks are maintained as offline developer pipelines (`scripts/`, `reports/`, `tests/`), keeping the main user interface uncluttered for everyday readers.

---

## 📊 Dataset Statistics & Preprocessing Pipeline

The dataset derives from the public **Book-Crossing** dataset through a multi-stage filtering and explicit preference modeling pipeline:

1. **Global User Activity Filter (Raw Interactions):** Users with $> 200$ global interaction records across the raw 1.1M dataset were identified as active community readers, resulting in **888 frequent readers**.
2. **Catalog Interaction Filter:** Book titles with $\ge 50$ total interaction records (including implicit `rating = 0` interactions such as shelf-adds and unrated reads) among those 888 active readers were retained, defining the **742-book catalog**.
3. **Explicit Preference Modeling:** Implicit non-ratings (`rating = 0`) were excluded from the collaborative filtering matrix so that unrated interactions are not conflated with negative sentiment or missing cells. Only positive explicit ratings ($1$–$10$) were retained:
   * **839** of the 888 frequent readers contributed at least one explicit rating to the 742 catalog books.
   * Total explicit ratings: **14,961**.
4. **Final Interaction Matrix Characteristics:**
   * **Matrix Dimensions:** $742 \text{ books} \times 839 \text{ users}$ ($622,538$ potential cells).
   * **Observed Non-Zero Ratings:** $14,961$ explicit ratings ($> 0$).
   * **Matrix Sparsity:** $\mathbf{97.60\%}$ unobserved interaction cells.
   * **Ratings Per Book (Explicit):** Mean = $20.16$, Median = $17.0$, Min = $3$, Max = $104$ *(note: the $\ge 50$ threshold applied to total interactions in Step 2, not explicit ratings alone)*.
   * **Ratings Per User (Explicit):** Mean = $17.83$, Median = $13.0$, Min = $1$, Max = $459$ *(note: the $> 200$ threshold applied globally in Step 1, not within this 742-book catalog alone)*.

---

## 🔬 Machine Learning Approach

1. **Collaborative Filtering:**
   - Model: `sklearn.neighbors.NearestNeighbors(metric='cosine', algorithm='brute')`
   - Input: Compressed Sparse Row (`scipy.sparse.csr_matrix`) of shape $742 \times 839$.
   - Similarity: $s_{\text{collab}} = 1.0 - d_{\text{cosine}}$.

2. **Content-Based Filtering:**
   - Model: `sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)`
   - Feature Soup: Title + Author (weighted $\times 2$) + Publisher.
   - Catalog Matrix: $742 \times 2,933$ L2-normalized document vectors with precomputed pairwise similarity.

3. **Hybrid Combination & Normalization:**
   - Min-Max scaling per candidate list:
     $$\tilde{s}_i = \frac{s_i - \min(s)}{\max(s) - \min(s) + 10^{-9}}$$
   - Weighted score:
     $$\text{hybrid\_score} = \alpha \cdot \tilde{s}_{\text{collab}} + (1.0 - \alpha) \cdot \tilde{s}_{\text{content}}$$

---

## 📈 Offline Evaluation Results

Evaluated using a leakage-free leave-one-out protocol on 793 eligible evaluation users with a positive threshold of $\ge 7.0$ (`seed=42`):

| Model Configuration | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline (Phase 2)** | 0.009332 | 0.046658 | 0.028439 | 0.006305 | 0.063052 | 0.033956 |
| **Pure Content (α = 0.00)** | 0.021438 | 0.107188 | 0.073984 | 0.015511 | 0.155107 | 0.089577 |
| **Content-Heavy (α = 0.25)** | 0.023455 | 0.117276 | 0.081206 | 0.016646 | 0.166456 | 0.096963 |
| **Balanced Hybrid (α = 0.50)** | 0.024968 | 0.124842 | 0.084199 | 0.016898 | 0.168979 | 0.098594 |
| **Collab-Heavy (α = 0.75)** | 0.022951 | 0.114754 | 0.080729 | 0.015637 | 0.156368 | 0.093998 |
| **Pure Collaborative (α = 1.00)** | 0.017654 | 0.088272 | 0.064603 | 0.012106 | 0.121059 | 0.074945 |

### Cold-Start Benchmark:
* **New User (Zero History, Popularity Fallback):** Precision@5: 0.008575, Recall@5: 0.042875, NDCG@5: 0.025396, Recall@10: 0.060530, NDCG@10: 0.031350.
* **Limited History (1 Seed, Content Fallback):** Precision@5: 0.009584, Recall@5: 0.047919, NDCG@5: 0.030461, Recall@10: 0.056747, NDCG@10: 0.033426.

---

## 📂 Project Structure

```text
├── artifacts/
│   ├── model.pkl                       # Original baseline NearestNeighbors model (Euclidean)
│   ├── model_cosine.pkl                # Upgraded NearestNeighbors model (Cosine Distance - Phase 1)
│   ├── book_pivot.pkl                  # Original dense user-item pivot table
│   ├── book_pivot_sparse.pkl           # Memory-efficient sparse CSR matrix payload (Phase 1)
│   ├── books_name.pkl                  # Original index of filtered book titles
│   ├── books_name_cosine.pkl           # Index of book titles for the cosine model (Phase 1)
│   ├── final_rating.pkl                # Cleaned ratings metadata DataFrame
│   ├── book_metadata_cosine.pkl        # Lightweight metadata lookup (poster URLs, authors - Phase 1)
│   ├── content_tfidf_vectorizer.pkl    # Fitted Scikit-learn TfidfVectorizer (Phase 3)
│   ├── content_tfidf_matrix.pkl        # Sparse CSR TF-IDF document-term matrix (Phase 3)
│   └── content_book_metadata.pkl       # Normalized book metadata catalog (Phase 3)
├── src/
│   ├── data/
│   │   └── preprocessing.py            # Data loading, cleaning, filtering & matrix construction
│   ├── models/
│   │   ├── recommender.py              # Collaborative BookRecommender (Cosine NearestNeighbors)
│   │   ├── content_recommender.py      # TF-IDF ContentRecommender on title, author, publisher
│   │   ├── hybrid_recommender.py       # Weighted HybridRecommender (normalized collaborative + content)
│   │   └── recommendation_service.py   # Unified service layer handling cold-start & warm-user policies
│   ├── utils/
│   │   ├── analytics.py                # Dataset descriptive metrics & distribution utilities
│   │   ├── artifacts.py                # Artifact serialization & loading utilities
│   │   └── explainability.py           # Transparent recommendation explanation generators
│   └── evaluation/
│       ├── metrics.py                  # Precision@K, Recall@K, NDCG@K ranking metrics
│       ├── evaluator.py                # Phase 2 leave-one-out evaluator & popularity baseline
│       └── hybrid_evaluator.py         # Phase 3 alpha sweep & cold-start simulation evaluator
├── scripts/
│   ├── train_recommender.py            # Phase 1 collaborative model training script
│   ├── evaluate_recommender.py         # Phase 2 collaborative evaluation pipeline runner
│   ├── train_content_recommender.py    # Phase 3 content-based TF-IDF training script
│   └── evaluate_hybrid_recommender.py  # Phase 3 hybrid alpha sweep & cold-start evaluation runner
├── reports/
│   ├── recommender_evaluation.json     # Phase 2 collaborative evaluation results
│   ├── recommender_evaluation.md       # Phase 2 evaluation report
│   ├── phase3_hybrid_evaluation.json   # Phase 3 alpha sweep & cold-start simulation results
│   └── phase3_hybrid_evaluation.md     # Phase 3 evaluation report
├── tests/
│   ├── test_recommender.py             # Phase 1 engine correctness unit tests (9 tests)
│   ├── test_evaluation.py              # Phase 2 metrics & leakage verification unit tests (10 tests)
│   ├── test_content_recommender.py     # Phase 3 content-based recommender unit tests (9 tests)
│   ├── test_hybrid_recommender.py      # Phase 3 hybrid & cold-start service unit tests (6 tests)
│   ├── test_service.py                 # Phase 4 service & artifact loading integration tests (8 tests)
│   └── test_phase5_readiness.py        # Phase 5 analytics, explainability & portability tests (5 tests)
├── app.py                              # Streamlit multi-mode recommendation web application
├── Book_Recommendor.ipynb              # Original exploratory notebook reference
├── setup.py                            # Package distribution configuration
├── setup.sh                            # Streamlit server setup for deployment
├── Procfile                            # Cloud deployment orchestration
├── requirements.txt                    # Python runtime dependencies with compatibility bounds
├── .gitignore                          # Standard environment, cache, and editor exclusions
└── README.md                           # Comprehensive project documentation
```

---

## 🚀 Running the Project

### 1. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```

### 2. Standalone Training & Offline Evaluation:
```bash
# Train collaborative model (Phase 1):
python scripts/train_recommender.py

# Train content-based TF-IDF model (Phase 3):
python scripts/train_content_recommender.py

# Run Phase 2 collaborative evaluation:
python scripts/evaluate_recommender.py

# Run Phase 3 hybrid alpha sweep & cold-start evaluation:
python scripts/evaluate_hybrid_recommender.py --seed 42 --output-dir reports
```

### 3. Run the Automated Test Suite:
```bash
python -m unittest discover tests -v
```
*Current test suite status: 47 / 47 tests passing (0 failures, 0 errors).*

---

## ⚠️ Limitations

1. **Metadata Depth:** Content metadata is limited to `title`, `author`, and `publisher`. Unstructured synopses, blurb text, genre taxonomies, and user reviews were not available in the dataset.
2. **Collaborative Catalog Boundary:** Collaborative filtering is restricted to the 742 catalog books selected by the interaction threshold ($\ge 50$ total interactions among frequent readers) until matrix recomputation.
3. **External Cover Delivery:** Cover art URLs point to legacy Amazon CDN hosts; remote missing images are handled via local placeholder cards.
4. **Uniform Fusion Weighting:** Linear $\alpha$ parameter applies uniformly across all items rather than item-specific uncertainty-adaptive weights.

---

## 🔮 Future Improvements

1. **Rich Metadata Ingestion:** Integrating verified open-catalog genre annotations (e.g., Open Library) to enrich content representations.
2. **Matrix Factorization:** Exploring implicit feedback models (e.g., BPR or ALS) for sparse interaction scaling.
3. **User-Specific Dynamic Alpha:** Dynamically calculating $\alpha$ per user based on individual rating history entropy and variance.
