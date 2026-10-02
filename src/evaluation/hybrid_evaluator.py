"""Evaluation pipeline for Hybrid Recommender across multiple alpha weights and cold-start simulations.

Implements leakage-free evaluation for:
  - Alpha sweep: [0.0, 0.25, 0.50, 0.75, 1.0]
  - Simulated new-user cold-start (popularity fallback)
  - Simulated single-interaction cold-start (single-seed content fallback)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.evaluator import HoldoutSplit
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.models.content_recommender import ContentRecommender
from src.models.hybrid_recommender import min_max_normalize


class HybridEvaluator:
    """Evaluates hybrid collaborative + content recommenders across alpha weights."""

    def __init__(
        self,
        holdout_split: HoldoutSplit,
        metadata_df: pd.DataFrame,
    ) -> None:
        """Initialize with holdout split and static book metadata.

        Args:
            holdout_split: Phase 2 HoldoutSplit instance.
            metadata_df: Book metadata DataFrame (title, author, publisher, etc.).
        """
        self.split = holdout_split
        self.train_df = holdout_split.train_df
        self.test_df = holdout_split.test_df
        self.eligible_users = holdout_split.eligible_users
        self.metadata_df = metadata_df

        self.item_names: List[str] = []
        self.title_to_idx: Dict[str, int] = {}
        self.collab_sim_matrix: Optional[np.ndarray] = None
        self.content_recommender: Optional[ContentRecommender] = None
        self.popularity_scores: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self) -> "HybridEvaluator":
        """Fit collaborative similarity matrix strictly on train_df and content model on metadata."""
        # 1. Collaborative training matrix strictly on train_df
        pivot_train = self.train_df.pivot_table(
            index="title",
            columns="user_id",
            values="rating",
        ).fillna(0.0)

        self.item_names = list(pivot_train.index)
        self.title_to_idx = {title: idx for idx, title in enumerate(self.item_names)}
        train_matrix = csr_matrix(pivot_train.values, dtype=np.float32)

        self.collab_sim_matrix = cosine_similarity(train_matrix, dense_output=True)
        np.fill_diagonal(self.collab_sim_matrix, 0.0)

        # 2. Content model on catalog metadata aligned with catalog titles
        meta_aligned = self.metadata_df.drop_duplicates("title").set_index("title")
        meta_aligned = meta_aligned.reindex(self.item_names).reset_index()

        self.content_recommender = ContentRecommender()
        self.content_recommender.fit(meta_aligned)

        # 3. Popularity baseline on train_df
        train_counts = self.train_df.groupby("title")["rating"].count()
        self.popularity_scores = np.array(
            [train_counts.get(title, 0) for title in self.item_names],
            dtype=np.float32,
        )

        self._is_fitted = True
        return self

    def evaluate_alpha(
        self,
        alpha: float,
        k_values: List[int] = [5, 10],
    ) -> Dict[str, float]:
        """Evaluate hybrid recommender for a specific alpha weight.

        Args:
            alpha: Collaborative weight in [0.0, 1.0].
            k_values: Cutoff ranks.

        Returns:
            Dictionary of ranking metrics.
        """
        if not self._is_fitted or self.collab_sim_matrix is None or self.content_recommender is None:
            raise RuntimeError("HybridEvaluator must be fit() before evaluation.")

        max_k = max(k_values)
        metric_accumulators: Dict[str, List[float]] = {
            f"Precision@{k}": [] for k in k_values
        }
        metric_accumulators.update({f"Recall@{k}": [] for k in k_values})
        metric_accumulators.update({f"NDCG@{k}": [] for k in k_values})

        n_catalog = len(self.item_names)

        for u in self.eligible_users:
            test_row = self.test_df[self.test_df["user_id"] == u]
            if test_row.empty:
                continue

            test_item = test_row["title"].values[0]
            actual = [test_item]

            user_train = self.train_df[self.train_df["user_id"] == u]
            user_train_titles = set(user_train["title"])
            user_train_indices = [self.title_to_idx[t] for t in user_train_titles if t in self.title_to_idx]

            # 1. Collaborative scores
            if alpha > 0.0:
                u_collab_vec = np.zeros(n_catalog, dtype=np.float32)
                for _, row in user_train.iterrows():
                    t = row["title"]
                    if t in self.title_to_idx:
                        u_collab_vec[self.title_to_idx[t]] = float(row["rating"])
                collab_scores = self.collab_sim_matrix.dot(u_collab_vec)
            else:
                collab_scores = np.zeros(n_catalog, dtype=np.float32)

            # 2. Content scores using positive training interactions (rating >= 7.0)
            if alpha < 1.0:
                pos_train = user_train[user_train["rating"] >= self.split.positive_threshold]
                if pos_train.empty:
                    pos_train = user_train

                liked_titles = [t for t in pos_train["title"] if t in self.title_to_idx]
                liked_ratings = [float(r) for t, r in zip(pos_train["title"], pos_train["rating"]) if t in self.title_to_idx]

                content_scores = self.content_recommender.predict_user_scores(
                    liked_titles=liked_titles,
                    weights=liked_ratings,
                )
            else:
                content_scores = np.zeros(n_catalog, dtype=np.float32)

            # 3. Min-Max normalization
            norm_collab = min_max_normalize(collab_scores)
            norm_content = min_max_normalize(content_scores)

            # 4. Weighted combination
            hybrid_scores = alpha * norm_collab + (1.0 - alpha) * norm_content

            # 5. Mask training items
            for idx in user_train_indices:
                hybrid_scores[idx] = -np.inf

            # 6. Rank candidates
            top_indices = np.argsort(hybrid_scores)[::-1][:max_k]
            top_predictions = [self.item_names[idx] for idx in top_indices]

            for k in k_values:
                metric_accumulators[f"Precision@{k}"].append(
                    precision_at_k(actual, top_predictions, k=k)
                )
                metric_accumulators[f"Recall@{k}"].append(
                    recall_at_k(actual, top_predictions, k=k)
                )
                metric_accumulators[f"NDCG@{k}"].append(
                    ndcg_at_k(actual, top_predictions, k=k)
                )

        return {metric: float(np.mean(vals)) for metric, vals in metric_accumulators.items()}

    def evaluate_cold_start_simulation(
        self,
        k_values: List[int] = [5, 10],
    ) -> Dict[str, Dict[str, float]]:
        """Simulate cold-start conditions and evaluate fallback strategies.

        Scenario 1: Simulated New User (0 history) -> Popularity fallback.
        Scenario 2: Single-seed Cold Start (1 seed book) -> Content fallback from 1 seed item.

        Returns:
            Dictionary with results for both cold-start scenarios.
        """
        if not self._is_fitted or self.content_recommender is None or self.popularity_scores is None:
            raise RuntimeError("HybridEvaluator must be fit() before evaluation.")

        max_k = max(k_values)

        # Metrics containers
        new_user_acc: Dict[str, List[float]] = {f"Precision@{k}": [] for k in k_values}
        new_user_acc.update({f"Recall@{k}": [] for k in k_values})
        new_user_acc.update({f"NDCG@{k}": [] for k in k_values})

        single_seed_acc: Dict[str, List[float]] = {f"Precision@{k}": [] for k in k_values}
        single_seed_acc.update({f"Recall@{k}": [] for k in k_values})
        single_seed_acc.update({f"NDCG@{k}": [] for k in k_values})

        for u in self.eligible_users:
            test_row = self.test_df[self.test_df["user_id"] == u]
            if test_row.empty:
                continue

            test_item = test_row["title"].values[0]
            actual = [test_item]

            user_train = self.train_df[self.train_df["user_id"] == u]

            # Scenario 1: 0 history known -> Popularity fallback across entire catalog
            pop_scores = self.popularity_scores.copy()
            top_pop_indices = np.argsort(pop_scores)[::-1][:max_k]
            top_pop_predictions = [self.item_names[idx] for idx in top_pop_indices]

            for k in k_values:
                new_user_acc[f"Precision@{k}"].append(precision_at_k(actual, top_pop_predictions, k=k))
                new_user_acc[f"Recall@{k}"].append(recall_at_k(actual, top_pop_predictions, k=k))
                new_user_acc[f"NDCG@{k}"].append(ndcg_at_k(actual, top_pop_predictions, k=k))

            # Scenario 2: Exactly 1 positive seed book provided from user's training history
            pos_train = user_train[user_train["rating"] >= self.split.positive_threshold]
            seed_book = pos_train.iloc[0]["title"] if not pos_train.empty else user_train.iloc[0]["title"]

            seed_scores = self.content_recommender.predict_user_scores([seed_book])
            # Mask the single seed book
            if seed_book in self.title_to_idx:
                seed_scores[self.title_to_idx[seed_book]] = -np.inf

            top_seed_indices = np.argsort(seed_scores)[::-1][:max_k]
            top_seed_predictions = [self.item_names[idx] for idx in top_seed_indices]

            for k in k_values:
                single_seed_acc[f"Precision@{k}"].append(precision_at_k(actual, top_seed_predictions, k=k))
                single_seed_acc[f"Recall@{k}"].append(recall_at_k(actual, top_seed_predictions, k=k))
                single_seed_acc[f"NDCG@{k}"].append(ndcg_at_k(actual, top_seed_predictions, k=k))

        return {
            "new_user_popularity_fallback": {m: float(np.mean(vals)) for m, vals in new_user_acc.items()},
            "single_seed_content_fallback": {m: float(np.mean(vals)) for m, vals in single_seed_acc.items()},
        }

    def run_sweep(
        self,
        alphas: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
        k_values: List[int] = [5, 10],
    ) -> Dict[str, Any]:
        """Run complete alpha sweep and cold-start evaluations."""
        if not self._is_fitted:
            self.fit()

        alpha_results: Dict[str, Dict[str, float]] = {}
        for a in alphas:
            key = f"alpha_{a:.2f}"
            alpha_results[key] = self.evaluate_alpha(alpha=a, k_values=k_values)

        cold_start_results = self.evaluate_cold_start_simulation(k_values=k_values)

        return {
            "alpha_sweep": alpha_results,
            "cold_start_simulation": cold_start_results,
        }
