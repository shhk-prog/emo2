import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from .metrics import compute_rsa_similarity
from .splits import get_grouped_kfold_splits

class LayerProber:
    """Evaluates how well a specific hidden representation layer predicts target affective values."""
    def __init__(self, alpha: float = 1.0, cv: int = 5, seed: int = 42, 
                 use_pca: bool = True, n_components: int = 50):
        self.alpha = alpha
        self.cv = cv
        self.seed = seed
        self.use_pca = use_pca
        self.n_components = n_components

    def _build_pipeline(self, n_samples: int) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            n_comp = min(self.n_components, n_samples)
            steps.append(('pca', PCA(n_components=n_comp, random_state=self.seed)))
        steps.append(('ridge', Ridge(alpha=self.alpha)))
        return Pipeline(steps)

    def evaluate_probing(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray) -> Dict[str, float]:
        """
        Runs GroupKFold cross-validation pipeline to predict target y (1D array) from features X (2D array).
        Normalizes and (optionally) runs PCA inside the fold to prevent data leakage.
        Returns dictionary of metrics: r2, mse, rmse, mae, pearson_r, spearman_r.
        """
        if len(X) < self.cv or len(y) < self.cv:
            return {"r2": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "pearson_r": 0.0, "spearman_r": 0.0}

        # Create a dummy dataframe to leverage our splits.py utility
        df = pd.DataFrame({'stimulus_id': group_ids})
        
        y_preds = np.zeros_like(y, dtype=float)

        for train_idx, val_idx in get_grouped_kfold_splits(df, n_splits=self.cv, group_col='stimulus_id'):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._build_pipeline(len(X_train))
            model.fit(X_train, y_train)
            y_preds[val_idx] = model.predict(X_val)

        r2 = float(r2_score(y, y_preds))
        mse = float(mean_squared_error(y, y_preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y, y_preds))
        
        pr, _ = pearsonr(y, y_preds)
        sr, _ = spearmanr(y, y_preds)

        return {
            "r2": r2 if not np.isnan(r2) else 0.0,
            "mse": mse if not np.isnan(mse) else 0.0,
            "rmse": rmse if not np.isnan(rmse) else 0.0,
            "mae": mae if not np.isnan(mae) else 0.0,
            "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
            "spearman_r": float(sr) if not np.isnan(sr) else 0.0
        }

class FixedSplitProber:
    """Evaluates representations on a fixed train/dev/test split (e.g., AIPsy-Affect minimal pairs)."""
    def __init__(self, alpha: float = 1.0, seed: int = 42, 
                 use_pca: bool = True, n_components: int = 50, task_type: str = "classification"):
        self.alpha = alpha
        self.seed = seed
        self.use_pca = use_pca
        self.n_components = n_components
        self.task_type = task_type # "classification" or "regression"

    def _build_pipeline(self, n_samples: int) -> Pipeline:
        steps = [('scaler', StandardScaler())] # Z-score normalization
        if self.use_pca:
            n_comp = min(self.n_components, n_samples)
            steps.append(('pca', PCA(n_components=n_comp, random_state=self.seed)))
            
        if self.task_type == "classification":
            steps.append(('clf', RidgeClassifier(alpha=self.alpha, random_state=self.seed)))
        else:
            steps.append(('ridge', Ridge(alpha=self.alpha, random_state=self.seed)))
            
        return Pipeline(steps)

    def evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if len(X_train) == 0 or len(X_test) == 0:
            return {}
            
        model = self._build_pipeline(len(X_train))
        model.fit(X_train, y_train)
        
        if self.task_type == "classification":
            y_preds = model.predict(X_test)
            acc = accuracy_score(y_test, y_preds)
            try:
                # RidgeClassifier uses decision_function for scores
                y_scores = model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_scores)
            except Exception:
                auc = 0.5
            return {"accuracy": float(acc), "roc_auc": float(auc)}
        else:
            y_preds = model.predict(X_test)
            r2 = float(r2_score(y_test, y_preds))
            mse = float(mean_squared_error(y_test, y_preds))
            pr, _ = pearsonr(y_test, y_preds)
            sr, _ = spearmanr(y_test, y_preds)
            return {
                "r2": r2 if not np.isnan(r2) else 0.0,
                "mse": mse if not np.isnan(mse) else 0.0,
                "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
                "spearman_r": float(sr) if not np.isnan(sr) else 0.0
            }


def compute_distance_matrix(vectors: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Computes pairwise distance matrix for a set of vectors. Supports 'euclidean' and 'correlation'."""
    dist_array = pdist(vectors, metric=metric)
    return squareform(dist_array)

def run_rsa_analysis(
    layer_vectors: Dict[int, np.ndarray],
    target_vectors: Dict[str, np.ndarray],
    metric: str = 'euclidean'
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Computes Representational Similarity Analysis (RSA) across layers for multiple targets.
    
    Args:
        layer_vectors: Dict of layer_idx -> activation vectors.
        target_vectors: Dict of target_name -> target representation vectors.
                        e.g., {'human_VA': ..., 'reported_post_VA': ..., 'delta_VA': ...}
        metric: Distance metric to use ('euclidean' or 'correlation').
                        
    Returns:
        Dict: target_name -> {layer_idx -> {spearman: float, pearson: float}}
    """
    results: Dict[str, Dict[int, Dict[str, float]]] = {target_name: {} for target_name in target_vectors.keys()}
    
    # Precompute target distance matrices
    target_dist_mats = {
        name: compute_distance_matrix(vecs, metric=metric)
        for name, vecs in target_vectors.items()
    }
    
    for layer_idx, X_layer in layer_vectors.items():
        layer_dist_mat = compute_distance_matrix(X_layer, metric=metric)
        
        for target_name, target_dist_mat in target_dist_mats.items():
            rsa_scores = compute_rsa_similarity(layer_dist_mat, target_dist_mat)
            results[target_name][layer_idx] = rsa_scores
            
    return results

def run_shuffled_baseline_probing(
    X: np.ndarray, y: np.ndarray, group_ids: np.ndarray, prober: LayerProber, n_permutations: int = 10
) -> Dict[str, float]:
    """
    Runs permutation test by shuffling labels across groups to compute baseline random chance expectation.
    Maintains group structure during shuffling if possible, or shuffles y securely.
    """
    r2_scores = []
    rng = np.random.default_rng(prober.seed)
    
    for _ in range(n_permutations):
        # Create a shuffled copy of y
        y_shuffled = rng.permutation(y)
        res = prober.evaluate_probing(X, y_shuffled, group_ids)
        r2_scores.append(res["r2"])
        
    return {
        "mean_shuffled_r2": float(np.mean(r2_scores)),
        "std_shuffled_r2": float(np.std(r2_scores))
    }

def run_length_control_probing(
    X_train: np.ndarray, lengths_train: np.ndarray, 
    X_test: np.ndarray, lengths_test: np.ndarray,
    alpha: float = 1.0, seed: int = 42, use_pca: bool = True
) -> Dict[str, float]:
    """
    Trains a probe to predict the string length (or token count) from representations.
    This serves as a control: if representations only encode surface-level text length,
    this probe will have high R2, but it shouldn't correlate strongly with affect if affect is properly disentangled.
    """
    prober = FixedSplitProber(alpha=alpha, seed=seed, use_pca=use_pca, task_type="regression")
    return prober.evaluate(X_train, lengths_train, X_test, lengths_test)

def run_shuffled_control_classification(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray,
    prober: FixedSplitProber, n_permutations: int = 10
) -> Dict[str, float]:
    """
    Runs label shuffling on the training set to establish a baseline for classification.
    """
    acc_scores = []
    auc_scores = []
    rng = np.random.default_rng(prober.seed)
    
    for _ in range(n_permutations):
        y_train_shuffled = rng.permutation(y_train)
        res = prober.evaluate(X_train, y_train_shuffled, X_test, y_test)
        acc_scores.append(res.get("accuracy", 0.0))
        auc_scores.append(res.get("roc_auc", 0.5))
        
    return {
        "mean_shuffled_accuracy": float(np.mean(acc_scores)),
        "std_shuffled_accuracy": float(np.std(acc_scores)),
        "mean_shuffled_roc_auc": float(np.mean(auc_scores)),
        "std_shuffled_roc_auc": float(np.std(auc_scores))
    }
