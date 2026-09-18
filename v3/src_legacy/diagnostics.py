"""
v3/src/diagnostics.py

Comprehensive Out-of-Distribution (OOD) and Activation Geometry Diagnostics.
Implements:
1. Mahalanobis Distance using Ledoit-Wolf regularized covariance shrinkage (high-dimensional d=1536 robust).
2. L2 Norm Ratio and Relative Norm Deviation.
3. Cosine Similarity to In-Distribution Mean.
4. Top-k PCA Subspace Projection Distance.
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

class MultivariateActivationDiagnostics:
    def __init__(self, n_components=10):
        """
        Initializes the multivariate diagnostics fitted on in-distribution activations.
        """
        self.cov_estimator = LedoitWolf()
        self.pca = PCA(n_components=n_components)
        self.mean_vector = None
        self.precision_matrix = None # Sigma^{-1}
        self.in_dist_mahalanobis = []
        self.in_dist_norms = []
        self.fitted = False

    def fit(self, in_dist_activations: np.ndarray):
        """
        Fits the baseline distribution statistics on target in-distribution activations.
        Shape: (N, D) where D is hidden_dim (e.g. 1536).
        """
        X = np.asarray(in_dist_activations, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.mean_vector = np.mean(X, axis=0)
        self.cov_estimator.fit(X)
        self.precision_matrix = self.cov_estimator.get_precision()
        
        # Fit PCA
        n_samples, n_features = X.shape
        k = min(self.pca.n_components, n_samples - 1, n_features)
        if k >= 1:
            self.pca = PCA(n_components=k)
            self.pca.fit(X)
        else:
            self.pca = None
            
        # Compute baseline in-distribution statistics
        self.in_dist_norms = np.linalg.norm(X, axis=1)
        diff = X - self.mean_vector
        # Vectorized Mahalanobis
        # D_M^2 = sum_j (diff * precision)_j * diff_j
        term = np.dot(diff, self.precision_matrix)
        sq_dist = np.sum(term * diff, axis=1)
        self.in_dist_mahalanobis = np.sqrt(np.maximum(sq_dist, 0.0))
        self.fitted = True
        return self

    def compute_mahalanobis(self, act: np.ndarray) -> float:
        """
        Computes the Mahalanobis distance D_M(h) from the target distribution.
        """
        if not self.fitted:
            raise RuntimeError("Diagnostics estimator not fitted!")
        h = np.asarray(act, dtype=np.float64).reshape(-1)
        diff = h - self.mean_vector
        term = np.dot(diff, self.precision_matrix)
        sq_dist = np.dot(term, diff)
        return float(np.sqrt(max(sq_dist, 0.0)))

    def compute_diagnostics(self, act: np.ndarray) -> dict:
        """
        Computes a full suite of diagnostic metrics for a candidate activation vector.
        Returns:
            - mahalanobis_distance: D_M
            - mahalanobis_percentile: Empirical percentile within in-distribution baseline
            - norm_ratio: ||h|| / mean(||h_in||)
            - cosine_similarity_to_mean: cos(h, mu)
            - pca_reconstruction_error: L2 distance to top-k PCA subspace
        """
        if not self.fitted:
            raise RuntimeError("Diagnostics estimator not fitted!")
            
        h = np.asarray(act, dtype=np.float64).reshape(-1)
        norm_h = float(np.linalg.norm(h))
        mean_in_norm = float(np.mean(self.in_dist_norms))
        norm_ratio = norm_h / (mean_in_norm + 1e-9)
        
        # Cosine similarity to mean
        norm_mu = float(np.linalg.norm(self.mean_vector))
        if norm_h > 1e-9 and norm_mu > 1e-9:
            cos_sim = float(np.dot(h, self.mean_vector) / (norm_h * norm_mu))
        else:
            cos_sim = 0.0
            
        d_m = self.compute_mahalanobis(h)
        
        # Empirical percentile
        if len(self.in_dist_mahalanobis) > 0:
            pct = float(np.mean(self.in_dist_mahalanobis <= d_m) * 100.0)
        else:
            pct = 50.0
            
        # PCA projection error
        if self.pca is not None:
            h_centered = (h - self.mean_vector).reshape(1, -1)
            h_proj = self.pca.inverse_transform(self.pca.transform(h_centered))
            pca_err = float(np.linalg.norm(h_centered - h_proj))
        else:
            pca_err = 0.0
            
        return {
            "mahalanobis_distance": d_m,
            "mahalanobis_percentile": pct,
            "norm_ratio": norm_ratio,
            "cosine_similarity": cos_sim,
            "pca_projection_error": pca_err
        }
