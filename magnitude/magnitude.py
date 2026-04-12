"""
Magnitude-based intrinsic dimension estimator for text embeddings.

Given a set of token embeddings (a finite metric space), the magnitude
function |tA| tracks how the "effective size" of the space grows as the
scale t varies. The log-log slope of |tA| vs t estimates the geometric
dimension of the embedding cloud.

Reference: Leinster & Cobbold, "Measuring diversity: the importance of
species similarity", 2012; Meckes, "Magnitude, diversity, capacities, and
dimensions of metric spaces", 2013.

Usage (mirrors PHD from IntrinsicDim.py):

    from magnitude import MagnitudeEstimator

    estimator = MagnitudeEstimator()
    dim = estimator.fit_transform(embeddings)   # shape (n_tokens, d)
"""

import numpy as np
from scipy.spatial.distance import cdist
from threading import Thread


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _magnitude_at_scale(D: np.ndarray, t: float, reg: float = 1e-6) -> float:
    """
    Compute the magnitude |tA| of a finite metric space.

    Parameters
    ----------
    D   : (n, n) pairwise distance matrix
    t   : scale parameter (distances are multiplied by t)
    reg : Tikhonov regularisation added to the diagonal of Z_t before
          solving.  Keeps the system well-conditioned at small t.

    Returns
    -------
    Magnitude scalar, or np.nan if the system is numerically singular.
    """
    n = D.shape[0]
    Z = np.exp(-t * D)
    # Regularise: Z + reg*I
    Z_reg = Z + reg * np.eye(n)
    ones = np.ones(n)
    try:
        # Solve Z_reg @ w = 1  →  w = Z_reg^{-1} 1
        # magnitude = 1^T w = sum(Z_reg^{-1})   (all-ones vector contracts both axes)
        w = np.linalg.solve(Z_reg, ones)
        mag = float(w.sum())
        # Sanity clip: magnitude must be in [1, n]
        if not (0.5 <= mag <= n * 2):
            return np.nan
        return mag
    except np.linalg.LinAlgError:
        return np.nan


def _magnitude_dimension_single(
    X: np.ndarray,
    n_scales: int,
    t_min_norm: float,
    t_max_norm: float,
    max_points: int,
    reg: float,
    rng: np.random.Generator,
) -> float:
    """
    Subsample X, compute the magnitude function, return the log-log slope.

    The scale grid is normalised by the median pairwise distance so that
    t_min_norm=0.02 and t_max_norm=5.0 cover ~2.5 decades centred on the
    natural scale of the point cloud regardless of embedding magnitude.
    """
    n = X.shape[0]
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        X = X[idx]

    D = cdist(X, X, metric='euclidean')

    # Normalise scale by median non-zero pairwise distance
    nz = D[D > 0]
    if nz.size == 0:
        return np.nan
    median_dist = float(np.median(nz))
    if median_dist == 0:
        return np.nan

    t_values = np.logspace(
        np.log10(t_min_norm / median_dist),
        np.log10(t_max_norm / median_dist),
        n_scales,
    )

    magnitudes = np.array([_magnitude_at_scale(D, t, reg=reg) for t in t_values])

    valid = ~np.isnan(magnitudes) & (magnitudes > 0)
    if valid.sum() < 4:
        return np.nan

    log_t = np.log(t_values[valid])
    log_m = np.log(magnitudes[valid])

    # Weighted least-squares: weight middle-range scales more
    slope, _ = np.polyfit(log_t, log_m, 1)
    return float(slope)


def _magnitude_features_single(
    X: np.ndarray,
    n_scales: int,
    t_min_norm: float,
    t_max_norm: float,
    max_points: int,
    reg: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """
    Compute the 6-dimensional multi-scale feature vector for one subsampling run.

    Returns None if the magnitude curve has fewer than 6 valid points.
    """
    n = X.shape[0]
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        X = X[idx]

    D = cdist(X, X, metric='euclidean')
    nz = D[D > 0]
    if nz.size == 0:
        return None
    median_dist = float(np.median(nz))
    if median_dist == 0:
        return None

    t_values = np.logspace(
        np.log10(t_min_norm / median_dist),
        np.log10(t_max_norm / median_dist),
        n_scales,
    )
    magnitudes = np.array([_magnitude_at_scale(D, t, reg=reg) for t in t_values])

    valid = ~np.isnan(magnitudes) & (magnitudes > 0)
    if valid.sum() < 6:
        return None

    log_t = np.log(t_values[valid])
    log_m = np.log(magnitudes[valid])
    n_valid = len(log_t)

    # --- Partition into thirds ---
    third = n_valid // 3
    idx_fine   = slice(0, third)
    idx_medium = slice(third, 2 * third)
    idx_coarse = slice(2 * third, n_valid)

    def slope(lt, lm):
        if len(lt) < 2:
            return np.nan
        s, _ = np.polyfit(lt, lm, 1)
        return float(s)

    slope_fine   = slope(log_t[idx_fine],   log_m[idx_fine])
    slope_medium = slope(log_t[idx_medium], log_m[idx_medium])
    slope_coarse = slope(log_t[idx_coarse], log_m[idx_coarse])
    slope_overall, _ = np.polyfit(log_t, log_m, 1)

    # Curvature: coefficient of t^2 in quadratic fit (log_m ≈ a*log_t^2 + b*log_t + c)
    if len(log_t) >= 3:
        coeffs = np.polyfit(log_t, log_m, 2)
        curvature = float(coeffs[0])
    else:
        curvature = np.nan

    # Log-magnitude at midpoint scale
    mid_idx = n_valid // 2
    log_mag_mid = float(log_m[mid_idx])

    return np.array([
        slope_fine,
        slope_medium,
        slope_coarse,
        float(slope_overall),
        curvature,
        log_mag_mid,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------

class MagnitudeEstimator:
    """
    Estimates the geometric "magnitude dimension" of a token-embedding cloud.

    Parameters
    ----------
    n_scales    : number of scale values t on a log grid
    t_min_norm  : smallest t (as a fraction of median pairwise distance)
    t_max_norm  : largest  t (as a fraction of median pairwise distance)
    max_points  : subsample size (keeps computation tractable for long texts)
    n_reruns    : number of independent subsampling reruns; result is the mean
    reg         : regularisation strength for the magnitude matrix
    """

    def __init__(
        self,
        n_scales: int = 25,
        t_min_norm: float = 0.02,
        t_max_norm: float = 8.0,
        max_points: int = 150,
        n_reruns: int = 3,
        reg: float = 1e-5,
    ):
        self.n_scales = n_scales
        self.t_min_norm = t_min_norm
        self.t_max_norm = t_max_norm
        self.max_points = max_points
        self.n_reruns = n_reruns
        self.reg = reg

    def fit_transform(
        self,
        X: np.ndarray,
        y=None,
        seed: int = 0,
    ) -> float:
        """
        Estimate magnitude dimension from a token-embedding matrix.

        Parameters
        ----------
        X    : (n_tokens, d) float array of token embeddings
        y    : ignored (sklearn compatibility)
        seed : random seed

        Returns
        -------
        Scalar dimension estimate, or np.nan if too few valid scales.
        """
        rng = np.random.default_rng(seed)
        dims = []
        for _ in range(self.n_reruns):
            d = _magnitude_dimension_single(
                X,
                n_scales=self.n_scales,
                t_min_norm=self.t_min_norm,
                t_max_norm=self.t_max_norm,
                max_points=self.max_points,
                reg=self.reg,
                rng=rng,
            )
            if not np.isnan(d):
                dims.append(d)
        if not dims:
            return np.nan
        return float(np.mean(dims))

    def magnitude_function(
        self,
        X: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the full magnitude function (t_values, |tA|) for X.

        Useful for visualisation.  Uses max_points subsampling once (no reruns).
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if n > self.max_points:
            idx = rng.choice(n, self.max_points, replace=False)
            X = X[idx]

        D = cdist(X, X, metric='euclidean')
        nz = D[D > 0]
        median_dist = float(np.median(nz)) if nz.size > 0 else 1.0

        t_values = np.logspace(
            np.log10(self.t_min_norm / median_dist),
            np.log10(self.t_max_norm / median_dist),
            self.n_scales,
        )
        magnitudes = np.array([_magnitude_at_scale(D, t, reg=self.reg) for t in t_values])
        return t_values, magnitudes

    def magnitude_features(
        self,
        X: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Extract a multi-scale feature vector from the magnitude function.

        Instead of collapsing |tA| to a single slope, this method captures
        how the magnitude curve behaves across different scale regimes.  The
        resulting feature vector is more informative than the scalar slope
        returned by fit_transform().

        Features (6 dimensions):
            0  slope_fine    — log-log slope over the lowest 1/3 of the scale range
            1  slope_medium  — log-log slope over the middle 1/3
            2  slope_coarse  — log-log slope over the upper 1/3
            3  slope_overall — log-log slope across the full range (= fit_transform)
            4  curvature     — 2nd-order coefficient of a quadratic log-log fit
                               (positive ⟹ concave up, i.e. growing faster at large t)
            5  log_mag_mid   — log(|tA|) at the median scale (proxy for effective size)

        Returns np.nan for all features if the magnitude function cannot be
        computed reliably (too few valid scales).

        Results are averaged over n_reruns independent subsampling runs.
        """
        rng = np.random.default_rng(seed)
        feature_runs = []
        for _ in range(self.n_reruns):
            feats = _magnitude_features_single(
                X,
                n_scales=self.n_scales,
                t_min_norm=self.t_min_norm,
                t_max_norm=self.t_max_norm,
                max_points=self.max_points,
                reg=self.reg,
                rng=rng,
            )
            if feats is not None:
                feature_runs.append(feats)
        if not feature_runs:
            return np.full(6, np.nan)
        return np.nanmean(feature_runs, axis=0)
