import numpy as np
from scipy.spatial.distance import cdist


def magnitude_at_t(X: np.ndarray, t: float, reg: float = 1e-6) -> float:
    """
    Compute the magnitude |tA| for a set of embeddings X at scale t.

    Parameters
    ----------
    X   : (n, d) embedding matrix
    t   : scale parameter
    reg : small regularization for stability

    """
    D = cdist(X, X, metric='euclidean')  # Pairwise distances
    Z = np.exp(-t * D)
    Z += reg * np.eye(Z.shape[0])

    # Solve Z w = 1
    w = np.linalg.solve(Z, np.ones(Z.shape[0]))
    return float(w.sum())


class MagnitudeEstimator:
    def __init__(
        self,
        seed: int = 0,
        n_scales: int = 25,
        t_min_norm: float = 0.02,
        t_max_norm: float = 8.0,
        max_points: int = 150,
        n_reruns: int = 3,
    ):
        """
        Estimates the geometric "magnitude dimension" of a token-embedding cloud.

        Parameters
        ----------
        seed        : random seed for subsampling
        n_scales    : number of scale values t on a log grid
        t_min_norm  : smallest t (as a fraction of median pairwise distance)
        t_max_norm  : largest  t (as a fraction of median pairwise distance)
        max_points  : subsample size (keeps computation tractable for long texts)
        n_reruns    : number of independent subsampling reruns; result is the mean
        """
        self.rng = np.random.default_rng(seed)
        self.n_scales = n_scales
        self.t_min_norm = t_min_norm
        self.t_max_norm = t_max_norm
        self.max_points = max_points
        self.n_reruns = n_reruns

    def _magnitude_features_single(
        self,
        X: np.ndarray,
        n_scales: int,
        t_min_norm: float,
        t_max_norm: float,
        max_points: int,
    ) -> np.ndarray | None:
        """
        Subsample X, compute the magnitude function, return the log-log slope.

        The scale grid is normalised by the median pairwise distance so that
        t_min_norm=0.02 and t_max_norm=5.0 cover ~2.5 decades centred on the
        natural scale of the point cloud regardless of embedding magnitude.
        """

        n = X.shape[0]
        if n > max_points:
            X = X[self.rng.choice(n, max_points, replace=False)]

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

        mags = []
        for t in t_values:
            m = magnitude_at_t(X, t)
            assert 0.01 <= m <= len(X) * 2, "Unexpected magnitude value"
            mags.append(m)

        magnitudes = np.array(mags)

        log_t = np.log(t_values)
        log_m = np.log(magnitudes)
        n_valid = len(log_t)
        third = n_valid // 3

        def slope(lt, lm):
            assert len(lt) >= 2, "Need at least 2 points to compute slope"
            return float(np.polyfit(lt, lm, 1)[0])

        slope_overall, _ = np.polyfit(log_t, log_m, 1)
        curvature = float(np.polyfit(log_t, log_m, 2)[0]) if n_valid >= 3 else np.nan

        return np.array([
            slope(log_t[:third],        log_m[:third]),
            slope(log_t[third:2*third], log_m[third:2*third]),
            slope(log_t[2*third:],      log_m[2*third:]),
            float(slope_overall),
            curvature,
            float(log_m[n_valid // 2]),
        ], dtype=float)


    def magnitude_features(self, X: np.ndarray) -> np.ndarray:
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
        runs = [
            self._magnitude_features_single(
                X, self.n_scales, self.t_min_norm, self.t_max_norm,
                self.max_points
            )
            for _ in range(self.n_reruns)
        ]
        runs = [r for r in runs if r is not None]
        return np.nanmean(runs, axis=0) if runs else np.full(6, np.nan)
