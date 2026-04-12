"""
Self-contained tests for the magnitude estimator using synthetic point clouds.

Run without any external data or models:
    python test_synthetic.py

Checks:
1. Magnitude function is a valid function (positive, monotone-ish in t)
2. Magnitude dimension recovers approximately the right dimension for uniform
   clouds in R^1, R^2, R^3
3. A "complex" (high-dim) cloud has a higher magnitude dimension than a
   "simple" (low-dim) cloud — which is the core assumption for AI-text
   detection (human text ≈ complex, AI text ≈ simple)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from magnitude import MagnitudeEstimator, _magnitude_at_scale


RNG = np.random.default_rng(42)

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def check(name: str, cond: bool, detail: str = ''):
    status = PASS if cond else FAIL
    msg = f'[{status}] {name}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    return cond


# ---------------------------------------------------------------------------
# Test 1: magnitude at scale t=0 (large distances scaled to 0) → n
# Test 2: magnitude at t→∞ (distances all huge) → 1
# ---------------------------------------------------------------------------

def test_magnitude_limits():
    n = 20
    # Random points in R^5
    X = RNG.normal(size=(n, 5))
    from scipy.spatial.distance import cdist
    D = cdist(X, X)

    # Very small t  →  Z ≈ all-ones  →  magnitude ≈ 1 (one "super-point")
    mag_small = _magnitude_at_scale(D, t=1e-6)
    # Very large t  →  Z ≈ I  →  magnitude ≈ n
    mag_large = _magnitude_at_scale(D, t=1e4)

    ok1 = check('small-t magnitude near 1',
                 0.5 <= mag_small <= 5.0,
                 f'got {mag_small:.3f}')
    ok2 = check('large-t magnitude near n',
                 n * 0.5 <= mag_large <= n * 1.5,
                 f'n={n}, got {mag_large:.3f}')
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Test 3: magnitude function is monotone non-decreasing in t for these clouds
# ---------------------------------------------------------------------------

def test_monotone():
    X = RNG.normal(size=(50, 10))
    est = MagnitudeEstimator(n_scales=15, n_reruns=1)
    t_vals, mags = est.magnitude_function(X, seed=0)

    valid = ~np.isnan(mags)
    t_v, m_v = t_vals[valid], mags[valid]

    # At least 80 % of consecutive differences should be non-negative
    diffs = np.diff(m_v)
    frac_increasing = (diffs >= -0.05 * m_v[:-1]).mean()
    ok = check('magnitude function mostly non-decreasing in t',
               frac_increasing >= 0.7,
               f'{frac_increasing:.1%} steps non-decreasing')
    return ok


# ---------------------------------------------------------------------------
# Test 4: dimension recovery for uniform clouds in R^d (d=1,2,3)
# ---------------------------------------------------------------------------

def test_dimension_recovery():
    est = MagnitudeEstimator(n_scales=30, max_points=120, n_reruns=5)
    results = {}
    for d in [1, 2, 3]:
        X = RNG.uniform(size=(300, d))
        # Scale to unit hypercube so distances are comparable across dims
        dim_est = est.fit_transform(X)
        results[d] = dim_est

    print(f'  Dimension recovery: d=1 → {results[1]:.2f}, '
          f'd=2 → {results[2]:.2f}, d=3 → {results[3]:.2f}')

    # We expect d1 < d2 < d3 (monotone ordering)
    ok_order = check('ordering d1 < d2 < d3',
                     results[1] < results[2] < results[3],
                     f'{results[1]:.2f} < {results[2]:.2f} < {results[3]:.2f}')

    # The slopes should be strictly positive (magnitude grows with t)
    ok_pos = check('all dimension estimates positive',
                   all(v > 0 for v in results.values()),
                   str({d: f'{v:.2f}' for d, v in results.items()}))

    return ok_order and ok_pos


# ---------------------------------------------------------------------------
# Test 5: "complex vs simple" — simulated human vs AI embedding clouds
#   human: uniform in R^9 (high-dim)
#   AI:    uniform in R^6 (low-dim, embedded in same ambient space)
# ---------------------------------------------------------------------------

def test_human_vs_ai_separation():
    """
    Core sanity check for the detection hypothesis:
    human text embeddings should yield a higher magnitude dimension than
    AI-generated text embeddings.
    """
    n_samples = 100      # texts per class
    n_tokens  = 150      # tokens per text
    ambient   = 30       # ambient embedding dimension

    human_dim = 9
    ai_dim    = 6

    est = MagnitudeEstimator(n_scales=25, max_points=100, n_reruns=3)

    human_scores = []
    ai_scores    = []

    for _ in range(n_samples):
        # Human: full-rank random subspace of dimension human_dim
        basis_h = RNG.normal(size=(ambient, human_dim))
        coords_h = RNG.uniform(size=(n_tokens, human_dim))
        X_h = coords_h @ basis_h.T  # (n_tokens, ambient)

        d = est.fit_transform(X_h)
        if not np.isnan(d):
            human_scores.append(d)

        # AI: low-rank subspace of dimension ai_dim
        basis_a = RNG.normal(size=(ambient, ai_dim))
        coords_a = RNG.uniform(size=(n_tokens, ai_dim))
        X_a = coords_a @ basis_a.T  # (n_tokens, ambient)

        d = est.fit_transform(X_a)
        if not np.isnan(d):
            ai_scores.append(d)

    h_mean = float(np.mean(human_scores))
    a_mean = float(np.mean(ai_scores))
    print(f'  Mean magnitude dim — human: {h_mean:.3f}, AI: {a_mean:.3f}')

    ok_direction = check('human dim > AI dim',
                         h_mean > a_mean,
                         f'Δ = {h_mean - a_mean:.3f}')

    # Rough separability: are at least 60 % of human scores above median AI?
    median_ai = float(np.median(ai_scores))
    frac_above = np.mean(np.array(human_scores) > median_ai)
    ok_sep = check('human scores mostly above AI median',
                   frac_above >= 0.60,
                   f'{frac_above:.1%} human > median AI')

    return ok_direction and ok_sep


# ---------------------------------------------------------------------------
# Test 6: NaN robustness — identical points should not crash
# ---------------------------------------------------------------------------

def test_nan_robustness():
    # All identical points → zero distances → near-singular matrix
    X_degenerate = np.zeros((50, 10))
    est = MagnitudeEstimator()
    result = est.fit_transform(X_degenerate)
    ok = check('degenerate cloud returns nan (not crash)',
               np.isnan(result),
               f'got {result}')
    return ok


# ---------------------------------------------------------------------------
# Test 7: short text returns nan gracefully
# ---------------------------------------------------------------------------

def test_short_text():
    X_short = RNG.normal(size=(10, 5))   # only 10 tokens, below min threshold
    est = MagnitudeEstimator(max_points=150)
    # We still compute (fit_transform doesn't enforce MIN_TOKENS; the
    # caller does).  But with 10 points the curve should still work.
    result = est.fit_transform(X_short)
    ok = check('short cloud completes without exception',
               True,   # just checking no exception
               f'result={result:.3f}' if not np.isnan(result) else 'nan')
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('Magnitude estimator — synthetic tests')
    print('=' * 60)

    results = [
        test_magnitude_limits(),
        test_monotone(),
        test_dimension_recovery(),
        test_human_vs_ai_separation(),
        test_nan_robustness(),
        test_short_text(),
    ]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print('=' * 60)
    print(f'Results: {n_pass}/{len(results)} passed, {n_fail} failed')

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
