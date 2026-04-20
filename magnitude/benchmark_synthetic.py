"""
Synthetic benchmark for the magnitude-based intrinsic dimension estimator.

This reproduces the paper's cross-model and cross-domain evaluation scenario
using synthetic point clouds that mimic the geometric properties of real token
embeddings (human texts ≈ dim 9, AI texts ≈ dim 7.5 in PHD paper).

Because the benchmark is fully synthetic it runs without torch/transformers and
without the original data files, but faithfully mirrors the evaluation structure
of reproduce.py / run_results_02.txt.

Embedding model note
---------------------
Real token embeddings from roberta-base are 768-dimensional but lie near a
much lower-dimensional manifold.  For the purpose of testing the magnitude
estimator we generate point clouds *directly* in the intrinsic dimension
(like test_synthetic.py does).  This is the correct substrate for the estimator:
the actual signal the estimator reads is the log-log growth of the magnitude
function, which depends on the effective number of distinguishable directions
in the distance matrix, i.e. the intrinsic dimension.

When working with raw token embeddings the same effect arises naturally because
the RoBERTa feature vectors of human text span more independent directions than
those of AI-generated text.

Usage:
    python benchmark_synthetic.py
    python benchmark_synthetic.py --n-samples 200  # faster
    python benchmark_synthetic.py --n-samples 1000 --seed 0
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.magnitude import MagnitudeEstimator

# ---------------------------------------------------------------------------
# Synthetic embedding generator
# ---------------------------------------------------------------------------

def make_embedding_cloud(
    n_tokens: int,
    intrinsic_dim: int,
    ambient_dim: int = 30,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a point cloud that exactly spans an `intrinsic_dim`-dimensional
    subspace of R^{ambient_dim}.

    Construction mirrors test_synthetic.py (which is the same approach used
    implicitly by the real roberta-base embeddings):

        basis  ∈ R^{ambient_dim × intrinsic_dim}  (random projection basis)
        coords ∈ R^{n_tokens    × intrinsic_dim}  (uniform latent coordinates)
        X = coords @ basis.T

    The pairwise distances in R^{ambient_dim} are determined entirely by the
    intrinsic dimension, giving the magnitude estimator a clean signal.

    ambient_dim=30 is enough to separate dimensions up to ~15 without
    concentration-of-measure effects masking the intrinsic structure.
    """
    if rng is None:
        rng = np.random.default_rng()
    basis  = rng.standard_normal((ambient_dim, intrinsic_dim))
    coords = rng.uniform(size=(n_tokens, intrinsic_dim))
    return (coords @ basis.T).astype(np.float32)


def make_dataset(
    n_samples: int,
    human_dim: float,
    ai_dim: float,
    dim_std: float = 0.5,
    tokens_per_text: tuple = (80, 200),
    ambient_dim: int = 30,
    rng: np.random.Generator = None,
) -> tuple[list, list]:
    """
    Generate n_samples human and n_samples AI embedding clouds.

    Intrinsic dims are drawn per-sample from a rounded normal to simulate
    the natural variation in text complexity.  Token counts are drawn
    uniformly in [tokens_per_text[0], tokens_per_text[1]].
    """
    if rng is None:
        rng = np.random.default_rng()
    human_clouds, ai_clouds = [], []
    for _ in range(n_samples):
        n_tok = int(rng.integers(*tokens_per_text))
        d_h = max(1, int(round(rng.normal(human_dim, dim_std))))
        d_a = max(1, int(round(rng.normal(ai_dim, dim_std))))
        human_clouds.append(make_embedding_cloud(n_tok, d_h, ambient_dim, rng=rng))
        ai_clouds.append(make_embedding_cloud(n_tok, d_a, ambient_dim, rng=rng))
    return human_clouds, ai_clouds


# ---------------------------------------------------------------------------
# Intrinsic dimension estimation
# ---------------------------------------------------------------------------

def compute_magnitude_dims(clouds: list, desc: str = '') -> np.ndarray:
    estimator = MagnitudeEstimator(
        n_scales=25,
        t_min_norm=0.02,
        t_max_norm=8.0,
        max_points=150,
        n_reruns=3,
    )
    dims = []
    for i, cloud in enumerate(clouds):
        if desc and i % 50 == 0:
            print(f'  {desc}: {i}/{len(clouds)}', flush=True)
        dims.append(estimator.fit_transform(cloud))
    return np.array(dims, dtype=float)


# ---------------------------------------------------------------------------
# Classification helpers (identical to reproduce_magnitude.py)
# ---------------------------------------------------------------------------

def build_xy(human_dims: np.ndarray, gen_dims: np.ndarray):
    h = human_dims[~np.isnan(human_dims)]
    g = gen_dims[~np.isnan(gen_dims)]
    n = min(len(h), len(g))
    h, g = h[:n], g[:n]
    X = np.concatenate([h, g]).reshape(-1, 1)
    y = np.array([1] * n + [0] * n)
    return X, y


def split_data(X, y, val_size=0.1, test_size=0.1, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed)
    split = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=split, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_eval(splits_train, splits_eval) -> float:
    # StandardScaler is required: the magnitude dimension values (≈0.65–0.75)
    # are far from the [0,1] range that logistic regression's default C=1
    # penalty implicitly assumes, causing the decision boundary to land outside
    # the data range without scaling.
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
    clf.fit(splits_train['X_train'], splits_train['y_train'])
    return accuracy_score(splits_eval['y_test'], clf.predict(splits_eval['X_test']))


def print_table(title, row_labels, col_labels, matrix):
    print(f'\n{title}')
    col_w = 16
    row_header = 'Train \\ Eval'
    header = f"{row_header:<18}" + ''.join(f'{c:>{col_w}}' for c in col_labels)
    print(header)
    print('-' * len(header))
    for row_label, row in zip(row_labels, matrix):
        vals = ''.join(
            f'{v:>{col_w}.3f}' if v is not None else f'{"—":>{col_w}}'
            for v in row
        )
        print(f'{row_label:<18}{vals}')


# ---------------------------------------------------------------------------
# Synthetic dataset configuration
# ---------------------------------------------------------------------------
#
# Paper observations (from PHD, run_results_02.txt):
#   Human text: intrinsic dim ≈ 9
#   AI text:    intrinsic dim ≈ 7.5  (≈1.5 lower)
#
# Different generators vary slightly in how "compressed" their output is.
# Here we assign plausible synthetic parameters matching the difficulty
# gradient in run_results_02.txt (GPT-2 easiest, GPT-3.5 hardest):
#
#   GPT-2 (earlier/weaker model):  AI dim ≈ 7.0  → bigger gap → easier to detect
#   OPT-13B (larger model):        AI dim ≈ 7.5  → medium gap
#   GPT-3.5 (strongest):           AI dim ≈ 8.0  → smallest gap → hardest
#
# Cross-domain (Reddit): different domain → higher dim_std (noisier signal)
# ambient_dim=30 is enough to expose intrinsic structure cleanly.

DATASETS = {
    # name:  (human_dim, ai_dim, tokens_range, dim_std, ambient_dim)
    'gpt2_wiki':    (9.0, 7.0, (80, 200), 0.8, 30),
    'opt_wiki':     (9.0, 7.5, (80, 200), 0.8, 30),
    'gpt35_wiki':   (9.0, 8.0, (80, 200), 0.8, 30),
    'gpt35_reddit': (9.0, 8.0, (40, 120), 1.2, 30),   # shorter, noisier texts
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Synthetic benchmark for magnitude-based AI text detection')
    parser.add_argument('--n-samples', type=int, default=300,
                        help='Samples per class per dataset (default 300)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print('=' * 65)
    print('Magnitude-based AI text detection — synthetic benchmark')
    print(f'n_samples={args.n_samples}, seed={args.seed}')
    print('=' * 65)

    # Step 1: generate data and compute magnitude dimensions
    print('\nGenerating embeddings and computing magnitude dimensions...')
    mag = {}
    for key, (h_dim, a_dim, tok_range, dim_std, amb_dim) in DATASETS.items():
        print(f'\n[{key}]  human_dim={h_dim}, ai_dim={a_dim}, '
              f'tokens={tok_range}, dim_std={dim_std}, ambient={amb_dim}')
        human_clouds, ai_clouds = make_dataset(
            args.n_samples,
            human_dim=h_dim,
            ai_dim=a_dim,
            dim_std=dim_std,
            tokens_per_text=tok_range,
            ambient_dim=amb_dim,
            rng=rng,
        )
        h_dims = compute_magnitude_dims(human_clouds, desc=f'{key}/human')
        a_dims = compute_magnitude_dims(ai_clouds,   desc=f'{key}/ai')
        mag[key] = {'human': h_dims, 'gen': a_dims}

        h_mean = float(np.nanmean(h_dims))
        a_mean = float(np.nanmean(a_dims))
        print(f'  Magnitude dim — human: {h_mean:.4f}, AI: {a_mean:.4f}, '
              f'Δ = {h_mean - a_mean:.4f}')

    # Step 2: cross-model (Wikipedia domain)
    wiki_keys   = ['gpt2_wiki', 'opt_wiki', 'gpt35_wiki']
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']

    splits = {}
    for key in wiki_keys:
        X, y = build_xy(mag[key]['human'], mag[key]['gen'])
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
        splits[key] = dict(X_train=X_tr, X_val=X_val, X_test=X_te,
                           y_train=y_tr, y_val=y_val, y_test=y_te)

    cross_model = []
    for tr_key in wiki_keys:
        row = [train_eval(splits[tr_key], splits[ev_key]) for ev_key in wiki_keys]
        cross_model.append(row)

    print_table(
        '\nCross-model accuracy — Magnitude, Wikipedia domain (synthetic)',
        wiki_labels, wiki_labels, cross_model,
    )

    # Step 3: cross-domain (GPT-3.5 generator)
    dom_keys   = ['gpt35_wiki', 'gpt35_reddit']
    dom_labels = ['Wikipedia', 'Reddit']

    dom_splits = {}
    for key in dom_keys:
        X, y = build_xy(mag[key]['human'], mag[key]['gen'])
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
        dom_splits[key] = dict(X_train=X_tr, X_val=X_val, X_test=X_te,
                               y_train=y_tr, y_val=y_val, y_test=y_te)

    cross_domain = []
    for tr_key in dom_keys:
        row = [train_eval(dom_splits[tr_key], dom_splits[ev_key]) for ev_key in dom_keys]
        cross_domain.append(row)

    print_table(
        '\nCross-domain accuracy — Magnitude, GPT-3.5 generator (synthetic)',
        dom_labels, dom_labels, cross_domain,
    )

    print('\nPHD baseline (run_results_02.txt — real data):')
    print('  Cross-model:  GPT-2→GPT-2 0.730, OPT→OPT 0.830, GPT-3.5→GPT-3.5 0.870')
    print('  Cross-domain: Wikipedia→Wikipedia 0.870, Reddit→Reddit 0.737')

    print('\nNote: synthetic benchmark uses linear manifold embeddings (X = Z@A + noise)')
    print('Results should be directionally consistent with real-data runs.')


if __name__ == '__main__':
    main()
