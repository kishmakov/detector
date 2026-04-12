"""
Multi-scale magnitude feature benchmark for AI text detection.

Compares two classifiers:
  1. Single-feature (overall log-log slope, as in benchmark_synthetic.py)
  2. Multi-scale features (6-dim vector: slopes at fine/medium/coarse scales,
     overall slope, curvature, and log-magnitude at midpoint scale)

The idea is that the magnitude curve encodes richer information than the
single slope: human text embeddings may differ from AI embeddings not only
in the average slope but in how the slope varies across scales and in the
curvature of the log-log curve.

Reference: magnitude.md — "slope at small/medium/large scales, curvature
of the log-log magnitude curve, thresholded magnitude values at selected t"

Usage:
    python multiscale_benchmark.py
    python multiscale_benchmark.py --n-samples 200   # faster
    python multiscale_benchmark.py --n-samples 500 --seed 0
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

sys.path.insert(0, os.path.dirname(__file__))
from magnitude import MagnitudeEstimator


# ---------------------------------------------------------------------------
# Synthetic embedding generator (same as benchmark_synthetic.py)
# ---------------------------------------------------------------------------

def make_embedding_cloud(n_tokens, intrinsic_dim, ambient_dim=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    basis  = rng.standard_normal((ambient_dim, intrinsic_dim))
    coords = rng.uniform(size=(n_tokens, intrinsic_dim))
    return (coords @ basis.T).astype(np.float32)


def make_dataset(n_samples, human_dim, ai_dim, dim_std=0.5,
                 tokens_per_text=(80, 200), ambient_dim=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    human_clouds, ai_clouds = [], []
    for _ in range(n_samples):
        n_tok = int(rng.integers(*tokens_per_text))
        d_h = max(1, int(round(rng.normal(human_dim, dim_std))))
        d_a = max(1, int(round(rng.normal(ai_dim,    dim_std))))
        human_clouds.append(make_embedding_cloud(n_tok, d_h, ambient_dim, rng=rng))
        ai_clouds.append(make_embedding_cloud(n_tok,   d_a, ambient_dim, rng=rng))
    return human_clouds, ai_clouds


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_features(clouds, use_multiscale=True, desc=''):
    """Compute feature matrix for a list of embedding clouds."""
    est = MagnitudeEstimator(n_scales=30, t_min_norm=0.02, t_max_norm=8.0,
                             max_points=150, n_reruns=3)
    rows = []
    for i, cloud in enumerate(clouds):
        if desc and i % 50 == 0:
            print(f'  {desc}: {i}/{len(clouds)}', flush=True)
        if use_multiscale:
            feats = est.magnitude_features(cloud)   # shape (6,)
        else:
            scalar = est.fit_transform(cloud)        # shape ()
            feats = np.array([scalar])
        rows.append(feats)
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def build_xy(human_feats, ai_feats):
    """Stack human and AI features; drop rows with any NaN."""
    h = human_feats[~np.any(np.isnan(human_feats), axis=1)]
    a = ai_feats[~np.any(np.isnan(ai_feats),   axis=1)]
    n = min(len(h), len(a))
    X = np.vstack([h[:n], a[:n]])
    y = np.array([1] * n + [0] * n)
    return X, y


def split_data(X, y, val_size=0.1, test_size=0.1, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed)
    split = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=split, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_eval(splits_train, splits_eval):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(splits_train['X_train'], splits_train['y_train'])
    return accuracy_score(splits_eval['y_test'], clf.predict(splits_eval['X_test']))


# ---------------------------------------------------------------------------
# Dataset config (mirrored from benchmark_synthetic.py)
# ---------------------------------------------------------------------------

DATASETS = {
    'gpt2_wiki':    (9.0, 7.0, (80, 200), 0.8, 30),
    'opt_wiki':     (9.0, 7.5, (80, 200), 0.8, 30),
    'gpt35_wiki':   (9.0, 8.0, (80, 200), 0.8, 30),
    'gpt35_reddit': (9.0, 8.0, (40, 120), 1.2, 30),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scenario(n_samples, seed, use_multiscale):
    tag = 'multi-scale (6-dim)' if use_multiscale else 'single slope'
    print(f'\n{"=" * 65}')
    print(f'Scenario: {tag}')
    print(f'n_samples={n_samples}, seed={seed}')
    print('=' * 65)

    rng = np.random.default_rng(seed)
    feats = {}

    print('\nGenerating embeddings and computing features...')
    for key, (h_dim, a_dim, tok_range, dim_std, amb_dim) in DATASETS.items():
        print(f'\n[{key}]  human_dim={h_dim}, ai_dim={a_dim}')
        human_clouds, ai_clouds = make_dataset(
            n_samples, h_dim, a_dim, dim_std, tok_range, amb_dim, rng=rng)
        h_feats = compute_features(human_clouds, use_multiscale,
                                   desc=f'{key}/human')
        a_feats = compute_features(ai_clouds,    use_multiscale,
                                   desc=f'{key}/ai')
        feats[key] = {'human': h_feats, 'gen': a_feats}

        # Report per-feature mean difference for multi-scale
        if use_multiscale:
            h_mean = np.nanmean(h_feats, axis=0)
            a_mean = np.nanmean(a_feats, axis=0)
            delta  = h_mean - a_mean
            names  = ['slope_fine', 'slope_medium', 'slope_coarse',
                      'slope_overall', 'curvature', 'log_mag_mid']
            print('  Feature deltas (human − AI):')
            for name, d in zip(names, delta):
                print(f'    {name:<15}: {d:+.4f}')
        else:
            h_m = float(np.nanmean(h_feats))
            a_m = float(np.nanmean(a_feats))
            print(f'  slope_overall — human: {h_m:.4f}, AI: {a_m:.4f}, '
                  f'Δ={h_m - a_m:.4f}')

    # Cross-model (Wikipedia)
    wiki_keys   = ['gpt2_wiki', 'opt_wiki', 'gpt35_wiki']
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']

    splits = {}
    for key in wiki_keys:
        X, y = build_xy(feats[key]['human'], feats[key]['gen'])
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
        splits[key] = dict(X_train=X_tr, X_val=X_val, X_test=X_te,
                           y_train=y_tr, y_val=y_val, y_test=y_te)

    print(f'\nCross-model accuracy — {tag}, Wikipedia domain:')
    col_w = 14
    row_hdr = 'Train \\ Eval'
    header = f'{row_hdr:<18}' + ''.join(f'{c:>{col_w}}' for c in wiki_labels)
    print(header)
    print('-' * len(header))
    cross_model = []
    for tr_key in wiki_keys:
        row = [train_eval(splits[tr_key], splits[ev_key]) for ev_key in wiki_keys]
        cross_model.append(row)
        label = wiki_labels[wiki_keys.index(tr_key)]
        vals  = ''.join(f'{v:>{col_w}.3f}' for v in row)
        print(f'{label:<18}{vals}')

    # Cross-domain (GPT-3.5)
    dom_keys   = ['gpt35_wiki', 'gpt35_reddit']
    dom_labels = ['Wikipedia', 'Reddit']

    dom_splits = {}
    for key in dom_keys:
        X, y = build_xy(feats[key]['human'], feats[key]['gen'])
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
        dom_splits[key] = dict(X_train=X_tr, X_val=X_val, X_test=X_te,
                               y_train=y_tr, y_val=y_val, y_test=y_te)

    print(f'\nCross-domain accuracy — {tag}, GPT-3.5 generator:')
    header = f'{row_hdr:<18}' + ''.join(f'{c:>{col_w}}' for c in dom_labels)
    print(header)
    print('-' * len(header))
    cross_domain = []
    for tr_key in dom_keys:
        row = [train_eval(dom_splits[tr_key], dom_splits[ev_key]) for ev_key in dom_keys]
        cross_domain.append(row)
        label = dom_labels[dom_keys.index(tr_key)]
        vals  = ''.join(f'{v:>{col_w}.3f}' for v in row)
        print(f'{label:<18}{vals}')

    return cross_model, cross_domain


def main():
    parser = argparse.ArgumentParser(
        description='Multi-scale magnitude feature benchmark')
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('Multi-scale magnitude features — comparison benchmark')
    print(f'n_samples={args.n_samples}, seed={args.seed}')

    cm_single,  cd_single  = run_scenario(args.n_samples, args.seed, use_multiscale=False)
    cm_multi,   cd_multi   = run_scenario(args.n_samples, args.seed, use_multiscale=True)

    # Summary comparison
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']
    dom_labels  = ['Wikipedia', 'Reddit']

    print('\n' + '=' * 65)
    print('SUMMARY: diagonal accuracy (train = eval)')
    print('=' * 65)
    print(f'\n{"Dataset":<20}  {"Single slope":>14}  {"Multi-scale":>12}  {"Delta":>8}')
    print('-' * 60)
    for i, label in enumerate(wiki_labels):
        s = cm_single[i][i]
        m = cm_multi[i][i]
        print(f'{label:<20}  {s:>14.3f}  {m:>12.3f}  {m - s:>+8.3f}')
    for i, label in enumerate(dom_labels):
        s = cd_single[i][i]
        m = cd_multi[i][i]
        print(f'{label:<20}  {s:>14.3f}  {m:>12.3f}  {m - s:>+8.3f}')

    print('\nPHD baseline (run_results_02.txt — real data):')
    print('  Cross-model:  GPT-2 0.730, OPT 0.830, GPT-3.5 0.870')
    print('  Cross-domain: Wikipedia 0.870, Reddit 0.737')


if __name__ == '__main__':
    main()
