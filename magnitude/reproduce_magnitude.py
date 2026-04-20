"""
Reproduce Table 3 from "Intrinsic Dimension Estimation for Robust Detection
of AI-Generated Texts" (arxiv 2306.04723) using the magnitude-based dimension
estimator instead of PHD.

Structure mirrors main_paper_data/reproduce.py so results are directly
comparable.  Computes both scalar (single log-log slope) and multi-scale
(6-dim) features in a single RoBERTa forward pass per text, then evaluates
both classifiers.

Usage:
    python reproduce_magnitude.py --n-samples 500
    python reproduce_magnitude.py --n-samples 500 --force-recompute
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.magnitude import MagnitudeEstimator
from src.text_utils import preprocess_text

MIN_TOKENS = 40   # skip texts with fewer usable tokens
FEAT_NAMES = ['slope_fine', 'slope_medium', 'slope_coarse',
              'slope_overall', 'curvature', 'log_mag_mid']


def process_texts(texts, tokenizer, model, desc='') -> tuple[np.ndarray, np.ndarray]:
    """
    Run RoBERTa once per text; return (scalars, feature_matrix).
    scalars       : (N,) float array of single log-log slope
    feature_matrix: (N, 6) float array of multi-scale features
    """
    estimator = MagnitudeEstimator()
    scalars = []
    features = []

    for text in tqdm(texts, desc=desc):
        inputs = tokenizer(
            preprocess_text(text),
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        with torch.no_grad():
            outp = model(**inputs)

        emb = outp[0][0].numpy()[1:-1]   # drop CLS and SEP

        if emb.shape[0] < MIN_TOKENS:
            scalars.append(float('nan'))
            features.append(np.full(6, np.nan))
        else:
            scalars.append(estimator.fit_transform(emb))
            features.append(estimator.magnitude_features(emb))

    return (
        np.array(scalars, dtype=float),
        np.array(features, dtype=float),
    )


def load_texts(path: str, n_samples: int | None = None):
    df = pd.read_json(path)
    human = df['gold_completion'].tolist()
    gen = df['gen_completion'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    ).tolist()
    n = min(len(human), len(gen))
    if n_samples:
        n = min(n, n_samples)
    return human[:n], gen[:n]


def load_or_compute(texts, data_path: str, suffix: str,
                    tokenizer, model, force: bool = False):
    scalar_path = data_path + f'.mag_{suffix}.npy'
    feat_path   = data_path + f'.magfeat_{suffix}.npy'
    if not force and os.path.exists(scalar_path) and os.path.exists(feat_path):
        print(f'  Loading cached from {scalar_path}')
        return np.load(scalar_path), np.load(feat_path)
    scalars, feats = process_texts(texts, tokenizer, model, desc=f'  [{suffix}]')
    np.save(scalar_path, scalars)
    np.save(feat_path,   feats)
    return scalars, feats


# ---------------------------------------------------------------------------
# Build (X, y) helpers
# ---------------------------------------------------------------------------

def build_xy_scalar(human_scalars, gen_scalars):
    h = human_scalars[~np.isnan(human_scalars)]
    g = gen_scalars[~np.isnan(gen_scalars)]
    n = min(len(h), len(g))
    X = np.concatenate([h[:n], g[:n]]).reshape(-1, 1)
    y = np.array([1] * n + [0] * n)
    return X, y


def build_xy_multiscale(human_feats, gen_feats):
    h = human_feats[~np.any(np.isnan(human_feats), axis=1)]
    g = gen_feats[~np.any(np.isnan(gen_feats), axis=1)]
    n = min(len(h), len(g))
    X = np.vstack([h[:n], g[:n]])
    y = np.array([1] * n + [0] * n)
    return X, y


def split_data(X, y, val_size=0.1, test_size=0.1, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed)
    split = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=split, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def train_eval(X_train, y_train, X_test, y_test) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(title, row_labels, col_labels, matrix):
    print(f'\n{title}')
    col_w = 14
    row_hdr = 'Train \\ Eval'
    header = f'{row_hdr:<16}' + ''.join(f'{c:>{col_w}}' for c in col_labels)
    print(header)
    print('-' * len(header))
    for label, row in zip(row_labels, matrix):
        vals = ''.join(f'{v:>{col_w}.3f}' for v in row)
        print(f'{label:<16}{vals}')


# ---------------------------------------------------------------------------
# Cross-model / cross-domain evaluation
# ---------------------------------------------------------------------------

def evaluate_mode(data, build_xy_fn, wiki_keys, wiki_labels,
                  domain_keys, domain_labels, title_prefix):
    # Cross-model (Wikipedia)
    splits = {}
    for key in wiki_keys:
        X, y = build_xy_fn(data[key]['human'], data[key]['gen'])
        X_tr, _, X_te, y_tr, _, y_te = split_data(X, y)
        splits[key] = (X_tr, y_tr, X_te, y_te)

    cm = []
    for tr_key in wiki_keys:
        X_tr, y_tr, _, _ = splits[tr_key]
        row = []
        for ev_key in wiki_keys:
            _, _, X_te, y_te = splits[ev_key]
            row.append(train_eval(X_tr, y_tr, X_te, y_te))
        cm.append(row)
    print_table(f'Cross-model accuracy — {title_prefix}, Wikipedia domain',
                wiki_labels, wiki_labels, cm)

    # Cross-domain (GPT-3.5)
    dom_splits = {}
    for key in domain_keys:
        X, y = build_xy_fn(data[key]['human'], data[key]['gen'])
        X_tr, _, X_te, y_tr, _, y_te = split_data(X, y)
        dom_splits[key] = (X_tr, y_tr, X_te, y_te)

    cd = []
    for tr_key in domain_keys:
        X_tr, y_tr, _, _ = dom_splits[tr_key]
        row = []
        for ev_key in domain_keys:
            _, _, X_te, y_te = dom_splits[ev_key]
            row.append(train_eval(X_tr, y_tr, X_te, y_te))
        cd.append(row)
    print_table(f'\nCross-domain accuracy — {title_prefix}, GPT-3.5 generator',
                domain_labels, domain_labels, cd)

    return cm, cd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Table 3 using magnitude dimension instead of PHD')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(__file__), '..', 'main_paper_data', 'data')
    data_dir = os.path.realpath(data_dir)

    if not os.path.isdir(data_dir):
        sys.exit(f'Data directory not found: {data_dir}\n'
                 f'Pass --data-dir PATH to specify its location.')

    files = {
        'gpt2_wiki':    os.path.join(data_dir, 'human_gpt2_wikip.json_pp'),
        'opt_wiki':     os.path.join(data_dir, 'human_opt13_wikip.json_pp'),
        'gpt35_wiki':   os.path.join(data_dir, 'human_gpt3_davinci_003_wikip.json_pp'),
        'gpt35_reddit': os.path.join(data_dir, 'human_gpt3_davinci_003_reddit.json_pp'),
    }
    for key, path in files.items():
        if not os.path.exists(path):
            sys.exit(f'Missing data file: {path}')

    print('Loading tokenizer and model: roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    model.eval()

    wiki_keys   = ['gpt2_wiki', 'opt_wiki', 'gpt35_wiki']
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']
    domain_keys   = ['gpt35_wiki', 'gpt35_reddit']
    domain_labels = ['Wikipedia', 'Reddit']

    scalars = {}
    feats   = {}

    print(f'\nProcessing up to {args.n_samples} samples per class...')
    for key, path in files.items():
        print(f'\n[{key}]')
        human_texts, gen_texts = load_texts(path, n_samples=args.n_samples)
        print(f'  {len(human_texts)} human, {len(gen_texts)} AI texts')

        suffix = f'n{args.n_samples}'
        h_sc, h_ft = load_or_compute(human_texts, path, f'human_{suffix}',
                                     tokenizer, model, args.force_recompute)
        g_sc, g_ft = load_or_compute(gen_texts,   path, f'gen_{suffix}',
                                     tokenizer, model, args.force_recompute)

        scalars[key] = {'human': h_sc, 'gen': g_sc}
        feats[key]   = {'human': h_ft, 'gen': g_ft}

        h_mean = np.nanmean(h_sc)
        g_mean = np.nanmean(g_sc)
        print(f'  Scalar mag dim — human: {h_mean:.3f}, AI: {g_mean:.3f}, '
              f'Δ={h_mean - g_mean:.3f}')

        h_fmean = np.nanmean(h_ft, axis=0)
        g_fmean = np.nanmean(g_ft, axis=0)
        print('  Feature deltas (human − AI):')
        for name, d in zip(FEAT_NAMES, h_fmean - g_fmean):
            print(f'    {name:<15}: {d:+.4f}')

    # ------------------------------------------------------------------ #
    # Evaluate both modes
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 65)
    scalar_cm, scalar_cd = evaluate_mode(
        scalars, build_xy_scalar,
        wiki_keys, wiki_labels, domain_keys, domain_labels,
        'Scalar magnitude')

    print('\n' + '=' * 65)
    multi_cm, multi_cd = evaluate_mode(
        feats, build_xy_multiscale,
        wiki_keys, wiki_labels, domain_keys, domain_labels,
        'Multi-scale magnitude (6-dim)')

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 65)
    print('SUMMARY — diagonal accuracy (train = eval on same dataset)')
    print('=' * 65)
    print(f'\n{"Dataset":<22}  {"Scalar":>10}  {"Multi-scale":>12}  {"Delta":>8}  '
          f'{"PHD (real)":>12}')
    print('-' * 72)
    phd_diag = {'GPT-2': 0.730, 'OPT': 0.830, 'GPT-3.5': 0.870,
                'Wikipedia': 0.870, 'Reddit': 0.737}
    for i, label in enumerate(wiki_labels):
        s = scalar_cm[i][i]
        m = multi_cm[i][i]
        p = phd_diag.get(label, float('nan'))
        print(f'{label:<22}  {s:>10.3f}  {m:>12.3f}  {m - s:>+8.3f}  {p:>12.3f}')
    for i, label in enumerate(domain_labels):
        s = scalar_cd[i][i]
        m = multi_cd[i][i]
        p = phd_diag.get(label, float('nan'))
        print(f'{label:<22}  {s:>10.3f}  {m:>12.3f}  {m - s:>+8.3f}  {p:>12.3f}')


if __name__ == '__main__':
    main()
