"""
Reproduce Table 3 from "Intrinsic Dimension Estimation for Robust Detection
of AI-Generated Texts" (arxiv 2306.04723) using the magnitude-based dimension
estimator instead of PHD.

Structure mirrors main_paper_data/reproduce.py so results are directly
comparable.

Usage:
    python reproduce_magnitude.py --n-samples 500
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
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from magnitude import MagnitudeEstimator

MIN_TOKENS = 40   # skip texts with fewer usable tokens


def preprocess_text(text: str) -> str:
    return text.replace('\n', ' ').replace('  ', ' ')


def get_magnitude_single(text: str, estimator: MagnitudeEstimator,
                         tokenizer, model) -> float:
    """
    Tokenise text with roberta-base, extract token embeddings, compute
    magnitude dimension.  Returns nan for texts that are too short.
    """
    inputs = tokenizer(
        preprocess_text(text),
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )
    with torch.no_grad():
        outp = model(**inputs)

    # Drop CLS and SEP tokens (first and last)
    embeddings = outp[0][0].numpy()[1:-1]   # (n_tokens, 768)

    if embeddings.shape[0] < MIN_TOKENS:
        return float('nan')

    return estimator.fit_transform(embeddings)


def compute_magnitude_array(texts, tokenizer, model, desc='') -> np.ndarray:
    estimator = MagnitudeEstimator()
    dims = []
    for text in tqdm(texts, desc=desc):
        dims.append(get_magnitude_single(text, estimator, tokenizer, model))
    return np.array(dims, dtype=float)


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
                    tokenizer, model, force: bool = False) -> np.ndarray:
    cached = data_path + f'.mag_{suffix}.npy'
    if not force and os.path.exists(cached):
        print(f'  Loading cached magnitude from {cached}')
        return np.load(cached)
    dims = compute_magnitude_array(texts, tokenizer, model,
                                   desc=f'  Magnitude [{suffix}]')
    np.save(cached, dims)
    return dims


def build_xy(human_dims: np.ndarray, gen_dims: np.ndarray):
    h = human_dims[~np.isnan(human_dims)]
    g = gen_dims[~np.isnan(gen_dims)]
    n = min(len(h), len(g))
    h, g = h[:n], g[:n]
    X = np.concatenate([h, g]).reshape(-1, 1)
    y = np.array([1] * n + [0] * n)
    return X, y


def split_data(X, y, val_size=0.1, test_size=0.1, seed=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed)
    split = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=split, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_classifier(X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test) -> float:
    return accuracy_score(y_test, clf.predict(X_test))


def print_table(title, row_labels, col_labels, matrix):
    print(f'\n{title}')
    col_w = 14
    header = f"{'Train \\ Eval':<16}" + ''.join(f'{c:>{col_w}}' for c in col_labels)
    print(header)
    print('-' * len(header))
    for row_label, row in zip(row_labels, matrix):
        vals = ''.join(
            f'{v:>{col_w}.3f}' if v is not None else f'{"—":>{col_w}}'
            for v in row
        )
        print(f'{row_label:<16}{vals}')


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Table 3 using magnitude dimension instead of PHD')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--data-dir', default=None,
                        help='Path to data directory (default: sibling of this file)')
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

    model_path = 'roberta-base'
    print(f'Loading tokenizer and model: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # ------------------------------------------------------------------ #
    # Step 1: compute (or load cached) magnitude dimension arrays
    # ------------------------------------------------------------------ #
    print(f'\nComputing magnitude dimension for up to {args.n_samples} samples per class...')

    mag = {}
    for key, path in files.items():
        print(f'\n[{key}] Loading {path}')
        human_texts, gen_texts = load_texts(path, n_samples=args.n_samples)
        print(f'  {len(human_texts)} human, {len(gen_texts)} AI texts')

        mag[key] = {
            'human': load_or_compute(
                human_texts, path, f'human_n{args.n_samples}',
                tokenizer, model, force=args.force_recompute),
            'gen': load_or_compute(
                gen_texts, path, f'gen_n{args.n_samples}',
                tokenizer, model, force=args.force_recompute),
        }

        h_mean = np.nanmean(mag[key]['human'])
        g_mean = np.nanmean(mag[key]['gen'])
        print(f'  Magnitude dim mean — human: {h_mean:.3f}, AI: {g_mean:.3f}')

    # ------------------------------------------------------------------ #
    # Step 2: Cross-model accuracy (Wikipedia domain)
    # ------------------------------------------------------------------ #
    wiki_keys   = ['gpt2_wiki', 'opt_wiki', 'gpt35_wiki']
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']

    splits = {}
    for key in wiki_keys:
        X, y = build_xy(mag[key]['human'], mag[key]['gen'])
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        splits[key] = dict(X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)

    cross_model_matrix = []
    for train_key in wiki_keys:
        row = []
        clf = train_classifier(splits[train_key]['X_train'],
                               splits[train_key]['y_train'])
        for eval_key in wiki_keys:
            acc = evaluate(clf, splits[eval_key]['X_test'],
                           splits[eval_key]['y_test'])
            row.append(acc)
        cross_model_matrix.append(row)

    print_table(
        'Cross-model accuracy — Magnitude classifier, Wikipedia domain',
        wiki_labels, wiki_labels, cross_model_matrix,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Cross-domain accuracy (GPT-3.5 generator)
    # ------------------------------------------------------------------ #
    domain_keys   = ['gpt35_wiki', 'gpt35_reddit']
    domain_labels = ['Wikipedia', 'Reddit']

    domain_splits = {}
    for key in domain_keys:
        X, y = build_xy(mag[key]['human'], mag[key]['gen'])
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        domain_splits[key] = dict(X_train=X_train, X_val=X_val, X_test=X_test,
                                  y_train=y_train, y_val=y_val, y_test=y_test)

    cross_domain_matrix = []
    for train_key in domain_keys:
        row = []
        clf = train_classifier(domain_splits[train_key]['X_train'],
                               domain_splits[train_key]['y_train'])
        for eval_key in domain_keys:
            acc = evaluate(clf, domain_splits[eval_key]['X_test'],
                           domain_splits[eval_key]['y_test'])
            row.append(acc)
        cross_domain_matrix.append(row)

    print_table(
        '\nCross-domain accuracy — Magnitude classifier, GPT-3.5 generator',
        domain_labels, domain_labels, cross_domain_matrix,
    )

    print('\nBaseline PHD results (from run_results_02.txt):')
    print('  Cross-domain: Wikipedia→Wikipedia 0.870, Reddit→Reddit 0.737')
    print('  Cross-model:  GPT-2→GPT-2 0.730, OPT→OPT 0.830, GPT-3.5→GPT-3.5 0.870')


if __name__ == '__main__':
    main()
