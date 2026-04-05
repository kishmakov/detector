"""
Reproduce Table 2 from "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts"
(arxiv 2306.04723)

Reproduces:
  - Cross-model accuracy matrix (PHD column, Wikipedia domain)
  - Cross-domain accuracy matrix (PHD column, GPT-3.5 generator)

Usage:
    python reproduce.py --model-path roberta-base --n-samples 500 --data-dir data/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from IntrinsicDim import PHD

MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7


def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')


def get_phd_single(text, solver, tokenizer, model):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outp = model(**inputs)

    # Drop CLS and SEP tokens
    mx_points = inputs['input_ids'].shape[1] - 2
    mn_points = MIN_SUBSAMPLE

    if mx_points <= mn_points + INTERMEDIATE_POINTS:
        return float('nan')

    step = (mx_points - mn_points) // INTERMEDIATE_POINTS
    embeddings = outp[0][0].numpy()[1:-1]

    return solver.fit_transform(
        embeddings,
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step,
    )


def compute_phd_array(texts, tokenizer, model, desc=""):
    solver = PHD(alpha=1.0, metric='euclidean', n_points=9)
    dims = []
    for text in tqdm(texts, desc=desc):
        dims.append(get_phd_single(text, solver, tokenizer, model))
    return np.array(dims, dtype=float)


def load_texts(path, n_samples=None):
    # Each file is a single JSON array (not JSONL)
    df = pd.read_json(path)
    human = df['gold_completion'].tolist()
    gen = df['gen_completion'].apply(lambda x: x[0] if isinstance(x, list) else x).tolist()
    n = min(len(human), len(gen))
    if n_samples:
        n = min(n, n_samples)
    return human[:n], gen[:n]


def cache_path(data_path, suffix):
    return data_path + f'.phd_{suffix}.npy'


def load_or_compute(texts, data_path, suffix, tokenizer, model, force=False):
    cp = cache_path(data_path, suffix)
    if not force and os.path.exists(cp):
        print(f"  Loading cached PHD from {cp}")
        return np.load(cp)
    dims = compute_phd_array(texts, tokenizer, model, desc=f"  PHD [{suffix}]")
    np.save(cp, dims)
    return dims


def build_xy(human_phd, gen_phd):
    # Remove NaNs
    h = human_phd[~np.isnan(human_phd)]
    g = gen_phd[~np.isnan(gen_phd)]
    n = min(len(h), len(g))
    h, g = h[:n], g[:n]
    X = np.concatenate([h, g]).reshape(-1, 1)
    y = np.array([1] * n + [0] * n)
    return X, y


def train_classifier(X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)


def split_data(X, y, val_size=0.1, test_size=0.1, seed=42):
    # 80/10/10
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed
    )
    split = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=split, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_table(title, row_labels, col_labels, matrix):
    print(f"\n{title}")
    col_w = 14
    header = f"{'Train \\ Eval':<16}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(header)
    print("-" * len(header))
    for row_label, row in zip(row_labels, matrix):
        vals = "".join(f"{v:>{col_w}.3f}" if v is not None else f"{'—':>{col_w}}" for v in row)
        print(f"{row_label:<16}{vals}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 of PHD paper")
    parser.add_argument('--model-path', default='roberta-base',
                        help='HuggingFace model name or local path (default: roberta-base)')
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing the .json_pp data files')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Max samples per class per dataset (default: 500)')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Ignore cached PHD .npy files and recompute')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        sys.exit(f"Data directory not found: {data_dir}")

    # Data file paths
    files = {
        'gpt2_wiki':   os.path.join(data_dir, 'human_gpt2_wikip.json_pp'),
        'opt_wiki':    os.path.join(data_dir, 'human_opt13_wikip.json_pp'),
        'gpt35_wiki':  os.path.join(data_dir, 'human_gpt3_davinci_003_wikip.json_pp'),
        'gpt35_reddit': os.path.join(data_dir, 'human_gpt3_davinci_003_reddit.json_pp'),
    }

    for key, path in files.items():
        if not os.path.exists(path):
            sys.exit(f"Missing data file: {path}")

    print(f"Loading tokenizer and model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()

    # ------------------------------------------------------------------ #
    # Step 1: Load texts and compute (or load cached) PHD arrays
    # ------------------------------------------------------------------ #
    print(f"\nComputing PHD for up to {args.n_samples} samples per class...")

    phd = {}
    for key, path in files.items():
        print(f"\n[{key}] Loading {path}")
        human_texts, gen_texts = load_texts(path, n_samples=args.n_samples)
        print(f"  {len(human_texts)} human, {len(gen_texts)} AI texts")

        phd[key] = {
            'human': load_or_compute(human_texts, path, f'human_n{args.n_samples}',
                                     tokenizer, model, force=args.force_recompute),
            'gen':   load_or_compute(gen_texts,   path, f'gen_n{args.n_samples}',
                                     tokenizer, model, force=args.force_recompute),
        }

        h_mean = np.nanmean(phd[key]['human'])
        g_mean = np.nanmean(phd[key]['gen'])
        print(f"  PHD mean — human: {h_mean:.2f}, AI: {g_mean:.2f}")

    # ------------------------------------------------------------------ #
    # Step 2: Cross-model accuracy (Wikipedia domain, Table 2 lower)
    # ------------------------------------------------------------------ #
    wiki_keys   = ['gpt2_wiki', 'opt_wiki', 'gpt35_wiki']
    wiki_labels = ['GPT-2', 'OPT', 'GPT-3.5']

    # Build X, y and split for each Wikipedia dataset
    splits = {}
    for key in wiki_keys:
        X, y = build_xy(phd[key]['human'], phd[key]['gen'])
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        splits[key] = dict(X_train=X_train, X_val=X_val, X_test=X_test,
                           y_train=y_train, y_val=y_val, y_test=y_test)

    cross_model_matrix = []
    for train_key in wiki_keys:
        row = []
        clf = train_classifier(splits[train_key]['X_train'], splits[train_key]['y_train'])
        for eval_key in wiki_keys:
            acc = evaluate(clf, splits[eval_key]['X_test'], splits[eval_key]['y_test'])
            row.append(acc)
        cross_model_matrix.append(row)

    print_table(
        "Table 2 (lower): Cross-model accuracy — PHD classifier, Wikipedia domain",
        wiki_labels, wiki_labels, cross_model_matrix
    )

    # ------------------------------------------------------------------ #
    # Step 3: Cross-domain accuracy (GPT-3.5 generator, Table 2 upper)
    # ------------------------------------------------------------------ #
    domain_keys   = ['gpt35_wiki', 'gpt35_reddit']
    domain_labels = ['Wikipedia', 'Reddit']

    domain_splits = {}
    for key in domain_keys:
        X, y = build_xy(phd[key]['human'], phd[key]['gen'])
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
        "\nTable 3 (upper, partial): Cross-domain accuracy — PHD classifier, GPT-3.5 generator",
        domain_labels, domain_labels, cross_domain_matrix
    )

    print("\nPaper reference values (Table 3):")
    print("  Cross-domain PHD: Wikipedia→Wikipedia 0.843, Reddit→Reddit 0.776, cross ~0.78–0.86")
    print("  Cross-model PHD:  diagonal ~0.76–0.84, off-diagonal ~0.76–0.84 (very stable)")


if __name__ == '__main__':
    main()
