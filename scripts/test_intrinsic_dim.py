import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_regression import make_dataset, train_eval
from main_paper_data.IntrinsicDim import PHD

ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = ROOT / "tmp"

_phd = PHD()

MIN_TOKENS = 51  # need range(50, n_tokens, 40) be non-empty


def collect_phd_features(dataset, cache_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows_path = TEMP_DIR / f"{cache_name}_rows.npy"
    labels_path = TEMP_DIR / f"{cache_name}_labels.npy"

    if rows_path.exists() and labels_path.exists():
        print(f"Loading cached features from {rows_path}")
        X, y = np.load(rows_path), np.load(labels_path)
        print(f"Loaded {len(X)} examples from disk")
        return X, y

    n_texts = int(sys.argv[1]) if len(sys.argv) > 1 else len(dataset)
    n_texts = min(n_texts, len(dataset))
    print(f"Going to process {n_texts} texts from the dataset (cache: {cache_name})...")

    rows, labels = [], []
    skipped = 0
    for i in tqdm(range(n_texts)):
        emb, label = dataset[i]
        if emb.shape[0] < MIN_TOKENS:
            skipped += 1
            continue
        dim = _phd.fit_transform(emb, max_points=emb.shape[0])
        rows.append([dim])
        labels.append(label)

    if skipped:
        print(f"Skipped {skipped} texts with fewer than {MIN_TOKENS} tokens")

    X, y = np.array(rows), np.array(labels)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(rows_path, X)
    np.save(labels_path, y)
    print(f"Saved data to {cache_name}_*.npy")
    return X, y


def run_experiment(train, test, model):
    suffix = f"_{model}" if model else ""
    print(f"\n=== train={train}, test={test}, model={model or 'all'} ===")
    ds_train = make_dataset(source=train, model=model)
    ds_test  = make_dataset(source=test,  model=model)
    X_train, y_train = collect_phd_features(ds_train, f"{train}_phd{suffix}")
    X_test,  y_test  = collect_phd_features(ds_test,  f"{test}_phd{suffix}")
    train_eval(X_train, y_train, X_test, y_test)


experiments = [
    ("reddit", "wiki",   None),           #1: train=reddit, test=wiki, completions=all
    ("wiki",   "reddit", None),           #2: train=wiki, test=reddit, completions=all
    ("reddit", "wiki",   "gpt3"),         #3: train=reddit, test=wiki, completions=gpt3
    ("wiki",   "reddit", "gpt3"),         #4: train=wiki, test=reddit, completions=gpt3
    ("reddit", "wiki",   "gpt-5.4-mini"), #5: train=reddit, test=wiki, completions=gpt-5.4-mini
    ("wiki",   "reddit", "gpt-5.4-mini"), #6: train=wiki, test=reddit, completions=gpt-5.4-mini
    ("reddit", "wiki",   "gemini-3.1-pro"), #7: train=reddit, test=wiki, completions=gemini-3.1-pro
    ("wiki",   "reddit", "gemini-3.1-pro"), #8: train=wiki, test=reddit, completions=gemini-3.1-pro
]

for train, test, model in experiments:
    run_experiment(train, test, model)
