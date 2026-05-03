import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import cross_validate as _cv
from tqdm import tqdm

from .embeddings_dataset import EmbeddingsDataset
from .embeddings_provider import embeddings_provider_by_name


def make_dataset(*, name: str = "bert-base-uncased", source=None, model=None) -> EmbeddingsDataset:
    root = Path(os.environ.get("DETECTOR_ROOT", Path(__file__).resolve().parents[1]))
    db_path = root / "data" / "completions.db"
    provider = embeddings_provider_by_name(name=name)
    print(f"Embedder: {provider.name}")
    return EmbeddingsDataset(provider, db_path, source=source, model=model)


def mean_features(emb: np.ndarray) -> np.ndarray:
    return emb.mean(axis=0)


def collect_features(
    dataset,
    feature_fn,
    cache_name: str,
    tmp_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if tmp_dir is None:
        root = Path(os.environ.get("DETECTOR_ROOT", Path(__file__).resolve().parents[1]))
        tmp_dir = root / "tmp"

    rows_path   = tmp_dir / f"{cache_name}_rows.npy"
    labels_path = tmp_dir / f"{cache_name}_labels.npy"

    if rows_path.exists() and labels_path.exists():
        print(f"Loading cached features from {rows_path}")
        X, y = np.load(rows_path), np.load(labels_path)
        print(f"Loaded {len(X)} examples from disk")
        return X, y

    print(f"Computing features for {len(dataset)} texts (cache: {cache_name})...")
    rows, labels = [], []
    for i in tqdm(range(len(dataset))):
        emb, label = dataset[i]
        feat = feature_fn(emb)
        if feat is None:
            continue
        rows.append(feat)
        labels.append(label)

    X, y = np.array(rows), np.array(labels)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(rows_path, X)
    np.save(labels_path, y)
    print(f"Saved to {cache_name}_*.npy")

    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def train_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc":  round(float(roc_auc_score(y_test, y_prob)), 3),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "f1":       round(float(f1_score(y_test, y_pred)), 3),
    }
    print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    return metrics


def cross_validate(X: np.ndarray, y: np.ndarray):
    cv = _cv(LogisticRegression(max_iter=1000), X, y, cv=5,
             scoring=["roc_auc", "accuracy", "f1"])
    print(f"ROC-AUC:  {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
    print(f"Accuracy: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
    print(f"F1-Score: {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
