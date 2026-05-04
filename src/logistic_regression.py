import os
import time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate as _cv

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


def compute_feature_matrix(dataset, feature_fn) -> np.ndarray:
    rows = []
    feature_shape = None
    total = len(dataset)
    started_at = time.perf_counter()
    for i in range(total):
        emb, _ = dataset[i]
        feat = feature_fn(emb)
        if feat is not None:
            feat = np.asarray(feat)
            feature_shape = feature_shape or feat.shape
        rows.append(feat)
        finished = i + 1
        if finished % 100 == 0 or finished == total:
            avg_time = (time.perf_counter() - started_at) / finished
            print(f"Finished {finished}/{total} steps; avg {avg_time:.3f}s/step")

    if feature_shape is None:
        feature_shape = (1,)
    rows = [
        row if row is not None else np.full(feature_shape, np.nan)
        for row in rows
    ]
    return np.array(rows)


def collect_features(
    dataset,
    feature_fn,
    cache_name: str,
    tmp_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    rows_path   = tmp_dir / f"{cache_name}_rows.npy"

    y = np.array(dataset.labels())
    X = None
    if rows_path.exists():
        print(f"Loading cached features from {rows_path}")
        X = np.load(rows_path)
        print(f"Loaded {len(X)} feature rows from disk")
        if len(X) != len(y):
            print(f"Ignoring cached features with {len(X)} rows; expected {len(y)}")
            X = None

    if X is None:
        print(f"Computing features for {len(dataset)} texts (cache: {cache_name})...")
        X = compute_feature_matrix(dataset, feature_fn)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(rows_path, X)
        print(f"Saved to {rows_path}")

    assert X is not None
    assert y is not None
    assert len(X) == len(y), f"Feature rows and labels differ: {len(X)} != {len(y)}"
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def train_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "ROC-AUC":  round(float(roc_auc_score(y_test, y_prob)), 3),
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "Precision": round(float(precision_score(y_test, y_pred)), 3),
        "Recall":    round(float(recall_score(y_test, y_pred)), 3),
        "F1-Score": round(float(f1_score(y_test, y_pred)), 3),
        "Confusion matrix": "\n" + str(confusion_matrix(y_test, y_pred, labels=[0, 1])),
        "Classification report": "\n" + classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["human", "machine"],
            zero_division=0,
        ),
    }
    return metrics


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    cv = _cv(LogisticRegression(max_iter=1000), X, y, cv=5,
             scoring=["roc_auc", "accuracy", "f1"])
    metrics = {
        "ROC-AUC":  f"{cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}",
        "Accuracy": f"{cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}",
        "F1-Score": f"{cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}",
    }
    return metrics
