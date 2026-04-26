import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import cross_validate
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embeddings_dataset import EmbeddingsDataset
from src.embeddings_provider import embeddings_provider_by_name


DB_PATH = ROOT / "data" / "completions.db"
TEMP_DIR = ROOT / "tmp"


def make_dataset(*, name: str = "bert-base-uncased", source=None, model=None) -> EmbeddingsDataset:
    provider = embeddings_provider_by_name(name=name)
    print(f"Embedder: {provider.name}")
    return EmbeddingsDataset(provider, DB_PATH, source=source, model=model)


def collect_features(dataset, feature_fn, cache_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Produce (X, y) tuples for logistic regression to train on.
    Args:
        dataset: The dataset to collect features from.
        feature_fn: A function that takes an embedding and returns a feature vector.
        cache_name: A name for the cache files.
    """
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
    for i in tqdm(range(n_texts)):
        emb, label = dataset[i]
        rows.append(feature_fn(emb))
        labels.append(label)

    X, y = np.array(rows), np.array(labels)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(rows_path, X)
    np.save(labels_path, y)
    print(f"Saved data to {cache_name}_*.npy")
    return X, y


def train_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")


def cross_validate(X: np.ndarray, y: np.ndarray):
    print("Running cross_validate for logistic regression...")
    cv = cross_validate(LogisticRegression(max_iter=1000), X, y, cv=5,
                        scoring=["roc_auc", "accuracy", "f1"])
    print(f"ROC-AUC:  {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
    print(f"Accuracy: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
    print(f"F1-Score: {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
