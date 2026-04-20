import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings_dataset import EmbeddingsDataset
from src.model import model_iterator

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"
DATA_DIR_OUT = Path(__file__).parent.parent / "data"


def make_dataset() -> EmbeddingsDataset:
    model = next(iter(model_iterator()))
    print(f"Model: {model.name}")
    return EmbeddingsDataset(model, DATA_DIR)


def collect_features(dataset, feature_fn, cache_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows_path = DATA_DIR_OUT / f"{cache_name}_rows.npy"
    labels_path = DATA_DIR_OUT / f"{cache_name}_labels.npy"

    if rows_path.exists() and labels_path.exists():
        print(f"Loading cached features from {rows_path}")
        X, y = np.load(rows_path), np.load(labels_path)
        print(f"Loaded {len(X)} examples from disk")
        return X, y

    n_texts = int(sys.argv[1]) if len(sys.argv) > 1 else len(dataset)
    n_texts = min(n_texts, len(dataset))
    print(f"N_texts: {n_texts}")

    rows, labels = [], []
    for i in tqdm(range(n_texts)):
        emb, label = dataset[i]
        rows.append(feature_fn(emb))
        labels.append(label)

    X, y = np.array(rows), np.array(labels)
    DATA_DIR_OUT.mkdir(parents=True, exist_ok=True)
    np.save(rows_path, X)
    np.save(labels_path, y)
    print(f"Saved data to {cache_name}_*.npy")
    return X, y


def run(X: np.ndarray, y: np.ndarray):
    print("Running cross_validate for logistic regression...")
    cv = cross_validate(LogisticRegression(max_iter=1000), X, y, cv=5,
                        scoring=["roc_auc", "accuracy", "f1"])
    print(f"ROC-AUC:  {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
    print(f"Accuracy: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
    print(f"F1-Score: {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
