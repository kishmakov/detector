import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.magnitude import MagnitudeEstimator
from src.embeddings_dataset import EmbeddingsDataset
from src.model import model_iterator

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
ROWS_PATH = OUTPUT_DIR / "rows.npy"
LABELS_PATH = OUTPUT_DIR / "labels.npy"

model = next(iter(model_iterator()))
print(f"Model: {model.name}")

dataset = EmbeddingsDataset(model, DATA_DIR)
est = MagnitudeEstimator()

if ROWS_PATH.exists() and LABELS_PATH.exists():
    print(f"Loading saved features from {ROWS_PATH} and {LABELS_PATH}")
    X = np.load(ROWS_PATH)
    y = np.load(LABELS_PATH)
    print(f"Loaded {len(X)} examples from disk")
else:
    n_texts = int(sys.argv[1]) if len(sys.argv) > 1 else len(dataset)
    n_texts = min(n_texts, len(dataset))
    print(f"N_texts: {n_texts}")


    rows, labels = [], []
    for i in tqdm(range(n_texts)):
        emb, label = dataset[i]
        features = est.magnitude_features(emb)
        assert features is not None, "Magnitude features should not be empty"
        rows.append(features)
        labels.append(label)

    X = np.array(rows)
    y = np.array(labels)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ROWS_PATH, X)
    np.save(LABELS_PATH, y)
    print(f"Saved features to {ROWS_PATH} and {LABELS_PATH}")

print("Dataset prepared. Running logistic regression...")

clf = LogisticRegression(max_iter=1000)

scoring = ["roc_auc", "accuracy", "f1"]
cv = cross_validate(clf, X, y, cv=5, scoring=scoring)

print(f"ROC-AUC:  {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
print(f"Accuracy: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
print(f"F1-Score: {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
