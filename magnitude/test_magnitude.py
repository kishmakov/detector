import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.magnitude import MagnitudeEstimator
from src.embeddings_dataset import EmbeddingsDataset
from src.model import model_iterator

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"

model = next(iter(model_iterator()))
print(f"Model: {model.name}")

dataset = EmbeddingsDataset(model, DATA_DIR)
est = MagnitudeEstimator()

print(f"Number of samples: {len(dataset)}")

rows, labels = [], []
for i in range(len(dataset)):
    emb, label = dataset[i]
    features = est.magnitude_features(emb)
    assert features is not None, "Magnitude features should not be empty"
    rows.append(features)
    labels.append(label)

print("Dataset prepared. Running logistic regression...")

clf = LogisticRegression(max_iter=1000)

scoring = ["roc_auc", "accuracy", "f1"]

X, y = np.array(rows), np.array(labels)
cv = cross_validate(clf, X, y, cv=5, scoring=scoring)

print(f"ROC-AUC:  {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
print(f"Accuracy: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
print(f"F1-Score: {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
