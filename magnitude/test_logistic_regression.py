import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings_dataset import EmbeddingsDataset
from src.model import model_iterator

DATA_DIR = Path(__file__).parent.parent / "main_paper_data" / "data"

model = next(iter(model_iterator()))
print(f"Model: {model.name}")

dataset = EmbeddingsDataset(model, DATA_DIR)

X = np.array([dataset[i][0].mean(axis=0) for i in range(len(dataset))])
y = np.array([dataset[i][1] for i in range(len(dataset))])

clf = LogisticRegression(max_iter=1000)

scoring = ["roc_auc", "accuracy", "f1"]
results = cross_validate(clf, X, y, cv=5, scoring=scoring)

print(f"ROC-AUC: {results['test_roc_auc'].mean():.3f} ± {results['test_roc_auc'].std():.3f}")
print(f"Accuracy: {results['test_accuracy'].mean():.3f} ± {results['test_accuracy'].std():.3f}")
print(f"F1-Score: {results['test_f1'].mean():.3f} ± {results['test_f1'].std():.3f}")