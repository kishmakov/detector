import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
