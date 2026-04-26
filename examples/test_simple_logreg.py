import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_regression import make_dataset, collect_features, run

dataset = make_dataset()
X, y = collect_features(dataset, lambda emb: emb.mean(axis=0), "simple_logreg")
run(X, y)
