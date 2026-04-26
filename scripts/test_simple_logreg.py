import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_regression import make_dataset, collect_features, train_eval

# Experiment #1: train=reddit, test=wiki

dataset_train = make_dataset(source="reddit")
dataset_test = make_dataset(source="wiki")

X_train, y_train = collect_features(dataset_train, lambda emb: emb.mean(axis=0), "reddit_mean")
X_test, y_test = collect_features(dataset_test, lambda emb: emb.mean(axis=0), "wiki_mean")

train_eval(X_train, y_train, X_test, y_test)