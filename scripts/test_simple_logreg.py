import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_regression import make_dataset, collect_features, train_eval

feature_fn = lambda emb: emb.mean(axis=0)

def run_experiment(train, test, model):
    suffix = f"_{model}" if model else ""
    ds_train = make_dataset(source=train, model=model)
    ds_test  = make_dataset(source=test, model=model)
    X_train, y_train = collect_features(ds_train, feature_fn, f"{train}_mean{suffix}")
    X_test,  y_test  = collect_features(ds_test,  feature_fn, f"{test}_mean{suffix}")
    train_eval(X_train, y_train, X_test, y_test)

experiments = [
    ("reddit", "wiki",   None), #1: train=reddit, test=wiki, completions=all
    ("wiki",   "reddit", None), #2: train=wiki, test=reddit, completions=all
    ("reddit", "wiki",   "gpt3"), #3: train=reddit, test=wiki, completions=gpt3
    ("wiki",   "reddit", "gpt3"), #4: train=wiki, test=reddit, completions=gpt3
    ("reddit", "wiki",   "gpt-5.4-mini"), #4: train=reddit, test=wiki, completions=gpt-5.4-mini
    ("wiki",   "reddit", "gpt-5.4-mini"), #5: train=wiki, test=reddit, completions=gpt-5.4-mini
    ("reddit", "wiki",   "gemini-3.1-pro"), #6: train=reddit, test=wiki, completions=gemini-3.1-pro
    ("wiki",   "reddit", "gemini-3.1-pro"), #7: train=wiki, test=reddit, completions=gemini-3.1-pro
]

for train, test, model in experiments:
    run_experiment(train, test, model)
