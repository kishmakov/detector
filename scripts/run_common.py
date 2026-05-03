import os, sys, argparse, re
from pathlib import Path

sys.path.insert(0, os.environ["DETECTOR_ROOT"])
from src.logistic_regression import make_dataset, collect_features, train_eval, mean_features
from src.intrinsic_dim import phd_features

EXPERIMENTS = [
    ("reddit", "wiki",   None),
    ("wiki",   "reddit", None),
    ("reddit", "wiki",   "gpt3"),
    ("wiki",   "reddit", "gpt3"),
    ("reddit", "wiki",   "gpt-5.4-mini"),
    ("wiki",   "reddit", "gpt-5.4-mini"),
    ("reddit", "wiki",   "gemini-3.1-pro"),
    ("wiki",   "reddit", "gemini-3.1-pro"),
]

METHODS = {
    "phd":    (phd_features,  "phd"),
    "logreg": (mean_features, "mean"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--exp",    type=int,         required=True)
ap.add_argument("--tmp-dir",                  required=True)
ap.add_argument("--method", choices=METHODS,  required=True)
args = ap.parse_args()

TEMP = Path(args.tmp_dir)
feature_fn, cache_tag = METHODS[args.method]

train_src, test_src, model = EXPERIMENTS[args.exp]
suffix = f"_{model}" if model else ""
model_tag = re.sub(r"[^a-zA-Z0-9]", "", model) if model else "all"
print(f"exp={args.exp}  train={train_src}  test={test_src}  model={model_tag}  method={args.method}")

Xtr, ytr = collect_features(make_dataset(source=train_src, model=model), feature_fn, f"{train_src}_{cache_tag}{suffix}", TEMP)
Xte, yte = collect_features(make_dataset(source=test_src,  model=model), feature_fn, f"{test_src}_{cache_tag}{suffix}",  TEMP)
metrics = train_eval(Xtr, ytr, Xte, yte)

out_path = TEMP / f"run_{train_src}_{test_src}_{model_tag}_{args.method}.txt"
out_path.write_text("\n".join(f"{k}: {v}" for k, v in metrics.items()) + "\n")
print(f"Saved metrics to {out_path}")
