import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DETECTOR_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from src.logistic_regression import make_dataset, collect_features, train_eval, mean_features
from src.intrinsic_dim import phd_features
from src.magnitude import magnitude_features

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

DEFAULT_EXPERIMENTS = tuple(range(len(EXPERIMENTS)))
DEFAULT_EXPERIMENTS_TEXT = ",".join(str(exp_idx) for exp_idx in DEFAULT_EXPERIMENTS)

METHODS = {
    "phd":       phd_features,
    "logreg":    mean_features,
    "magnitude": magnitude_features,
}


def parse_experiment_indices(value: str) -> tuple[int, ...]:
    valid = set(range(len(EXPERIMENTS)))
    exp_indices = []
    for part in value.split(","):
        try:
            exp_idx = int(part.strip())
        except ValueError:
            continue

        if exp_idx in valid:
            exp_indices.append(exp_idx)

    if not exp_indices:
        raise argparse.ArgumentTypeError(
            f"must contain at least one experiment index from {DEFAULT_EXPERIMENTS_TEXT}"
        )

    return tuple(exp_indices)


def add_experiment_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--exp",
        type=parse_experiment_indices,
        default=DEFAULT_EXPERIMENTS,
        metavar="N[,N...]",
        help=f"Comma-separated experiment indices. Defaults to {DEFAULT_EXPERIMENTS_TEXT}.",
    )
    ap.add_argument("--method", choices=METHODS, required=True)


def experiment_tag(exp_idx: int, method: str) -> str:
    train_src, test_src, model = EXPERIMENTS[exp_idx]
    model_tag = re.sub(r"[^a-zA-Z0-9]", "", model) if model else "all"
    return f"{train_src}_{test_src}_{model_tag}_{method}"


def _task_paths(tmp_dir: Path, run_id: str, jobn: str, exp_idx: int, method: str) -> tuple[Path, Path]:
    exp_tag = experiment_tag(exp_idx, method)
    task_id = f"{run_id}_{jobn}"
    return (
        tmp_dir / f"output_{task_id}.txt",
        tmp_dir / f"results_{task_id}_{exp_tag}.txt",
    )


@contextmanager
def capture_process_output(path: Path):
    with path.open("w") as output:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            with redirect_stdout(output), redirect_stderr(output):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def run_experiment(
    exp_idx: int,
    tmp_dir: Path,
    method: str,
    results_path: Path,
) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    feature_fn = METHODS[method]

    train_src, test_src, model = EXPERIMENTS[exp_idx]
    model_tag = re.sub(r"[^a-zA-Z0-9]", "", model) if model else "all"
    exp_tag = experiment_tag(exp_idx, method)
    print(f"exp={exp_idx}  train={train_src}  test={test_src}  model={model_tag}  method={method}")

    Xtr, ytr = collect_features(make_dataset(source=train_src, model=model), feature_fn, f"{exp_tag}_train", tmp_dir)
    Xte, yte = collect_features(make_dataset(source=test_src,  model=model), feature_fn, f"{exp_tag}_test",  tmp_dir)
    metrics = train_eval(Xtr, ytr, Xte, yte)

    key_width = max(len(k) for k in metrics)
    results_path.write_text("\n".join(f"{k:<{key_width}} : {v}" for k, v in metrics.items()) + "\n")
    return metrics


def run_task(
    exp_idx: int,
    tmp_dir: Path,
    method: str,
    run_id: str,
    jobn: str,
    *,
    capture_output: bool = True,
) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path, results_path = _task_paths(tmp_dir, run_id, jobn, exp_idx, method)
    print(f"Output: {output_path}")
    print(f"Metrics: {results_path}")

    def _run() -> dict:
        return run_experiment(exp_idx, tmp_dir, method, results_path=results_path)

    if not capture_output:
        return _run()

    with capture_process_output(output_path):
        return _run()


def main():
    ap = argparse.ArgumentParser()
    add_experiment_args(ap)
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--jobn", required=True)
    ap.add_argument("--no-capture-output", action="store_true")
    args = ap.parse_args()

    assert args.exp
    assert args.run_id
    assert args.jobn
    for exp_idx in args.exp:
        jobn = args.jobn if len(args.exp) == 1 else f"{args.jobn}_{exp_idx}"
        run_task(
            exp_idx,
            Path(args.tmp_dir),
            args.method,
            args.run_id,
            jobn,
            capture_output=not args.no_capture_output,
        )


if __name__ == "__main__":
    main()
