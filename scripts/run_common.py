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

METHODS = {
    "phd":       phd_features,
    "logreg":    mean_features,
    "magnitude": magnitude_features,
}


def experiment_tag(exp_idx: int, method: str) -> str:
    train_src, test_src, model = EXPERIMENTS[exp_idx]
    model_tag = re.sub(r"[^a-zA-Z0-9]", "", model) if model else "all"
    return f"{train_src}_{test_src}_{model_tag}_{method}"


def task_paths(tmp_dir: Path | str, run_id: str, jobn: str, exp_idx: int, method: str) -> tuple[Path, Path]:
    temp = Path(tmp_dir)
    exp_tag = experiment_tag(exp_idx, method)
    task_id = f"{run_id}_{jobn}"
    return (
        temp / f"output_{task_id}.txt",
        temp / f"results_{task_id}_{exp_tag}.txt",
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
    tmp_dir: Path | str,
    method: str,
    results_path: Path | str | None = None,
) -> dict:
    temp = Path(tmp_dir)
    temp.mkdir(parents=True, exist_ok=True)
    feature_fn = METHODS[method]

    train_src, test_src, model = EXPERIMENTS[exp_idx]
    model_tag = re.sub(r"[^a-zA-Z0-9]", "", model) if model else "all"
    exp_tag = experiment_tag(exp_idx, method)
    print(f"exp={exp_idx}  train={train_src}  test={test_src}  model={model_tag}  method={method}")

    Xtr, ytr = collect_features(make_dataset(source=train_src, model=model), feature_fn, f"{exp_tag}_train", temp)
    Xte, yte = collect_features(make_dataset(source=test_src,  model=model), feature_fn, f"{exp_tag}_test",  temp)
    metrics = train_eval(Xtr, ytr, Xte, yte)

    out_path = Path(results_path) if results_path is not None else temp / f"results_{exp_tag}.txt"
    key_width = max(len(k) for k in metrics)
    out_path.write_text("\n".join(f"{k:<{key_width}} : {v}" for k, v in metrics.items()) + "\n")
    return metrics


def run_task(
    exp_idx: int,
    tmp_dir: Path | str,
    method: str,
    run_id: str,
    jobn: str,
    *,
    capture_output: bool = True,
) -> dict:
    temp = Path(tmp_dir)
    temp.mkdir(parents=True, exist_ok=True)
    output_path, results_path = task_paths(temp, run_id, jobn, exp_idx, method)

    def _run() -> dict:
        print(f"Output: {output_path}")
        print(f"Metrics: {results_path}")
        return run_experiment(exp_idx, temp, method, results_path=results_path)

    if not capture_output:
        return _run()

    with capture_process_output(output_path):
        return _run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, choices=range(len(EXPERIMENTS)), required=True)
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--run-id", default=os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"))
    ap.add_argument("--jobn", default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    ap.add_argument("--no-capture-output", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id or f"local_{os.getpid()}"
    jobn = args.jobn or str(args.exp)
    run_task(
        args.exp,
        args.tmp_dir,
        args.method,
        run_id,
        jobn,
        capture_output=not args.no_capture_output,
    )


if __name__ == "__main__":
    main()
