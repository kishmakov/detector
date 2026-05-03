#!/usr/bin/env python3
"""
Submit test_intrinsic_dim experiments as 8 parallel SLURM tasks on Euler.
Usage: uv run scripts/test_intrinsic_dim_euler.py
"""
import os
import subprocess
import textwrap
from pathlib import Path

EULER   = "EULER"
SCRATCH = "/cluster/scratch/kshmakov/detector"
ROOT    = Path(__file__).resolve().parents[1]


def _env():
    return {**os.environ, "SSHPASS": os.environ["SSH_EULER_PASS"]}

def ssh(cmd: str) -> str:
    r = subprocess.run(
        ["sshpass", "-e", "ssh", EULER, cmd],
        capture_output=True, text=True, env=_env(), check=True,
    )
    return r.stdout.strip()

def write_remote(path: str, content: str):
    subprocess.run(
        ["sshpass", "-e", "ssh", EULER, f"cat > {path}"],
        input=content, text=True, env=_env(), check=True,
    )

def rsync(src: Path, dst: str):
    subprocess.run(
        ["sshpass", "-e", "rsync", "-a", str(src), f"{EULER}:{dst}"],
        env=_env(), check=True,
    )


# ── 0. Venv (skipped if already present) ──────────────────────────────────────
if ssh(f"test -d {SCRATCH}/.venv && echo yes || echo no") != "yes":
    print("Creating venv on Euler (takes a few minutes)...")
    ssh(
        f"module load stack/2024-06 gcc/12.2.0 python/3.12.8 && "
        f"python3 -m venv {SCRATCH}/.venv && "
        f"{SCRATCH}/.venv/bin/pip install -q scipy scikit-learn tqdm transformers && "
        f"{SCRATCH}/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu"
    )

# ── 1. Sync ────────────────────────────────────────────────────────────────────
print("Syncing files...")
ssh(f"mkdir -p {SCRATCH}/{{src,main_paper_data,data,scripts,tmp}}")
rsync(ROOT / "src/",                            f"{SCRATCH}/src/")
rsync(ROOT / "main_paper_data/IntrinsicDim.py", f"{SCRATCH}/main_paper_data/")
rsync(ROOT / "data/completions.db",             f"{SCRATCH}/data/")


# ── 2. Per-experiment runner ───────────────────────────────────────────────────
RUNNER = textwrap.dedent("""\
    import os, sys, argparse
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, os.environ["DETECTOR_ROOT"])
    from src.logistic_regression import make_dataset, train_eval
    from main_paper_data.IntrinsicDim import PHD

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

    MIN_TOKENS = 91
    _phd = PHD()

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",     type=int, required=True)
    ap.add_argument("--tmp-dir", required=True)
    args = ap.parse_args()

    TEMP = Path(args.tmp_dir)
    TEMP.mkdir(parents=True, exist_ok=True)

    def collect(dataset, name):
        rp = TEMP / f"{name}_rows.npy"
        lp = TEMP / f"{name}_labels.npy"
        if rp.exists() and lp.exists():
            return np.load(rp), np.load(lp)
        rows, labels = [], []
        for i in tqdm(range(len(dataset))):
            emb, label = dataset[i]
            if emb.shape[0] < MIN_TOKENS:
                continue
            rows.append([_phd.fit_transform(emb, max_points=emb.shape[0])])
            labels.append(label)
        X, y = np.array(rows), np.array(labels)
        np.save(rp, X)
        np.save(lp, y)
        mask = ~np.isnan(X).any(axis=1)
        return X[mask], y[mask]

    train_src, test_src, model = EXPERIMENTS[args.exp]
    suffix = f"_{model}" if model else ""
    print(f"exp={args.exp}  train={train_src}  test={test_src}  model={model or 'all'}")
    Xtr, ytr = collect(make_dataset(source=train_src, model=model), f"{train_src}_phd{suffix}")
    Xte, yte = collect(make_dataset(source=test_src,  model=model), f"{test_src}_phd{suffix}")
    train_eval(Xtr, ytr, Xte, yte)
""")

write_remote(f"{SCRATCH}/scripts/run_exp.py", RUNNER)


# ── 3. SLURM array script ──────────────────────────────────────────────────────
SBATCH = textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --job-name=intrinsic_dim
    #SBATCH --output={SCRATCH}/results_%a.txt
    #SBATCH --partition=bigmem.24h
    #SBATCH --array=0-7
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem-per-cpu=4G
    #SBATCH --time=24:00:00

    module load stack/2024-06 gcc/12.2.0 python/3.12.8
    export OMP_NUM_THREADS=8
    export DETECTOR_ROOT={SCRATCH}

    {SCRATCH}/.venv/bin/python {SCRATCH}/scripts/run_exp.py \\
        --exp $SLURM_ARRAY_TASK_ID \\
        --tmp-dir {SCRATCH}/tmp/exp_$SLURM_ARRAY_TASK_ID 2>&1
""")

write_remote(f"{SCRATCH}/run_array.sh", SBATCH)


# ── 4. Submit ──────────────────────────────────────────────────────────────────
job_id = ssh(f"sbatch {SCRATCH}/run_array.sh").split()[-1]
print(f"Submitted array job {job_id} (tasks 0–7)")
print(f"Results will be written to {SCRATCH}/results_N.txt (N = 0–7)")
print()
print(f'Track:   SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "squeue -u kshmakov"')
print(f'Results: SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "cat {SCRATCH}/results_N.txt"')
