#!/usr/bin/env python3
"""
Submit all 8 train/test splits as a SLURM array job on Euler.
Usage: uv run scripts/run_on_euler.py [args...]
Example: uv run scripts/run_on_euler.py --method phd
"""
import os
import sys
import subprocess
from pathlib import Path

EULER   = "EULER"
SCRATCH = "/cluster/scratch/kshmakov/detector"
ROOT    = Path(__file__).resolve().parents[1]
SCRIPT  = ROOT / "scripts" / "run_common.py"
EXTRA   = sys.argv[1:]


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


# ── Infrastructure ─────────────────────────────────────────────────────────────

if ssh(f"test -d {SCRATCH}/.venv && echo yes || echo no") != "yes":
    print("Creating venv on Euler (takes a few minutes)...")
    ssh(
        f"module load stack/2024-06 gcc/12.2.0 python/3.12.8 && "
        f"python3 -m venv {SCRATCH}/.venv && "
        f"{SCRATCH}/.venv/bin/pip install -q scipy scikit-learn tqdm transformers && "
        f"{SCRATCH}/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu"
    )

print("Syncing files...")
ssh(f"mkdir -p {SCRATCH}/{{src,main_paper_data,data,scripts,tmp}}")
rsync(ROOT / "src/",                            f"{SCRATCH}/src/")
rsync(ROOT / "main_paper_data/IntrinsicDim.py", f"{SCRATCH}/main_paper_data/")
rsync(ROOT / "data/completions.db",             f"{SCRATCH}/data/")


# ── Experiment script ──────────────────────────────────────────────────────────

remote_script = f"{SCRATCH}/scripts/{SCRIPT.name}"
print(f"Uploading {SCRIPT} ...")
write_remote(remote_script, SCRIPT.read_text())


# ── SLURM array script ─────────────────────────────────────────────────────────

extra_str = " ".join(EXTRA)
sbatch = f"""\
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

{SCRATCH}/.venv/bin/python {remote_script} \\
    --exp $SLURM_ARRAY_TASK_ID \\
    --tmp-dir {SCRATCH}/tmp/exp_$SLURM_ARRAY_TASK_ID \\
    {extra_str} 2>&1
"""

write_remote(f"{SCRATCH}/run_array.sh", sbatch)


# ── Submit ─────────────────────────────────────────────────────────────────────

job_id = ssh(f"sbatch {SCRATCH}/run_array.sh").split()[-1]
print(f"Submitted array job {job_id} (tasks 0–7)")
print(f"Results: {SCRATCH}/results_N.txt  (N = 0–7)")
print()
print(f'Track:   SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "squeue -u kshmakov"')
print(f'Fetch:   SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "cat {SCRATCH}/results_N.txt"')
