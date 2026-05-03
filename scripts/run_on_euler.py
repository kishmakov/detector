#!/usr/bin/env python3
"""
Submit train/test splits as SLURM jobs on Euler.
Usage: uv run scripts/run_on_euler.py [slurm args] [experiment args...]
Example: uv run scripts/run_on_euler.py --cpus 2 --hours 4 --method logreg --exp 2
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

EULER   = "EULER"
SCRATCH = "/cluster/scratch/kshmakov/detector"
ROOT    = Path(__file__).resolve().parents[1]
SCRIPT  = ROOT / "scripts" / "run_common.py"

os.environ["DETECTOR_ROOT"] = str(ROOT)
from run_common import EXPERIMENTS, METHODS, experiment_tag


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=int, choices=range(len(EXPERIMENTS)), metavar="N",
                help="Run only experiment N instead of submitting the full 0-7 array.")
ap.add_argument("--method", choices=METHODS, required=True)
ap.add_argument("--cpus", type=positive_int, default=4,
                help="CPU cores per task.")
ap.add_argument("--hours", type=positive_int, default=6,
                help="Wall-clock limit per task, in hours.")
ap.add_argument("--mem-per-cpu", default="1G",
                help="Memory per CPU core. Euler rejects node-level --mem requests.")
args, extra = ap.parse_known_args()
extra_str = shlex.join(extra)


def _env():
    return {**os.environ, "SSHPASS": os.environ["SSH_EULER_PASS"]}

def ssh(cmd: str) -> str:
    r = subprocess.run(
        ["sshpass", "-e", "ssh", EULER, cmd],
        capture_output=True, text=True, env=_env(),
    )
    if r.returncode != 0:
        if r.stdout:
            print(r.stdout, file=sys.stderr, end="")
        if r.stderr:
            print(r.stderr, file=sys.stderr, end="")
        r.check_returncode()
    return r.stdout.strip()

def write_remote(path: str, content: str):
    subprocess.run(
        ["sshpass", "-e", "ssh", EULER, f"cat > {path}"],
        input=content, text=True, env=_env(), check=True,
    )

def rsync(src: Path, dst: str):
    src_arg = f"{src}/" if src.is_dir() else str(src)
    subprocess.run(
        ["sshpass", "-e", "rsync", "-a", src_arg, f"{EULER}:{dst}"],
        env=_env(), check=True,
    )


# ── Infrastructure ─────────────────────────────────────────────────────────────

if ssh(f"test -d {SCRATCH}/.venv && echo yes || echo no") != "yes":
    print("Creating venv on Euler (takes a few minutes)...")
    ssh(
        f"module load stack/2024-06 gcc/12.2.0 python/3.12.8 && "
        f"python3 -m venv {SCRATCH}/.venv && "
        f"{SCRATCH}/.venv/bin/pip install -q scipy scikit-learn transformers && "
        f"{SCRATCH}/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu"
    )

if ssh(f"test -d {SCRATCH}/hf_cache/hub/models--bert-base-uncased && echo yes || echo no") != "yes":
    print("Caching bert-base-uncased on Euler...")
    ssh(
        f"module load stack/2024-06 gcc/12.2.0 python/3.12.8 && "
        f"HF_HOME={SCRATCH}/hf_cache {SCRATCH}/.venv/bin/python -c "
        f"\"from transformers import AutoTokenizer, AutoModel; "
        f"AutoTokenizer.from_pretrained('bert-base-uncased'); "
        f"AutoModel.from_pretrained('bert-base-uncased')\""
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


# ── SLURM script ───────────────────────────────────────────────────────────────

array_spec = str(args.exp) if args.exp is not None else "0-7"
task_description = experiment_tag(args.exp, args.method) if args.exp is not None else "tasks 0-7"

sbatch = f"""\
#!/bin/bash
#SBATCH --job-name=intrinsic_dim
#SBATCH --output={SCRATCH}/tmp/output_%A_%a.txt
#SBATCH --partition=bigmem.24h
#SBATCH --array={array_spec}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --time={args.hours}:00:00

module load stack/2024-06 gcc/12.2.0 python/3.12.8
export OMP_NUM_THREADS={args.cpus}
export DETECTOR_ROOT={SCRATCH}
export HF_HOME={SCRATCH}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

{SCRATCH}/.venv/bin/python {remote_script} \\
    --exp $SLURM_ARRAY_TASK_ID \\
    --tmp-dir {SCRATCH}/tmp \\
    --method {args.method} \\
    --run-id $SLURM_ARRAY_JOB_ID \\
    --jobn $SLURM_ARRAY_TASK_ID \\
    --no-capture-output \\
    {extra_str} 2>&1
"""

write_remote(f"{SCRATCH}/run_array.sh", sbatch)


# ── Submit ─────────────────────────────────────────────────────────────────────

job_id = ssh(f"sbatch {SCRATCH}/run_array.sh").split()[-1]
print(f"Submitted array job {job_id} ({task_description})")
print(f"Requested per task: {args.cpus} CPU(s), {args.mem_per_cpu} memory per CPU, {args.hours} hour(s)")
if args.exp is None:
    output_path = f"{SCRATCH}/tmp/output_{job_id}_N.txt"
    metrics_path = f"{SCRATCH}/tmp/results_{job_id}_N_<exp_tag>.txt"
else:
    output_path = f"{SCRATCH}/tmp/output_{job_id}_{args.exp}.txt"
    metrics_path = f"{SCRATCH}/tmp/results_{job_id}_{args.exp}_{task_description}.txt"
print("Files:")
print(f"  Output: {output_path}")
print(f"  Metric: {metrics_path}")

if args.exp is None:
    print("  N = 0-7")

print()
print(f'Track:   SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "squeue -u kshmakov"')
print(f'Output:  SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "cat {output_path}"')
print(f'Metrics: SSHPASS="$SSH_EULER_PASS" sshpass -e ssh EULER "cat {metrics_path}"')
