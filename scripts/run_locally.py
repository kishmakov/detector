#!/usr/bin/env python3
"""
Run train/test splits locally (sequential).
Usage: uv run scripts/run_locally.py --method METHOD [--exp N ...]
Example: uv run scripts/run_locally.py --method phd
"""
import argparse
from datetime import datetime
import os
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
TMP_BASE = ROOT / "tmp"

os.environ["DETECTOR_ROOT"] = str(ROOT)

from run_common import EXPERIMENTS, METHODS, run_task, task_paths


ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=int, choices=range(len(EXPERIMENTS)), action="append",
                help="Run only experiment N. May be passed multiple times.")
ap.add_argument("--method", choices=METHODS, required=True)
args = ap.parse_args()

TMP_BASE.mkdir(parents=True, exist_ok=True)

exp_indices = args.exp if args.exp is not None else range(len(EXPERIMENTS))
run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

for exp_idx in exp_indices:
    output_path, metrics_path = task_paths(TMP_BASE, run_id, str(exp_idx), exp_idx, args.method)
    print(f"Running experiment {exp_idx}")
    print(f"  Output: {output_path}")
    print(f"  Metric: {metrics_path}")
    run_task(exp_idx, TMP_BASE, args.method, run_id, str(exp_idx))
