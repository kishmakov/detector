#!/usr/bin/env python3
"""
Run train/test splits locally (sequential).
Usage: uv run scripts/run_locally.py --method METHOD [--exp N[,N...]]
Example: uv run scripts/run_locally.py --method phd
"""
import argparse
from datetime import datetime
import os
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
TMP_BASE = ROOT / "tmp"

os.environ["DETECTOR_ROOT"] = str(ROOT)

from run_common import add_experiment_args, run_task


ap = argparse.ArgumentParser()
add_experiment_args(ap)
args = ap.parse_args()

TMP_BASE.mkdir(parents=True, exist_ok=True)

run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

for exp_idx in args.exp:
    print(f"Running experiment {exp_idx}")
    run_task(exp_idx, TMP_BASE, args.method, run_id, str(exp_idx))
