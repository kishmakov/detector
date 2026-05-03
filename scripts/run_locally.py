#!/usr/bin/env python3
"""
Run all 8 train/test splits locally (sequential).
Usage: uv run scripts/run_locally.py [args...]
Example: uv run scripts/run_locally.py --method phd
"""
import os
import sys
import subprocess
from pathlib import Path

EXTRA    = sys.argv[1:]
ROOT     = Path(__file__).resolve().parents[1]
SCRIPT   = ROOT / "scripts" / "run_common.py"
TMP_BASE = ROOT / "tmp"

env = {**os.environ, "DETECTOR_ROOT": str(ROOT)}

for exp_idx in range(8):
    tmp_dir = TMP_BASE / f"exp_{exp_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--exp", str(exp_idx),
         "--tmp-dir", str(tmp_dir),
         *EXTRA],
        env=env,
        check=True,
    )
