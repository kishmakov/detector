#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

export DETECTOR_BASE_URL="${DETECTOR_BASE_URL:-http://35.209.211.6:8000}"

.venv/bin/pytest tests/test_analyze.py -v -m http "$@"
