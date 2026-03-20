#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../../config.sh"
cd "$SCRIPT_DIR/.."

export DETECTOR_BASE_URL="${DETECTOR_BASE_URL:-$SERVER_URL}"

.venv/bin/pytest tests/test_analyze.py -v -m http "$@"
