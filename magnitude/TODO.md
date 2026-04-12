# Magnitude-Based AI Text Detection

Goal: apply the magnitude function (a size invariant of metric spaces) as an
alternative intrinsic-dimension feature for distinguishing human vs AI text,
mirroring the PHD approach from the main paper (arxiv 2306.04723).

## Idea

Each text → token embeddings (via roberta-base) → finite metric space → magnitude
function |tA| over a scale grid → log-log slope ≈ "magnitude dimension".
Human text embeddings are geometrically more complex → higher dimension.

## Status

- [x] TODO.md created
- [x] `magnitude.py` — core estimator (MagnitudeEstimator class)
- [x] `reproduce_magnitude.py` — full pipeline mirroring reproduce.py
- [x] `test_synthetic.py` — unit tests / demo on synthetic embeddings
- [ ] Run on real data (requires data files + torch + transformers)
- [ ] Compare accuracy tables with PHD results from run_results_02.txt

## Files

| File | Purpose |
|------|---------|
| `magnitude.py` | MagnitudeEstimator: given embeddings → scalar dimension |
| `reproduce_magnitude.py` | Full pipeline: load data, compute magnitude, evaluate |
| `test_synthetic.py` | Self-contained tests with synthetic point clouds |
| `TODO.md` | This file |

## Expected Comparison Baseline (PHD, from run_results_02.txt)

Cross-model (Wikipedia domain):
- GPT-2 → GPT-2:  0.730
- OPT  → OPT:     0.830
- GPT-3.5→GPT-3.5:0.870

Cross-domain (GPT-3.5 generator):
- Wiki→Wiki: 0.870, Reddit→Reddit: 0.737

## Running

```bash
# Set up venv (first time)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn torch transformers tqdm pandas

# Synthetic test (no data required)
python test_synthetic.py

# Full pipeline (requires data at path in reproduce.py)
python reproduce_magnitude.py --n-samples 500
```
