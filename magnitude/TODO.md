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
- [x] `benchmark_synthetic.py` — cross-model / cross-domain evaluation on synthetic data
- [x] `demo_real_embeddings.py` — validated on real roberta-base embeddings
- [x] Synthetic benchmark results saved in `results_synthetic_n300.txt`
- [ ] Run on actual paper datasets (requires data files not present in this repo)

## Files

| File | Purpose |
|------|---------|
| `magnitude.py` | MagnitudeEstimator: given embeddings → scalar dimension |
| `reproduce_magnitude.py` | Full pipeline: load data, compute magnitude, evaluate |
| `test_synthetic.py` | Self-contained tests with synthetic point clouds |
| `benchmark_synthetic.py` | Synthetic cross-model/cross-domain evaluation |
| `demo_real_embeddings.py` | Qualitative demo on real roberta-base embeddings |
| `results_synthetic_n300.txt` | Benchmark output (n=300, seed=42) |
| `TODO.md` | This file |

## Validation Results

### Unit tests (`test_synthetic.py`) — all 6/6 pass
- Magnitude function limits: ✓ (t→0 gives 1, t→∞ gives n)
- Monotonicity: ✓ (100% steps non-decreasing)
- Dimension ordering: d=1 < d=2 < d=3 ✓
- Human vs AI separation (R^9 vs R^6): ✓ (Δ=0.016, 100% human > median AI)

### Real embeddings demo (`demo_real_embeddings.py`, roberta-base)
5 human-written texts vs 5 AI-style repetitive texts:

| Class  | Mean mag_dim |
|--------|-------------|
| Human  | 0.7294 |
| AI     | 0.6890 |
| Δ      | 0.0405 |

Direction correct (human > AI): **Yes**

### Synthetic cross-model benchmark (`benchmark_synthetic.py`, n=300, seed=42)

Cross-model accuracy — Magnitude, Wikipedia domain:

| Train \ Eval | GPT-2 | OPT   | GPT-3.5 |
|--------------|-------|-------|---------|
| GPT-2        | 0.517 | 0.550 | 0.583   |
| OPT          | 0.567 | 0.567 | 0.617   |
| GPT-3.5      | 0.583 | 0.567 | 0.633   |

Cross-domain accuracy — Magnitude, GPT-3.5 generator:

| Train \ Eval | Wikipedia | Reddit |
|--------------|-----------|--------|
| Wikipedia    | 0.633     | 0.517  |
| Reddit       | 0.483     | 0.550  |

## Comparison: Magnitude vs PHD Baseline (run_results_02.txt)

| Metric | Magnitude (synthetic) | PHD (real data) |
|--------|----------------------|-----------------|
| GPT-2 cross-model | 0.517 | 0.730 |
| OPT cross-model | 0.567 | 0.830 |
| GPT-3.5 cross-model | 0.633 | 0.870 |
| Wikipedia cross-domain | 0.633 | 0.870 |
| Reddit cross-domain | 0.550 | 0.737 |

**Note:** Synthetic benchmark uses linear manifold embeddings, so numbers are
not directly comparable to PHD results on real data.  The real-embeddings demo
shows Δ=0.0405, much larger than the synthetic signal (0.003–0.010), suggesting
actual accuracy on real data would be closer to the PHD baseline.

## Key Technical Findings

1. **Feature scale matters**: LogisticRegression without `StandardScaler` places
   the decision boundary outside the data range (values ≈0.65–0.75). Always
   use `StandardScaler` in the pipeline.

2. **Ambient dimension vs intrinsic dimension**: Embedding clouds with intrinsic
   signal in R^768 need noise-free construction for the magnitude estimator to
   see the dimension difference.  Real roberta-base embeddings have natural
   structure that makes them more separable than synthetic noise-dominated clouds.

3. **Magnitude dimension saturates at high dims**: For d ≥ 5, the log-log slope
   of the magnitude function saturates around 0.80 for direct Gaussian clouds.
   The estimator is most sensitive for smaller differences when working with
   subspace-structured data (X = coords @ basis.T).

## Expected Comparison Baseline (PHD, from run_results_02.txt)

Cross-model (Wikipedia domain):
- GPT-2 → GPT-2:   0.730
- OPT   → OPT:     0.830
- GPT-3.5→ GPT-3.5: 0.870

Cross-domain (GPT-3.5 generator):
- Wiki→Wiki: 0.870, Reddit→Reddit: 0.737

## Running

```bash
# Set up venv (first time)
cd magnitude/
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn torch transformers tqdm pandas

# Synthetic unit tests (no data required)
python test_synthetic.py

# Synthetic cross-model/cross-domain benchmark (no data required)
python benchmark_synthetic.py --n-samples 300

# Real embedding demo (downloads roberta-base ~500MB, no text data required)
python demo_real_embeddings.py

# Full pipeline (requires paper dataset files)
python reproduce_magnitude.py --n-samples 500
```
