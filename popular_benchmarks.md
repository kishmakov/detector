# Popular Benchmarks

## MGTBench

### Novelty

First attempt to compare many detectors fairly in a standardized way.
- three datasets: Essay, WP, and Reuters.
- added test of robustness against adversarial attacks:
    - paraphrasing
    - random spacing
    - adversarial perturbation

### Findings

- LM Detector consistently excels across various datasets
- metric-based methods exhibit better adaptability to different LLMs
- but for attribution model-based methods significantly outperform
  their metric-based counterparts


## RAID

### Novelty

Basically comprehensiveness.
- over 6 million generations
- from 11 generators
- in 8 domains, 4 decoding strategies
- with 11 adversarial attacks

### Findings

- Binoculars performed impressively well across models even
  at extremely low false positive rates
- Originality achieved high precision in some constrained scenarios
- GPTZero was unusually robust to adversarial attacks.

