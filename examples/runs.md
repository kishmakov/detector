# Various linear regression detector

## mean embedding + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.566 | 0.706 |        0.349 |          0.957 |
|    Accuracy | 0.710 | 0.571 |        0.389 |          0.598 |
|    F1-Score | 0.107 | 0.377 |        0.386 |          0.332 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.498 | 0.734 |        0.287 |          0.935 |
|    Accuracy | 0.748 | 0.571 |        0.395 |          0.874 |
|    F1-Score | 0.039 | 0.275 |        0.290 |          0.872 |

## PHD + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.677 | 0.909 |        0.489 |          0.792 |
|    Accuracy | 0.810 | 0.844 |        0.221 |          0.573 |
|    F1-Score | 0.000 | 0.866 |        0.000 |          0.301 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.544 | 0.885 |        0.558 |          0.678 |
|    Accuracy | 0.821 | 0.836 |        0.397 |          0.577 |
|    F1-Score | 0.013 | 0.774 |        0.568 |          0.582 |



<!--
## 6 magnitude features

```
0  slope_fine    — log-log slope over the lowest 1/3 of the scale range
1  slope_medium  — log-log slope over the middle 1/3
2  slope_coarse  — log-log slope over the upper 1/3
3  slope_overall — log-log slope across the full range (= fit_transform)
4  curvature     — 2nd-order coefficient of a quadratic log-log fit
                    (positive ⟹ concave up, i.e. growing faster at large t)
5  log_mag_mid   — log(|tA|) at the median scale
```

### N_texts = 8922

- ROC-AUC:  0.383 ± 0.108
- Accuracy: 0.349 ± 0.105
- F1-Score: 0.326 ± 0.172

### N_texts = 200

- ROC-AUC:  0.815 ± 0.027
- Accuracy: 0.820 ± 0.033
- F1-Score: 0.779 ± 0.049 -->