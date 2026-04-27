# Various linear regression detector

## mean embedding + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.559 | 0.706 |        0.336 |          0.957 |
|    Accuracy | 0.702 | 0.571 |        0.377 |          0.598 |
|    F1-Score | 0.116 | 0.377 |        0.371 |          0.332 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.497 | 0.734 |        0.289 |          0.935 |
|    Accuracy | 0.748 | 0.571 |        0.398 |          0.874 |
|    F1-Score | 0.039 | 0.275 |        0.291 |          0.872 |

## PHD + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | ????? | 0.909 |        ????? |          ????? |
|    Accuracy | ????? | 0.844 |        ????? |          ????? |
|    F1-Score | ????? | 0.866 |        ????? |          ????? |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | ????? | 0.885 |        ????? |          ????? |
|    Accuracy | ????? | 0.836 |        ????? |          ????? |
|    F1-Score | ????? | 0.774 |        ????? |          ????? |



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