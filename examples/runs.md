# Various linear regression detector

Human texts labeled as 0, generated as 1

## mean embedding + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.566 | 0.706 |        0.349 |          0.957 |
|    Accuracy | 0.710 | 0.571 |        0.389 |          0.598 |
|   Precision | 0.831 | 0.552 |        0.390 |          0.554 |
|      Recall | 0.823 | 0.861 |        0.394 |          0.996 |
|    F1-Score | 0.827 | 0.673 |        0.392 |          0.712 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.498 | 0.734 |        0.287 |          0.935 |
|    Accuracy | 0.748 | 0.571 |        0.395 |          0.874 |
|   Precision | 0.752 | 0.539 |        0.419 |          0.863 |
|      Recall | 0.990 | 0.980 |        0.543 |          0.890 |
|    F1-Score | 0.855 | 0.696 |        0.473 |          0.876 |

### cross-validation on reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.999 | 1.000 |        0.999 |          0.999 |
|    Accuracy | 0.990 | 0.995 |        0.991 |          0.992 |
|    F1-Score | 0.993 | 0.995 |        0.991 |          0.992 |

### cross-validation on wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.995 | 0.998 |        0.999 |          1.000 |
|    Accuracy | 0.992 | 0.991 |        0.993 |          0.996 |
|    F1-Score | 0.995 | 0.991 |        0.993 |          0.996 |


## PHD + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.    | 0.914 |        0.497 |          0.793 |
|    Accuracy | 0.    | 0.849 |        0.222 |          0.585 |
|   Precision |       | 0.867 |        0.222 |          0.550 |
|      Recall |       | 0.778 |        0.992 |          0.955 |
|    F1-Score | 0.    | 0.820 |        0.363 |          0.698 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.    | 0.    |        0.547 |          0.669 |
|    Accuracy | 0.    | 0.    |        0.397 |          0.570 |
|   Precision |       |       |        0.000 |          0.724 |
|      Recall |       |       |        0.000 |          0.464 |
|    F1-Score | 0.    | 0.    |        0.000 |          0.566 |

## Magnitude + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.    | 0.    |        0.    |          0.    |
|    Accuracy | 0.    | 0.    |        0.    |          0.    |
|   Precision |       |       |              |                |
|      Recall |       |       |              |                |
|    F1-Score | 0.    | 0.    |        0.    |          0.    |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.    | 0.    |        0.    |          0.    |
|    Accuracy | 0.    | 0.    |        0.    |          0.    |
|   Precision |       |       |              |                |
|      Recall |       |       |              |                |
|    F1-Score | 0.    | 0.    |        0.    |          0.    |


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