## Basic linear regression detector

```
LogisticRegression(max_iter=1000)

X = np.array([dataset[i][0].mean(axis=0) for i in range(len(dataset))])
y = np.array([dataset[i][1] for i in range(len(dataset))])
```

### N_texts = 8922

ROC-AUC: 0.998 ± 0.002
Accuracy: 0.982 ± 0.010
F1-Score: 0.982 ± 0.010

### N_texts = 200

ROC-AUC:  1.000 ± 0.000
Accuracy: 0.990 ± 0.012
F1-Score: 0.990 ± 0.013


## 6 magnitude features, N_texts = 8922

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

ROC-AUC:  0.383 ± 0.108
Accuracy: 0.349 ± 0.105
F1-Score: 0.326 ± 0.172

### N_texts = 200

ROC-AUC:  0.815 ± 0.027
Accuracy: 0.820 ± 0.033
F1-Score: 0.779 ± 0.049