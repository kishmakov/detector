## Basic linear regression detector

```
LogisticRegression(max_iter=1000)

X = np.array([dataset[i][0].mean(axis=0) for i in range(len(dataset))])
y = np.array([dataset[i][1] for i in range(len(dataset))])
```

ROC-AUC: 0.998 ± 0.002
Accuracy: 0.982 ± 0.010
F1-Score: 0.982 ± 0.010
