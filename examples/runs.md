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
|     ROC-AUC | 0.328 | 0.901 |        0.471 |          0.196 |
|    Accuracy | 0.842 | 0.810 |        0.456 |          0.377 |
|   Precision | 0.842 | 0.930 |        0.458 |          0.429 |
|      Recall | 1.000 | 0.680 |        0.482 |          0.746 |
|    F1-Score | 0.914 | 0.785 |        0.470 |          0.545 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.525 | 0.846 |        0.418 |          0.684 |
|    Accuracy | 0.750 | 0.784 |        0.406 |          0.610 |
|   Precision | 0.750 | 0.713 |        0.379 |          0.650 |
|      Recall | 1.000 | 0.953 |        0.293 |          0.478 |
|    F1-Score | 0.857 | 0.815 |        0.330 |          0.551 |

### cross-validation on reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.483 | 0.843 |        0.428 |          0.693 |
|    Accuracy | 0.750 | 0.785 |        0.472 |          0.654 |
|    F1-Score | 0.857 | 0.796 |        0.512 |          0.659 |

### cross-validation on wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.671 | 0.901 |        0.509 |                |
|    Accuracy | 0.842 | 0.836 |        0.577 |                |
|    F1-Score | 0.914 | 0.837 |        0.464 |                |

## Magnitude + LogReg cross-domain detection

### train=reddit, test=wiki

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.208 | 0.134 |        0.015 |          0.505 |
|    Accuracy | 0.602 | 0.251 |        0.049 |          0.500 |
|   Precision | 0.793 | 0.339 |        0.086 |          0.500 |
|      Recall | 0.714 | 0.486 |        0.094 |          0.996 |
|    F1-Score | 0.751 | 0.400 |        0.090 |          0.666 |

### train=wiki, test=reddit

| Completions |   all |  gpt3 | gpt-5.4-mini | gemini-3.1-pro |
|-------------|-------|-------|--------------|----------------|
|     ROC-AUC | 0.202 | 0.277 |        0.214 |          0.733 |
|    Accuracy | 0.750 | 0.216 |        0.228 |          0.618 |
|   Precision | 0.750 | 0.092 |        0.003 |          0.653 |
|      Recall | 1.000 | 0.064 |        0.002 |          0.505 |
|    F1-Score | 0.857 | 0.076 |        0.002 |          0.570 |

