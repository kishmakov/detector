import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logistic_regression_test import make_dataset, collect_features, run
from src.magnitude import MagnitudeEstimator

dataset = make_dataset()
est = MagnitudeEstimator()
X, y = collect_features(dataset, est.magnitude_features, "magnitude")
run(X, y)
