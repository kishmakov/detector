import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_paper_data.IntrinsicDim import PHD

MIN_TOKENS = 91  # range(50, n_tokens, 40) needs ≥2 points for slope
_phd = PHD()


def phd_features(emb: np.ndarray) -> np.ndarray | None:
    """Returns [phd_dim], or None if the text is too short to estimate."""
    if emb.shape[0] < MIN_TOKENS:
        return None
    return np.array([_phd.fit_transform(emb, max_points=emb.shape[0])])
