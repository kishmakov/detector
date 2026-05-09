import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_paper_data.IntrinsicDim import PHD

MIN_TOKENS = 55  # range(50, n_tokens, 4) needs ≥2 points for slope
_phd = PHD()


def phd_features(emb: np.ndarray) -> np.ndarray | None:
    """Returns [phd_dim], extending short token paths by cycling shifts."""
    if emb.shape[0] < MIN_TOKENS:
        assert emb.shape[0] >= 2, f"Input has only {len(emb)} points"

        shifts = emb[1:] - emb[:-1]
        extended = np.empty((MIN_TOKENS, emb.shape[1]), dtype=emb.dtype)
        extended[:emb.shape[0]] = emb
        for i in range(emb.shape[0], MIN_TOKENS):
            extended[i] = extended[i - 1] + shifts[(i - emb.shape[0]) % len(shifts)]
        emb = extended

    return np.array([_phd.fit_transform(emb, max_points=emb.shape[0], point_jump=4)])
