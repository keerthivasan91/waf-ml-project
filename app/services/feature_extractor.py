"""app/services/feature_extractor.py
Thin wrapper that imports the shared extractor and tokenizer from ml/.
At runtime the ml/ package is mounted as a volume or the extractor code
is copied into the container at /app/ml/.
"""
import sys, os
ML_PATH = os.environ.get("ML_PATH", "/app/ml")
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

from ml.feature_engineering.extractor import extract_features, to_vector  # noqa: E402
from ml.feature_engineering.tokenizer import CharTokenizer                 # noqa: E402

import numpy as np

_tokenizer = CharTokenizer(max_len=512)


def extract(request: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    feature_vector : (1, 25) float32
    token_ids      : (1, 512) int64
    """
    fvec      = to_vector(extract_features(request))           # (1, 25) float32
    token_ids = _tokenizer.encode_request(request).reshape(1, -1).astype("int64")
    return fvec, token_ids