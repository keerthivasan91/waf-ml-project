"""app/services/runtime_preprocessor.py
Shared preprocessing that EXACTLY matches training pipeline.
"""
import numpy as np
from app.services.feature_extractor import extract_features, to_vector
from app.core.logging import logger
from ml.feature_engineering.tokenizer import CharTokenizer  


def prepare_inputs(request: dict, scaler):
    tok = CharTokenizer(max_len=512)
    """
    Reproduce training pipeline exactly.

    Returns
    -------
    fvec_scaled : np.ndarray shape (1, n_features), dtype float32
    token_ids   : np.ndarray shape (1, max_len), dtype int64
    raw_features: dict
    """
    # 1) feature dict
    features = extract_features(request)

    # 2) vector in exact training order
    fvec = to_vector(features).astype(np.float32)

    # ensure batch dimension
    if fvec.ndim == 1:
        fvec = fvec.reshape(1, -1)

    # 3) scaler exactly like training
    fvec_scaled = scaler.transform(fvec).astype(np.float32)

    # 4) token ids exactly like training
    token_ids = tok.encode_request(request).reshape(1, -1).astype(np.int64)

    logger.debug("Prepared inputs | fvec=%s | token_ids=%s", fvec_scaled.shape, token_ids.shape)

    return fvec_scaled, token_ids, features