"""app/services/layer2a_anomaly.py — ONNX anomaly detection (Layer 2A)"""
import numpy as np
import onnxruntime as ort
import joblib
from app.core.config import settings
from app.core.logging import logger

_sess = None
_scaler = None
_threshold = None
_in_name = "features"


def load() -> None:
    global _sess, _scaler, _threshold, _in_name

    onnx_path   = settings.L2A_ONNX_PATH
    scaler_path = settings.SCALER_PATH
    thr_path    = settings.L2A_THRESHOLD_PATH

    if not onnx_path.exists():
        raise FileNotFoundError(f"L2A ONNX not found: {onnx_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"L2A scaler not found: {scaler_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"L2A threshold not found: {thr_path}")

    _sess = ort.InferenceSession(str(onnx_path))
    _in_name = _sess.get_inputs()[0].name

    _scaler = joblib.load(scaler_path)

    with open(thr_path, "r", encoding="utf-8") as f:
        _threshold = float(f.read().strip())

    logger.info("L2A loaded | input=%s | threshold=%.5f", _in_name, _threshold)


def get_scaler():
    if _scaler is None:
        raise RuntimeError("L2A scaler not loaded")
    return _scaler


def infer(fvec_scaled: np.ndarray) -> tuple[bool, float]:
    """
    Parameters
    ----------
    fvec_scaled : (1, n_features) float32
        Already scaled feature vector.

    Returns
    -------
    (is_anomaly: bool, score: float)
    """
    if _threshold is None:
        raise RuntimeError("L2A threshold not loaded")
    recon = _sess.run(None, {_in_name: fvec_scaled})[0]
    score = float(np.mean((fvec_scaled - recon) ** 2))
    return score >= _threshold, score