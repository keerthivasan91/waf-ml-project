"""app/services/layer2a_anomaly.py — ONNX anomaly detection (Layer 2A)

Shallow Autoencoder, one-class anomaly detector. INPUT_DIM=29 (post
domain-stripping + JSON/GraphQL feature additions — see
ml/feature_engineering/extractor.py).

Two distinct thresholds are in play, and they must not be conflated:

- `_threshold` (this module): L2A's own trained/calibrated operating
  threshold — "maximise recall subject to FPR<=5% on validation only".
  Used for L2A's own reported recall/FPR and for health-monitor
  recalibration (adaptive retraining). NOT used to gate pipeline routing.

- `settings.ESCALATION_THRESHOLD` (app/core/config.py): the CRC
  Decision 2 "selective escalation" cutoff (P85 of normal validation L2A
  scores) that the WAF middleware uses to decide whether a request is
  escalated to Layer 2B at all. This lives in the middleware, not here.
"""
import numpy as np
import onnxruntime as ort
from app.core.config import settings
from app.core.logging import logger

_sess: ort.InferenceSession = None
_threshold: float = None
_in_name: str = "features"


def load() -> None:
    global _sess, _threshold, _in_name

    onnx_path = settings.L2A_ONNX_PATH
    thr_path = settings.L2A_THRESHOLD_PATH

    if not onnx_path.exists():
        raise FileNotFoundError(f"L2A ONNX not found: {onnx_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"L2A threshold not found: {thr_path}")

    _sess = ort.InferenceSession(str(onnx_path))
    _in_name = _sess.get_inputs()[0].name

    with open(thr_path, "r", encoding="utf-8") as f:
        _threshold = float(f.read().strip())

    logger.info(
        "L2A loaded | input=%s | own_threshold=%.5f | escalation_threshold=%.8f",
        _in_name, _threshold, settings.ESCALATION_THRESHOLD,
    )


def score(feature_vector: np.ndarray) -> float:
    """
    Parameters
    ----------
    feature_vector : (1, 29) float32
        Already scaled feature vector

    Returns
    -------
    Raw reconstruction-error score (mean squared error).
    """
    recon = _sess.run(None, {_in_name: feature_vector})[0]
    return float(np.mean((feature_vector - recon) ** 2))


def infer(feature_vector: np.ndarray) -> tuple[bool, float]:
    """
    Back-compat wrapper: (is_anomaly per L2A's OWN threshold, raw score).

    NOTE: pipeline routing (whether to escalate to L2B) must use
    `settings.ESCALATION_THRESHOLD` in the middleware, not this bool —
    they are different thresholds selected for different purposes
    (see module docstring / CRC Decision 2).
    """
    s = score(feature_vector)
    return s >= _threshold, s
