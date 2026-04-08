"""app/services/layer2b_deep.py — ONNX deep classifier (Layer 2B)"""
import numpy as np
import onnxruntime as ort
import scipy.special
from app.core.config import settings
from app.core.logging import logger

_sess = None
_in_name = None

# MUST match training exactly
CLASS_NAMES = [
    "normal",
    "sqli",
    "xss",
    "lfi",
    "cmdi",
    "other_attack",
]

# if your ONNX model expects token_ids, keep True
USES_TOKENS = True


def load() -> None:
    global _sess, _in_name

    onnx_path = settings.L2B_ONNX_PATH
    if not onnx_path.exists():
        raise FileNotFoundError(f"L2B ONNX not found: {onnx_path}")

    _sess = ort.InferenceSession(str(onnx_path))
    _in_name = _sess.get_inputs()[0].name

    logger.info("L2B loaded | input=%s | uses_tokens=%s", _in_name, USES_TOKENS)


def infer(fvec_scaled: np.ndarray, token_ids: np.ndarray):
    """
    Returns
    -------
    label, confidence, probabilities
    """
    if USES_TOKENS:
        logits = _sess.run(None, {_in_name: token_ids.astype(np.int64)})[0][0]
    else:
        logits = _sess.run(None, {_in_name: fvec_scaled.astype(np.float32)})[0][0]

    probs = scipy.special.softmax(logits)
    pred_cls = int(np.argmax(probs))
    pred_conf = float(probs[pred_cls])
    pred_label = CLASS_NAMES[pred_cls]

    return pred_label, pred_conf, probs.tolist()