"""app/core/config.py"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME:    str  = "WAF-ML"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False

    MONGO_URI: str = "mongodb://mongodb:27017"
    MONGO_DB:  str = "waf_db"

    L2A_ONNX_PATH:      Path = Path("ml/exported_models/layer2a_best.onnx")
    L2A_THRESHOLD_PATH: Path = Path("ml/exported_models/layer2a_best_threshold.txt")
    L2B_ONNX_PATH:      Path = Path("ml/exported_models/layer2b_best.onnx")
    SCALER_PATH:        Path = Path("ml/exported_models/scaler_l2a.pkl")

    # ── Threat scoring — CRC Decision 2 locked config ──────────────────────
    # Source: fork-of-06-end-to-end-eval.ipynb ("06-end-to-end-eval (2).ipynb"),
    # cells 30-40 — selected on validation only, evaluated once on untouched
    # test set. Do not change without re-running that validation sweep.
    #
    # ESCALATION_THRESHOLD gates whether a request even reaches Layer 2B:
    # if L2A's raw reconstruction-error score is below this, the request is
    # "clearly normal" and allowed immediately, skipping L2B entirely
    # (this is the "selective escalation" — most traffic never touches L2B).
    # It is P85 of normal-only L2A validation scores, NOT the same as the
    # model's own operating threshold below.
    ESCALATION_THRESHOLD:  float = 0.00077472

    # L2A's own trained/calibrated anomaly threshold (maximise recall subject
    # to FPR<=5% on validation only). Distinct from ESCALATION_THRESHOLD above;
    # kept for health-monitor recalibration and for reporting L2A's own recall.
    # Overridden at runtime by the value baked into layer2a_best_threshold.txt.

    L2A_SCORE_MULTIPLIER:  float = 15.0   # c_L2A = min(50, l2a_score * this)
    L2B_CONF_MULTIPLIER:   float = 90.0   # c_L2B = l2b_confidence * this (0 if predicted "normal")
    SCORE_LOG_THRESHOLD:   int = 30
    SCORE_BLOCK_THRESHOLD: int = 70

    RATE_LIMIT_PER_MIN:    int = 100
    PROTECTED_APP_URL: str = "http://127.0.0.1:5000"
    HEALTH_CHECK_INTERVAL_SEC: int   = 60
    ERROR_RATE_THRESHOLD:      float = 0.10   # CRC Decision 1 / NB08: HEALTH_ERROR_THRESHOLD
    RETRAIN_MIN_SAMPLES:       int   = 200
    HEALTH_CAPTURE_PCT:        float = 100.0  # % of eligible allow/log traffic captured on breach

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()