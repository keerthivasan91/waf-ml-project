"""app/core/config.py"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME:    str  = "WAF-ML"
    APP_VERSION: str = "1.0.0"
    DEBUG:       bool = False

    # Local/native development defaults. Docker Compose overrides these with
    # service names so the same application works in both environments.
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB:  str = "waf_db"

    L2A_ONNX_PATH:      Path = Path("ml/exported_models/layer2a_best.onnx")
    L2A_THRESHOLD_PATH: Path = Path("ml/exported_models/layer2a_best_threshold.txt")
    L2B_ONNX_PATH:      Path = Path("ml/exported_models/layer2b_best.onnx")
    SCALER_PATH:        Path = Path("ml/exported_models/scaler_l2a.pkl")

    # ── Local adaptive-retraining assets ───────────────────────────────────
    # These point to frozen baseline training artifacts kept on the same
    # machine as the WAF. They can be overridden with absolute paths in .env.
    LOCAL_RETRAIN_ENABLED: bool = True
    LOCAL_RETRAIN_AUTO_PROMOTE: bool = True
    RETRAIN_BASE_CHECKPOINT: Path = Path(
        "ml/retraining_artifacts/base/layer2b_bigru_checkpoint.pt"
    )
    # Candidate checkpoints promoted by local retraining are written here so
    # the frozen baseline above is never overwritten.
    RETRAIN_DEPLOYED_CHECKPOINT: Path = Path(
        "ml/retraining_artifacts/deployed/layer2b_bigru_checkpoint.pt"
    )
    RETRAIN_BASE_TRAIN_X: Path = Path(
        "ml/retraining_artifacts/base/l2b_train_X_tokens.npy"
    )
    RETRAIN_BASE_TRAIN_Y: Path = Path(
        "ml/retraining_artifacts/base/l2b_train_y.npy"
    )
    RETRAIN_VAL_X: Path = Path(
        "ml/retraining_artifacts/base/l2b_val_X_tokens.npy"
    )
    RETRAIN_VAL_Y: Path = Path(
        "ml/retraining_artifacts/base/l2b_val_y.npy"
    )
    RETRAIN_L2A_NORMAL_VAL: Path = Path(
        "ml/retraining_artifacts/base/l2a_normal_val.npy"
    )
    RETRAIN_L2A_ATTACK_VAL: Path = Path(
        "ml/retraining_artifacts/base/l2a_attack_val.npy"
    )
    RETRAIN_LOCAL_RUNS_DIR: Path = Path("ml/retraining_runs")

    # Local retraining hyperparameters. These are intentionally configurable
    # without modifying code so a native Windows/Linux/macOS setup can use the
    # same application.
    RETRAIN_HOLDOUT_FRACTION: float = 0.20
    RETRAIN_EPOCHS: int = 15
    RETRAIN_LEARNING_RATE: float = 3e-5
    RETRAIN_OVERSAMPLE_FACTOR: int = 5
    RETRAIN_VAL_F1_TOLERANCE: float = 0.005
    RETRAIN_L2A_FPR_CAP: float = 0.05
    RETRAIN_MAX_BATCH_RATIO: float = 0.10
    RETRAIN_SEED: int = 42
    RETRAIN_BATCH_SIZE: int = 256
    # 0 = use the full frozen baseline train set. A positive value is useful
    # for a local CPU smoke test; validation always remains the full frozen set.
    RETRAIN_BASE_TRAIN_MAX_SAMPLES: int = 0
    # 0 = keep PyTorch's default; positive value explicitly sets torch threads.
    RETRAIN_TORCH_THREADS: int = 0

    # ── Threat scoring — CRC Decision 2 locked config ──────────────────────
    ESCALATION_THRESHOLD:  float = 0.00077472
    L2A_SCORE_MULTIPLIER:  float = 15.0
    L2B_CONF_MULTIPLIER:   float = 90.0
    SCORE_LOG_THRESHOLD:   int = 30
    SCORE_BLOCK_THRESHOLD: int = 70

    RATE_LIMIT_PER_MIN:    int = 100
    # Local/native default; Docker Compose overrides this to http://protected-app:5000.
    PROTECTED_APP_URL: str = "http://127.0.0.1:5000"
    HEALTH_CHECK_INTERVAL_SEC: int   = 60
    ERROR_RATE_THRESHOLD:      float = 0.10
    RETRAIN_MIN_SAMPLES:       int   = 200
    HEALTH_CAPTURE_PCT:        float = 100.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
