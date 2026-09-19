"""app/services/local_retraining.py

Local-machine adaptive retraining worker.

This mirrors the SpectraSpatialAI pattern: the FastAPI process accepts a
validated feedback batch, starts a background training thread on the same
machine, writes a local run directory, backs up the active artifacts, promotes
the validated result, and hot-reloads the runtime models.

The actual ML work is delegated to ml.retraining.offline_retrain so the web
process never imports PyTorch during normal inference startup.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.db.collections import model_versions, retrain_batches, retrain_log


ROOT = Path(__file__).resolve().parents[2]
_LOCK = threading.Lock()
_JOB: dict[str, Any] = {
    "status": "idle",
    "batch_id": None,
    "run_dir": None,
    "log_path": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "return_code": None,
}


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_local_retrain_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_JOB)


def _set_job(**updates: Any) -> None:
    with _LOCK:
        _JOB.update(updates)


def _required_inputs() -> dict[str, Path]:
    return {
        "checkpoint": _resolve(settings.RETRAIN_BASE_CHECKPOINT),
        "base_train_x": _resolve(settings.RETRAIN_BASE_TRAIN_X),
        "base_train_y": _resolve(settings.RETRAIN_BASE_TRAIN_Y),
        "val_x": _resolve(settings.RETRAIN_VAL_X),
        "val_y": _resolve(settings.RETRAIN_VAL_Y),
        "l2a_normal_val": _resolve(settings.RETRAIN_L2A_NORMAL_VAL),
        "l2a_attack_val": _resolve(settings.RETRAIN_L2A_ATTACK_VAL),
        "l2a_model": _resolve(settings.L2A_ONNX_PATH),
        "l2a_scaler": _resolve(settings.SCALER_PATH),
    }


def validate_local_environment() -> list[str]:
    missing = [
        f"{name}: {path}"
        for name, path in _required_inputs().items()
        if not path.exists()
    ]
    try:
        _resolve(settings.RETRAIN_LOCAL_RUNS_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        missing.append(f"runs_dir: {_resolve(settings.RETRAIN_LOCAL_RUNS_DIR)} ({exc})")
    return missing


def _training_command(batch_path: Path, output_dir: Path) -> list[str]:
    base = _required_inputs()
    return [
        sys.executable,
        "-m",
        "ml.retraining.offline_retrain",
        "--batch",
        str(batch_path),
        "--base-checkpoint",
        str(base["checkpoint"]),
        "--base-model-dir",
        str(_resolve(settings.L2A_ONNX_PATH).parent),
        "--base-train-x",
        str(base["base_train_x"]),
        "--base-train-y",
        str(base["base_train_y"]),
        "--val-x",
        str(base["val_x"]),
        "--val-y",
        str(base["val_y"]),
        "--l2a-normal-val",
        str(base["l2a_normal_val"]),
        "--l2a-attack-val",
        str(base["l2a_attack_val"]),
        "--output-dir",
        str(output_dir),
        "--min-samples",
        str(settings.RETRAIN_MIN_SAMPLES),
        "--holdout-fraction",
        str(settings.RETRAIN_HOLDOUT_FRACTION),
        "--epochs",
        str(settings.RETRAIN_EPOCHS),
        "--learning-rate",
        str(settings.RETRAIN_LEARNING_RATE),
        "--oversample-factor",
        str(settings.RETRAIN_OVERSAMPLE_FACTOR),
        "--val-f1-tolerance",
        str(settings.RETRAIN_VAL_F1_TOLERANCE),
        "--l2a-fpr-cap",
        str(settings.RETRAIN_L2A_FPR_CAP),
        "--max-batch-ratio",
        str(settings.RETRAIN_MAX_BATCH_RATIO),
        "--seed",
        str(settings.RETRAIN_SEED),
    ]


def _required_outputs(output_dir: Path) -> dict[str, Path]:
    return {
        "l2a_model": output_dir / "layer2a_best.onnx",
        "l2b_model": output_dir / "layer2b_best.onnx",
        "l2b_checkpoint": output_dir / "layer2b_bigru_checkpoint.pt",
        "threshold": output_dir / "layer2a_best_threshold.txt",
        "scaler": output_dir / "scaler_l2a.pkl",
    }


def _promote(run_dir: Path) -> dict[str, Any]:
    output_dir = run_dir / "artifacts"
    outputs = _required_outputs(output_dir)
    missing = [f"{name}: {path}" for name, path in outputs.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Retraining completed without required artifacts: " + "; ".join(missing)
        )

    active = {
        "l2a_model": _resolve(settings.L2A_ONNX_PATH),
        "l2b_model": _resolve(settings.L2B_ONNX_PATH),
        "l2b_checkpoint": _resolve(settings.RETRAIN_BASE_CHECKPOINT),
        "threshold": _resolve(settings.L2A_THRESHOLD_PATH),
        "scaler": _resolve(settings.SCALER_PATH),
    }

    backup_dir = run_dir / "backup_before_promotion"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for dst in active.values():
        if dst.exists():
            shutil.copy2(dst, backup_dir / dst.name)

    source_data = output_dir / "layer2a_best.onnx.data"
    target_data = active["l2a_model"].with_name("layer2a_best.onnx.data")
    had_target_data = target_data.exists()
    if source_data.exists() and had_target_data:
        shutil.copy2(target_data, backup_dir / target_data.name)

    copies = {
        outputs["l2a_model"]: active["l2a_model"],
        outputs["l2b_model"]: active["l2b_model"],
        outputs["l2b_checkpoint"]: active["l2b_checkpoint"],
        outputs["threshold"]: active["threshold"],
        outputs["scaler"]: active["scaler"],
    }

    try:
        for src, dst in copies.items():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        if source_data.exists():
            shutil.copy2(source_data, target_data)

        import app.services.feature_extractor as feature_extractor
        import app.services.layer2a_anomaly as l2a
        import app.services.layer2b_deep as l2b

        feature_extractor.reload_normalizer()
        l2a.load()
        l2b.load()
    except Exception:
        for dst in active.values():
            backup = backup_dir / dst.name
            if backup.exists():
                shutil.copy2(backup, dst)

        if source_data.exists():
            backup_data = backup_dir / target_data.name
            if backup_data.exists():
                shutil.copy2(backup_data, target_data)
            elif target_data.exists() and not had_target_data:
                target_data.unlink()

        import app.services.feature_extractor as feature_extractor
        import app.services.layer2a_anomaly as l2a
        import app.services.layer2b_deep as l2b

        feature_extractor.reload_normalizer()
        l2a.load()
        l2b.load()
        raise

    return {
        "promoted": True,
        "backup_dir": str(backup_dir),
        "active_model_dir": str(active["l2b_model"].parent),
    }


async def _persist_result(
    batch_id: str,
    status: str,
    payload: dict[str, Any],
) -> None:
    now = datetime.utcnow()
    await retrain_batches().update_one(
        {"batch_id": batch_id},
        {
            "$set": {
                "status": status,
                "finished_at": now,
                "local_retrain": payload,
            }
        },
    )
    await retrain_log().insert_one(
        {
            "timestamp": now,
            "status": status,
            "batch_id": batch_id,
            "run_type": "local",
            "n_raw": payload.get("n_raw", 0),
            "n_clean": payload.get("n_clean", 0),
            "n_rejected": payload.get("n_rejected", 0),
            "report": payload.get("report"),
            "run_dir": payload.get("run_dir"),
            "error": payload.get("error"),
        }
    )
    if status == "deployed":
        await model_versions().insert_one(
            {
                "timestamp": now,
                "event": "local_retrain_promote",
                "batch_id": batch_id,
                "run_dir": payload.get("run_dir"),
                "report": payload.get("report"),
            }
        )


def _worker(
    batch_id: str,
    batch_path: Path,
    run_dir: Path,
    loop: asyncio.AbstractEventLoop,
) -> None:
    log_path = run_dir / "training.log"
    output_dir = run_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        missing = validate_local_environment()
        if missing:
            raise RuntimeError(
                "Local retraining environment is incomplete. Missing: "
                + "; ".join(missing)
            )

        command = _training_command(batch_path, output_dir)
        _set_job(
            status="running",
            command=command,
            log_path=str(log_path),
            run_dir=str(run_dir),
        )

        logger.info("Local WAF retraining started | batch=%s", batch_id)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        _set_job(return_code=completed.returncode)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Local training process exited with code {completed.returncode}. "
                f"See {log_path}"
            )

        report_path = output_dir / "retraining_report.json"
        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))

        promotion = None
        if settings.LOCAL_RETRAIN_AUTO_PROMOTE:
            promotion = _promote(run_dir)

        final_status = "deployed" if promotion else "trained"
        payload = {
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "report": report,
            "promotion": promotion,
            "n_raw": int(batch.get("n_raw", len(batch.get("samples", [])))),
            "n_clean": int(batch.get("n_clean", len(batch.get("samples", [])))),
            "n_rejected": int(batch.get("n_rejected", 0)),
        }

        _set_job(
            status=final_status,
            finished_at=_now(),
            error=None,
            report=report,
            promotion=promotion,
        )
        asyncio.run_coroutine_threadsafe(
            _persist_result(batch_id, final_status, payload),
            loop,
        )
        logger.info(
            "Local WAF retraining finished | batch=%s | status=%s",
            batch_id,
            final_status,
        )
    except Exception as exc:
        logger.exception("Local WAF retraining failed | batch=%s", batch_id)
        payload = {
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "error": str(exc),
        }
        _set_job(
            status="failed",
            finished_at=_now(),
            error=str(exc),
            report=None,
        )
        asyncio.run_coroutine_threadsafe(
            _persist_result(batch_id, "failed", payload),
            loop,
        )


async def start_local_retrain(batch: dict[str, Any]) -> dict[str, Any]:
    """Start one local training job in a background thread."""
    if not settings.LOCAL_RETRAIN_ENABLED:
        raise RuntimeError("Local retraining is disabled by LOCAL_RETRAIN_ENABLED=false")

    with _LOCK:
        if _JOB.get("status") in {"starting", "running"}:
            raise RuntimeError("A local retraining job is already running.")

    batch_id = str(batch.get("batch_id") or "").strip()
    if not batch_id:
        raise RuntimeError("Retraining batch has no batch_id.")

    run_dir = _resolve(settings.RETRAIN_LOCAL_RUNS_DIR) / batch_id
    run_dir.mkdir(parents=True, exist_ok=True)
    batch_path = run_dir / "input_batch.json"
    batch_path.write_text(
        json.dumps(batch, indent=2, default=str),
        encoding="utf-8",
    )

    await retrain_batches().update_one(
        {"batch_id": batch_id},
        {
            "$set": {
                "status": "running",
                "local_retrain_started_at": datetime.utcnow(),
                "local_retrain_run_dir": str(run_dir),
            }
        },
    )

    loop = asyncio.get_running_loop()
    _set_job(
        status="starting",
        batch_id=batch_id,
        run_dir=str(run_dir),
        log_path=str(run_dir / "training.log"),
        started_at=_now(),
        finished_at=None,
        error=None,
        return_code=None,
    )

    thread = threading.Thread(
        target=_worker,
        args=(batch_id, batch_path, run_dir, loop),
        daemon=True,
        name=f"waf-local-retrain-{batch_id[:8]}",
    )
    thread.start()

    return get_local_retrain_status()
