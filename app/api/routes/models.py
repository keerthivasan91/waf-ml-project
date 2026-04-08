"""
app/api/routes/models.py

Endpoints for inspecting loaded model metadata and triggering
manual model reload without restarting the container.
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import os, time
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
from app.core.config import settings
from app.db.collections import model_versions

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/info")
async def model_info():
    """
    Returns metadata about the currently loaded ONNX models:
    file paths, file sizes, modification times, and the L2A threshold.
    """
    def file_meta(path: Path) -> dict:
        if not path.exists():
            return {"exists": False, "path": str(path)}
        stat = path.stat()
        return {
            "exists":        True,
            "path":          str(path),
            "size_kb":       round(stat.st_size / 1024, 1),
            "modified":      stat.st_mtime,
            "modified_human": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
            ),
        }

    return {
        "layer2a": {
            **file_meta(settings.L2A_ONNX_PATH),
            "threshold": l2a._threshold,
            "input_name": l2a._in_name,
        },
        "layer2b": {
            **file_meta(settings.L2B_ONNX_PATH),
            "input_name":   l2b._in_name,
            "uses_tokens":  l2b._uses_tokens,
            "class_names":  l2b.CLASS_NAMES,
        },
        "scaler": file_meta(settings.SCALER_PATH),
        "threshold_file": file_meta(settings.L2A_THRESHOLD_PATH),
    }


@router.post("/reload")
async def reload_models():
    """
    Hot-reload both ONNX models from disk without restarting the container.
    Use after placing new exported_models/ files in the mounted volume.
    """
    errors = []
    try:
        l2a.load()
    except Exception as e:
        errors.append(f"L2A: {e}")
    try:
        l2b.load()
    except Exception as e:
        errors.append(f"L2B: {e}")

    if errors:
        raise HTTPException(status_code=500,
                            detail={"reload_errors": errors})

    # Log the reload event
    import datetime
    await model_versions().insert_one({
        "timestamp":   datetime.datetime.utcnow(),
        "event":       "hot_reload",
        "l2a_path":    str(settings.L2A_ONNX_PATH),
        "l2b_path":    str(settings.L2B_ONNX_PATH),
        "l2a_threshold": l2a._threshold,
    })

    return {
        "status":      "reloaded",
        "l2a_threshold": l2a._threshold,
        "l2b_uses_tokens": l2b._uses_tokens,
    }


@router.get("/history")
async def model_history(limit: int = 20):
    """Returns the last N model reload events from MongoDB."""
    cursor = model_versions().find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)