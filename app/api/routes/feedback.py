"""app/api/routes/feedback.py — human review queue"""
import json
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from app.core.config import settings
from app.db.queries import (
    get_pending_feedback,
    get_latest_retrain_batch,
    get_retrain_batch,
)
from app.db.collections import feedback_queue, retrain_batches
from app.services.adaptive_retrain import run_retrain_cycle
from app.services.local_retraining import (
    get_local_retrain_status,
    start_local_retrain,
)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

@router.get("/pending")
async def pending_feedback(limit: int = 100):
    return await get_pending_feedback(limit=limit)

@router.post("/review/{request_id}")
async def submit_review(
    request_id: str,
    verified_label: str = Body(..., embed=True),
    is_poisoning:   bool = Body(False, embed=True),
):
    """Human reviewer marks a borderline request with its true label."""
    valid = {"normal", "sqli", "xss", "lfi", "other_attack", "false_positive"}
    if verified_label not in valid:
        raise HTTPException(400, f"Invalid label. Must be one of {valid}")

    result = await feedback_queue().update_one(
        {"request_id": request_id},
        {"$set": {"verified_label": verified_label, "poisoning_flag": is_poisoning}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Request not found in feedback queue")
    return {"status": "ok", "request_id": request_id, "verified_label": verified_label}

@router.post("/trigger-retrain")
async def trigger_retrain():
    """Manually trigger the adaptive retraining pipeline."""
    result = await run_retrain_cycle()
    return result

@router.post("/local-retrain/start")
async def start_local_retrain_endpoint(batch_id: str | None = None):
    """Start validated feedback training on the same machine as the WAF.

    A batch_id can be supplied to resume a specific prepared batch. This is
    important after a worker crash/restart: MongoDB may still say "running"
    even though the in-memory local worker is no longer active.
    """
    if not settings.LOCAL_RETRAIN_ENABLED:
        raise HTTPException(503, "Local retraining is disabled")

    state = get_local_retrain_status()
    if state.get("status") in {"starting", "running"}:
        raise HTTPException(409, "A local retraining job is already running")

    if batch_id:
        batch = await get_retrain_batch(batch_id)
        if not batch:
            raise HTTPException(404, f"Retraining batch not found: {batch_id}")
    else:
        batch = await get_latest_retrain_batch()

    # If the requested/latest batch is missing or already deployed, prepare a
    # fresh batch. A stale Mongo "running" state is reusable when no local
    # worker is actually active; this happens after an interrupted process or
    # FastAPI restart.
    if not batch or batch.get("status") == "deployed":
        prepared = await run_retrain_cycle()
        if prepared.get("status") != "queued":
            return prepared
        batch = await get_retrain_batch(prepared["batch_id"])

    if not batch:
        raise HTTPException(404, "No validated retraining batch is available")

    # A batch that is marked running in Mongo but has no active local worker is
    # stale. Reset it to queued before resuming the exact same validated batch.
    if batch.get("status") == "running":
        await retrain_batches().update_one(
            {"batch_id": batch["batch_id"]},
            {
                "$set": {
                    "status": "queued",
                    "resume_note": "Resumed after stale local worker state.",
                }
            },
        )
        batch["status"] = "queued"

    try:
        return await start_local_retrain(batch)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/local-retrain/status")
async def local_retrain_status():
    """Return the current local-machine training job state."""
    return get_local_retrain_status()


@router.get("/retrain-batches/latest")
async def latest_retrain_batch():
    """Return the latest clean batch manifest for offline training."""
    batch = await get_latest_retrain_batch()
    if not batch:
        raise HTTPException(404, "No retraining batch has been prepared")
    return batch


@router.get("/retrain-batches/latest/export")
async def export_latest_retrain_batch():
    """Download the latest clean retraining batch as JSON."""
    batch = await get_latest_retrain_batch()
    if not batch:
        raise HTTPException(404, "No retraining batch has been prepared")

    payload = json.dumps(batch, default=str, indent=2)
    batch_id = batch.get("batch_id", "latest")
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="waf_retrain_{batch_id}.json"'
        },
    )


@router.get("/retrain-batches/{batch_id}/export")
async def export_retrain_batch(batch_id: str):
    """Download one clean retraining batch as JSON for local backup/export."""
    batch = await get_retrain_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Retraining batch not found")

    payload = json.dumps(batch, default=str, indent=2)
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="waf_retrain_{batch_id}.json"'
        },
    )
