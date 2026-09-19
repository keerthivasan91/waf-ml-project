"""app/api/routes/feedback.py — human review queue"""
import json
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from app.db.queries import (
    get_pending_feedback,
    get_latest_retrain_batch,
    get_retrain_batch,
)
from app.db.collections import feedback_queue
from app.services.adaptive_retrain import run_retrain_cycle

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

@router.get("/retrain-batches/latest")
async def latest_retrain_batch():
    """Return the latest clean batch manifest for offline training."""
    batch = await get_latest_retrain_batch()
    if not batch:
        raise HTTPException(404, "No retraining batch has been prepared")
    return batch


@router.get("/retrain-batches/{batch_id}/export")
async def export_retrain_batch(batch_id: str):
    """Download one clean retraining batch as JSON for Kaggle/Colab."""
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
