"""app/api/routes/feedback.py — human review queue"""
from fastapi import APIRouter, Body, HTTPException
from app.db.queries import get_pending_feedback
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