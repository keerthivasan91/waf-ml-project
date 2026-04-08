"""app/api/routes/logs.py"""
from fastapi import APIRouter, Query
from app.db.queries import get_recent_logs, get_recent_threats

router = APIRouter(prefix="/api/logs", tags=["logs"])

@router.get("/recent")
async def recent_logs(
    limit: int = Query(100, ge=1, le=500),
    decision: str = Query(None, pattern="^(allow|log|block)$"),
):
    return await get_recent_logs(limit=limit, decision_filter=decision)

@router.get("/threats")
async def recent_threats(limit: int = Query(50, ge=1, le=200)):
    return await get_recent_threats(limit=limit)