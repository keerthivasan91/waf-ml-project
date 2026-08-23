"""app/api/routes/health.py"""
from fastapi import APIRouter
from app.db.mongodb import get_db
from app.db.queries import get_dashboard_stats
import app.services.health_monitor as health_monitor

router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("/")
async def health_check():
    """Basic liveness check."""
    try:
        await get_db().command("ping")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok}

@router.get("/stats")
async def stats():
    """Returns 24h and 1h traffic statistics for the dashboard."""
    return await get_dashboard_stats()

@router.post("/trigger-audit")
async def trigger_audit(error_rate: float = 1.0):
    """
    Manually run the CRC Decision 1 health-audit cycle (capture -> re-score
    -> disagreement -> feedback_queue) instead of waiting for the next
    60s monitor tick. Useful for testing/demoing the feedback loop.
    """
    return await health_monitor._trigger_audit(error_rate)