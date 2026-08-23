"""app/services/health_monitor.py — server health monitor + feedback loop

CRC Decision 1 (Section III-E), matches NB08 exactly. On a health breach,
this describes ONLY what NB08 actually demonstrates:

    health breach triggers capture of recent allow/log traffic
    -> re-scoring against current thresholds
    -> disagreement identification
    -> verified feedback into the adaptive retraining pipeline (via
       feedback_queue, same human-review path as borderline "log" traffic)

It does NOT automatically tighten thresholds during a breach or relax them
on recovery — NB08 does not implement or test that, so this module must not
either. This is a manuscript-matching correction, not an architecture change.
"""
import asyncio
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.logging import logger
from app.db.queries import (insert_health_snapshot, get_recent_allow_log_traffic,
                             insert_health_audit)
from app.db.collections import feedback_queue
from app.services.reaudit import reaudit
import httpx

_running = False
_last_audit_at: datetime | None = None


async def start_monitor() -> None:
    global _running, _last_audit_at
    _running = True
    _last_audit_at = datetime.utcnow()
    asyncio.create_task(_loop())
    logger.info("Health monitor started (interval=%ds)",
                settings.HEALTH_CHECK_INTERVAL_SEC)

async def stop_monitor() -> None:
    global _running
    _running = False

async def _loop() -> None:
    while _running:
        await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL_SEC)
        try:
            snapshot = await _check_app_health()
            await insert_health_snapshot(snapshot)
            if snapshot.get("error_rate", 0) >= settings.ERROR_RATE_THRESHOLD:
                logger.warning("Error rate %.2f%% exceeds threshold — triggering health audit",
                               snapshot["error_rate"] * 100)
                await _trigger_audit(snapshot["error_rate"])
        except Exception as e:
            logger.error("Health monitor error: %s", e)

async def _check_app_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:  # was 5.0
            r = await client.get(f"{settings.PROTECTED_APP_URL}/health")
            app_data = r.json() if r.status_code == 200 else {}
    except Exception:
        app_data = {}

    return {
        "timestamp":   datetime.utcnow(),
        "app_ok":      bool(app_data),
        "error_rate":  app_data.get("error_rate", 0.0),
        "latency_p99": app_data.get("latency_p99", 0.0),
        "cpu_pct":     app_data.get("cpu_pct", 0.0),
    }


async def _trigger_audit(error_rate: float) -> dict:
    """
    CRC Decision 1: capture -> re-score -> disagreement -> feedback.

    1. Capture allow/log traffic since the last audit (or health-check
       interval, whichever is more recent) — "passed to the backend" means
       allow or log, never block (blocked traffic never reached the app).
    2. Re-score each captured request with app/services/reaudit.py, i.e.
       CURRENT thresholds and CURRENT models — identical logic to the live
       pipeline, run offline.
    3. A "disagreement" is a request whose ORIGINAL decision was allow/log
       but whose re-audit label is non-normal (matches NB08's definition
       exactly — not an arbitrary score-delta threshold).
    4. Disagreements go into feedback_queue for human verification — the
       SAME verified-review path already used for borderline "log" traffic.
       Anti-poisoning, human review, and the actual retrain run happen in
       app/services/adaptive_retrain.py; this function does not duplicate
       that mechanism, only feeds it (per NB08's own framing).
    """
    global _last_audit_at

    window_start = _last_audit_at or (datetime.utcnow() - timedelta(seconds=settings.HEALTH_CHECK_INTERVAL_SEC))
    captured = await get_recent_allow_log_traffic(window_start, sample_pct=settings.HEALTH_CAPTURE_PCT)
    _last_audit_at = datetime.utcnow()

    if not captured:
        logger.info("Health audit: no eligible allow/log traffic to capture in window")
        report = {
            "timestamp": datetime.utcnow(), "error_rate": error_rate,
            "error_threshold": settings.ERROR_RATE_THRESHOLD,
            "traffic_window_captured": 0, "feedback_candidates": 0,
        }
        await insert_health_audit(report)
        return report

    disagreements = []
    for row in captured:
        try:
            result = reaudit(row["url"], row.get("method", "GET"), row.get("body", ""))
        except Exception as e:
            logger.error("Reaudit failed for request_id=%s: %s", row.get("request_id"), e)
            continue

        # NB08 disagreement definition: original allow/log, re-audit says non-normal
        if row["decision"] in ("allow", "log") and result["label"] != "normal":
            disagreements.append({**row, **{
                "reaudit_decision":  result["decision"],
                "reaudit_label":     result["label"],
                "reaudit_confidence": result["confidence"],
                "reaudit_l2a_score": result["l2a_score"],
            }})

    # Push disagreements into the existing verified-review feedback path.
    for d in disagreements:
        feedback_doc = {
            **d,
            "verified_label": None,
            "poisoning_flag": False,
            "auto_classified": False,
            "source": "health_audit",
        }
        try:
            await feedback_queue().insert_one(feedback_doc)
        except Exception as e:
            logger.error("Health-audit feedback insert failed: %s", e)

    report = {
        "timestamp": datetime.utcnow(),
        "error_rate": error_rate,
        "error_threshold": settings.ERROR_RATE_THRESHOLD,
        "traffic_window_captured": len(captured),
        "capture_pct": settings.HEALTH_CAPTURE_PCT,
        "feedback_candidates": len(disagreements),
    }
    await insert_health_audit(report)
    logger.info("Health audit complete: captured=%d disagreements=%d",
                len(captured), len(disagreements))
    return report
