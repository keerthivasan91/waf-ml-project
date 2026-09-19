"""app/api/routes/dashboard.py — Jinja2 SSR pages"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.db.queries import (get_dashboard_stats, get_recent_threats, get_pending_feedback,
                             get_pending_feedback_count, get_recent_retrain_logs,
                             get_recent_health_audits)

router    = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats   = await get_dashboard_stats()
    threats = await get_recent_threats(limit=20)
    pending_reviews = await get_pending_feedback_count()
    return templates.TemplateResponse(
    request=request,
    name="dashboard.html",
    context={
        "request": request,
        "stats": stats,
        "threats": threats,
        "pending_reviews": pending_reviews,
    },
)

@router.get("/dashboard/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    from app.db.queries import get_recent_logs
    logs = await get_recent_logs(limit=200)
    return templates.TemplateResponse(
        request=request,
        name="logs.html",
        context={"request": request, "logs": logs}
    )

@router.get("/dashboard/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    items = await get_pending_feedback(limit=100)
    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context={"request": request, "items": items}
    )

@router.get("/dashboard/retraining", response_class=HTMLResponse)
async def retraining_page(request: Request):
    pending_count = await get_pending_feedback_count()
    retrain_logs = await get_recent_retrain_logs(limit=20)
    audit_logs = await get_recent_health_audits(limit=20)
    return templates.TemplateResponse(
        request=request,
        name="retraining.html",
        context={
            "request": request,
            "pending_count": pending_count,
            "retrain_logs": retrain_logs,
            "audit_logs": audit_logs,
        }
    )

@router.get("/dashboard/threats", response_class=HTMLResponse)
async def threats_page(request: Request):
    from app.db.queries import get_recent_threats
    threats = await get_recent_threats(limit=100)
    return templates.TemplateResponse(
        request=request,
        name="threats.html",
        context={"request": request, "threats": threats}
    )

@router.get("/simulator", response_class=HTMLResponse)
async def simulator_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulator.html",
        context={
            "request": request,
            "version": "1.0.0",
        },
    )


@router.get("/dashboard/models", response_class=HTMLResponse)
async def models_page(request: Request):
    import app.services.layer2a_anomaly as l2a
    import app.services.layer2b_deep as l2b
    from app.core.config import settings
    import time

    def file_meta(path):
        if not path.exists():
            return {"exists": False, "path": str(path), "size_kb": 0, "modified_human": "—"}
        stat = path.stat()
        return {
            "exists": True,
            "path": str(path),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        }

    info = {
        "layer2a": {
            **file_meta(settings.L2A_ONNX_PATH),
            "own_threshold":        l2a._threshold,
            "escalation_threshold": settings.ESCALATION_THRESHOLD,
            "input_name":           l2a._in_name,
        },
        "layer2b": {
            **file_meta(settings.L2B_ONNX_PATH),
            "input_name": l2b._in_name,
            "uses_tokens": l2b.USES_TOKENS,
            "class_names": l2b.CLASS_NAMES,
        },
        "scaler": file_meta(settings.SCALER_PATH),
    }
    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={"request": request, "info": info}
    )