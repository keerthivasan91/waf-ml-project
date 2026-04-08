"""app/api/routes/dashboard.py — Jinja2 SSR pages"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.db.queries import get_dashboard_stats, get_recent_threats, get_pending_feedback

router    = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats   = await get_dashboard_stats()
    threats = await get_recent_threats(limit=20)
    return templates.TemplateResponse(
    request=request,
    name="dashboard.html",
    context={
        "request": request,
        "stats": stats,
        "threats": threats,
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