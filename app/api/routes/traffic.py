"""app/api/routes/traffic.py — on-demand WAF analysis endpoint"""
import uuid, time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.models.schemas.request import IncomingRequest
from app.models.schemas.threat  import ThreatResult
import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
import app.services.threat_scorer  as scorer
from app.services.feature_extractor import extract
from app.db.queries import insert_request_log, insert_threat_event

router = APIRouter(prefix="/api/traffic", tags=["traffic"])

@router.post("/analyze", response_model=ThreatResult)
async def analyze(req: IncomingRequest):
    """Run a request through the full WAF pipeline and return the threat result."""
    t0         = time.perf_counter()
    request_id = str(uuid.uuid4())
    url        = req.url
    body       = req.body

    # L1
    blocked, reason = l1.check(url, body)
    if blocked:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        return ThreatResult(request_id=request_id, decision="block",
                            score=100, label=reason, layer="L1", latency_ms=ms)

    # Features
    req_dict = {"url": url, "method": req.method, "headers": req.headers, "body": body}
    fvec, token_ids = extract(req_dict)

    # L2A
    is_anomaly, l2a_score = l2a.infer(fvec)
    if not is_anomaly:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        return ThreatResult(request_id=request_id, decision="allow",
                            score=0, label="normal", layer="L2A",
                            l2a_score=round(l2a_score, 5), latency_ms=ms)

    # L2B
    label, confidence, proba = l2b.infer(fvec, token_ids)
    score, decision = scorer.compute(l2a_score, label, confidence)
    ms = round((time.perf_counter() - t0) * 1000, 2)

    result = ThreatResult(
        request_id=request_id, decision=decision, score=score,
        label=label, layer="L2B", confidence=round(confidence, 4),
        l2a_score=round(l2a_score, 5),
        l2a_contrib=round(min(50.0, l2a_score * 15), 2),
        l2b_contrib=round(confidence * 50 if label != "normal" else 0.0, 2),
        latency_ms=ms,
    )

    # persist
    now = datetime.utcnow()
    doc = {**result.model_dump(), "ip": req.ip, "body_len": len(body), "timestamp": now}
    await insert_request_log(doc)
    if decision in ("block", "log"):
        await insert_threat_event(doc)

    return result