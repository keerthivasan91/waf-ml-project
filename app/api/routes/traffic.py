"""app/api/routes/traffic.py — on-demand WAF analysis endpoint

Kept in sync with app/middleware/waf_middleware.py's decision logic —
this is a second entry point into the SAME pipeline (selective
escalation via settings.ESCALATION_THRESHOLD, not L2A's own threshold),
not an independent implementation. If you change the pipeline, change
it in waf_middleware.py first and mirror it here.
"""
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
from app.core.config import settings
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

    # Features (scaled — see app/services/feature_extractor.py)
    req_dict = {"url": url, "method": req.method, "headers": req.headers, "body": body}
    fvec, token_ids = extract(req_dict)

    # L2A — raw score only; routing decision uses settings.ESCALATION_THRESHOLD,
    # NOT L2A's own operating threshold (see layer2a_anomaly.py docstring)
    l2a_score = l2a.score(fvec)

    if l2a_score < settings.ESCALATION_THRESHOLD:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        result = ThreatResult(request_id=request_id, decision="allow",
                            score=0, label="normal", layer="L2A",
                            l2a_score=round(l2a_score, 8), confidence=1.0,
                            latency_ms=ms)
        now = datetime.utcnow()
        doc = {**result.model_dump(), "ip": req.ip, "body_len": len(body),
               "method": req.method, "url": url[:500], "body": body[:2000],
               "timestamp": now}
        await insert_request_log(doc)
        return result

    # L2B
    label, confidence, proba = l2b.infer(fvec, token_ids)
    score, decision = scorer.compute(l2a_score, label, confidence)
    ms = round((time.perf_counter() - t0) * 1000, 2)

    l2a_contrib = round(min(50.0, l2a_score * settings.L2A_SCORE_MULTIPLIER), 2)
    l2b_contrib = round(confidence * settings.L2B_CONF_MULTIPLIER if label != "normal" else 0.0, 2)

    result = ThreatResult(
        request_id=request_id, decision=decision, score=score,
        label=label, layer="L2B", confidence=round(confidence, 4),
        l2a_score=round(l2a_score, 8),
        l2a_contrib=l2a_contrib,
        l2b_contrib=l2b_contrib,
        latency_ms=ms,
    )

    # persist
    now = datetime.utcnow()
    doc = {**result.model_dump(), "ip": req.ip, "body_len": len(body),
           "method": req.method, "url": url[:500], "timestamp": now}
    if decision in ("allow", "log"):
        doc["body"] = body[:2000]  # needed for health-audit re-scoring (CRC Decision 1)
    await insert_request_log(doc)
    if decision in ("block", "log"):
        await insert_threat_event(doc)

    return result
