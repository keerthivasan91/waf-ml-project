"""
Starlette middleware for the Hybrid Intelligent WAF.

Every application request is intercepted by the WAF, analyzed through:

    L1 → L2A → L2B → Threat Scorer → ALLOW / LOG / BLOCK

Allowed requests are forwarded to the protected FastAPI application.

The WAF analyzes only the clean path + query string, never the absolute URL.
"""

import time
import uuid
from datetime import datetime

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
import app.services.threat_scorer as scorer
from app.services.feature_extractor import extract
from app.db.queries import insert_request_log, insert_threat_event
from app.core.config import settings
from app.core.logging import logger


class WAFMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        # ============================================================
        # INTERCEPT APPLICATION TRAFFIC
        # ============================================================
        #
        # Every request coming to port 8000 is analyzed by the WAF.
        #
        # Examples:
        #   /api/products
        #   /api/products/search?q=laptop
        #   /api/users/profile?user_id=1
        #
        # WAF internal analysis uses only:
        #
        #   /api/products/search?q=laptop
        #
        # ============================================================

        # Do not intercept WAF/dashboard/internal endpoints.
        # These should continue to be handled by the WAF application itself.
        bypass_paths = (
            "/dashboard",
            "/static",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        )

        if request.url.path.startswith(bypass_paths):
            return await call_next(request)

        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        # ============================================================
        # READ REQUEST BODY
        # ============================================================

        raw_body = b""
        body_text = ""

        try:
            raw_body = await request.body()
            body_text = raw_body.decode("utf-8", errors="replace")
        except Exception:
            pass

        # ============================================================
        # CLEAN URL
        # ============================================================
        #
        # IMPORTANT:
        # Only path + query is passed to the ML pipeline.
        #
        # Example:
        #
        # Browser:
        # http://127.0.0.1:8000/api/products/search?q=laptop
        #
        # ML input:
        # /api/products/search?q=laptop
        #
        # ============================================================

        clean_path = request.url.path
        clean_query = f"?{request.url.query}" if request.url.query else ""
        clean_url = clean_path + clean_query

        req_dict = {
            "url": clean_url,
            "method": request.method,
            "headers": dict(request.headers),
            "body": body_text,
            "ip": request.client.host if request.client else None,
        }

        url = req_dict["url"]
        ip = req_dict["ip"]

        # ============================================================
        # LAYER 1 — RULE BASED FILTER
        # ============================================================

        l1_blocked, l1_reason = l1.check(url, body_text)

        if l1_blocked:

            ms = round((time.perf_counter() - t0) * 1000, 2)

            logger.info(
                "WAF L1 BLOCK | url=%s | reason=%s",
                url,
                l1_reason,
            )

            await _log_and_store(
                request_id=request_id,
                ip=ip,
                method=request.method,
                url=url,
                body_len=len(body_text),
                decision="block",
                score=100,
                label=l1_reason,
                layer="L1",
                latency_ms=ms,
                l2a_score=0.0,
                confidence=1.0,
            )

            return JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "reason": l1_reason,
                    "layer": "L1",
                    "request_id": request_id,
                },
            )

        # ============================================================
        # FEATURE EXTRACTION
        # ============================================================

        try:

            fvec, token_ids = extract(req_dict)

            logger.info(
                "FEATURE DEBUG | fvec shape=%s | fvec=%s",
                fvec.shape,
                fvec.tolist(),
            )

            logger.info(
                "TOKEN DEBUG | token_ids shape=%s | first_30=%s",
                token_ids.shape,
                (
                    token_ids[0][:30].tolist()
                    if len(token_ids.shape) > 1
                    else token_ids[:30].tolist()
                ),
            )

        except Exception as e:

            logger.error(
                "Feature extraction failed: %s",
                e,
                exc_info=True,
            )

            # If feature extraction fails, forward the request rather
            # than crashing the protected application.
            return await _forward(request, raw_body)

        # ============================================================
        # LAYER 2A — ANOMALY DETECTION
        # ============================================================

        try:

            l2a_score = l2a.score(fvec)

        except Exception as e:

            logger.error(
                "L2A inference failed: %s",
                e,
                exc_info=True,
            )

            l2a_score = 0.0

        # ============================================================
        # SELECTIVE ESCALATION
        # ============================================================

        if l2a_score < settings.ESCALATION_THRESHOLD:

            ms = round(
                (time.perf_counter() - t0) * 1000,
                2,
            )

            logger.info(
                "WAF DEBUG | url=%s | l2a_score=%.8f | "
                "escalation_threshold=%.8f | "
                "escalated=False | label=normal | confidence=1.0000",
                url,
                l2a_score,
                settings.ESCALATION_THRESHOLD,
            )

            logger.info(
                "WAF DECISION | url=%s | score=%s | decision=%s",
                url,
                0,
                "allow",
            )

            await _log_and_store(
                request_id=request_id,
                ip=ip,
                method=request.method,
                url=url,
                body_len=len(body_text),
                decision="allow",
                score=0,
                label="normal",
                layer="L2A",
                latency_ms=ms,
                l2a_score=l2a_score,
                confidence=1.0,
                body_text=body_text,
            )

            return await _forward(request, raw_body)

        # ============================================================
        # LAYER 2B — DEEP CLASSIFIER
        # ============================================================

        try:

            label, confidence, _ = l2b.infer(
                fvec,
                token_ids,
            )

        except Exception as e:

            logger.error(
                "L2B inference failed: %s",
                e,
                exc_info=True,
            )

            label = "other_attack"
            confidence = 0.5

        # ============================================================
        # THREAT SCORE
        # ============================================================

        score, decision = scorer.compute(
            l2a_score,
            label,
            confidence,
        )

        ms = round(
            (time.perf_counter() - t0) * 1000,
            2,
        )

        logger.info(
            "WAF DEBUG | url=%s | l2a_score=%.8f | "
            "escalated=True | label=%s | confidence=%.4f",
            url,
            l2a_score,
            label,
            confidence,
        )

        logger.info(
            "WAF DECISION | url=%s | score=%s | decision=%s",
            url,
            score,
            decision,
        )

        await _log_and_store(
            request_id=request_id,
            ip=ip,
            method=request.method,
            url=url,
            body_len=len(body_text),
            decision=decision,
            score=score,
            label=label,
            layer="L2B",
            latency_ms=ms,
            l2a_score=l2a_score,
            confidence=confidence,
            body_text=body_text,
        )

        # ============================================================
        # BLOCK
        # ============================================================

        if decision == "block":

            return JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "label": label,
                    "score": score,
                    "layer": "L2B",
                    "request_id": request_id,
                },
            )

        # ============================================================
        # ALLOW / LOG
        # ============================================================

        return await _forward(
            request,
            raw_body,
        )


# ================================================================
# FORWARD REQUEST TO PROTECTED APPLICATION
# ================================================================

async def _forward(
    request: Request,
    raw_body: bytes,
) -> Response:

    """
    Forward an allowed request to the protected FastAPI application.

    Local development setup:

        WAF       → 127.0.0.1:8000
        Backend   → 127.0.0.1:5000
    """

    # IMPORTANT:
    # Do NOT add /proxy.
    #
    # Incoming:
    #   /api/products/search?q=laptop
    #
    # Forwarded:
    #   http://127.0.0.1:5000/api/products/search?q=laptop

    target = settings.PROTECTED_APP_URL + request.url.path

    # Preserve query string.
    if request.url.query:
        target += f"?{request.url.query}"

    try:

        async with httpx.AsyncClient(
            timeout=10.0
        ) as client:

            r = await client.request(
                method=request.method,
                url=target,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower()
                    not in (
                        "host",
                        "content-length",
                    )
                },
                content=raw_body,
            )

        return Response(
            content=r.content,
            status_code=r.status_code,
            headers={
                k: v
                for k, v in r.headers.items()
                if k.lower()
                not in (
                    "content-length",
                    "transfer-encoding",
                    "connection",
                )
            },
            media_type=r.headers.get(
                "content-type"
            ),
        )

    except Exception as e:

        logger.error(
            "Proxy forward failed: %s",
            e,
            exc_info=True,
        )

        return Response(
            content=b"Bad Gateway",
            status_code=502,
        )


# ================================================================
# DATABASE LOGGING
# ================================================================

async def _log_and_store(
    request_id,
    ip,
    method,
    url,
    body_len,
    decision,
    score,
    label,
    layer,
    latency_ms,
    l2a_score=None,
    confidence=None,
    body_text="",
):

    now = datetime.utcnow()

    doc = {
        "request_id": request_id,
        "ip": ip,
        "method": method,
        "url": url[:500],
        "body_len": body_len,
        "decision": decision,
        "score": score,
        "label": label,
        "layer": layer,
        "latency_ms": latency_ms,
        "timestamp": now,
    }

    # ============================================================
    # HEALTH FEEDBACK / RE-AUDIT SUPPORT
    # ============================================================

    if decision in ("allow", "log"):

        doc["body"] = body_text[:2000]

    # ============================================================
    # REQUEST LOG
    # ============================================================

    try:

        await insert_request_log(doc)

    except Exception as e:

        logger.error(
            "Log insert failed: %s",
            e,
            exc_info=True,
        )

    # ============================================================
    # THREAT EVENT
    # ============================================================

    if decision in ("block", "log"):

        threat_doc = {
            **doc,
            "l2a_score": l2a_score,
            "confidence": confidence,
        }

        try:

            await insert_threat_event(
                threat_doc
            )

        except Exception as e:

            logger.error(
                "Threat insert failed: %s",
                e,
                exc_info=True,
            )

    # ============================================================
    # FEEDBACK QUEUE
    # ============================================================

    if decision == "log":

        from app.db.collections import feedback_queue

        feedback_doc = {
            **doc,
            "l2a_score": l2a_score,
            "confidence": confidence,
            "verified_label": None,
            "poisoning_flag": False,
            "auto_classified": False,
        }

        try:

            await feedback_queue().insert_one(
                feedback_doc
            )

        except Exception as e:

            logger.error(
                "Feedback insert failed: %s",
                e,
                exc_info=True,
            )