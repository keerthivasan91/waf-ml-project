"""app/services/reaudit.py — shared re-scoring pipeline

Used by the server-health feedback loop (CRC Decision 1 / NB08) to
re-score previously allowed/logged traffic against CURRENT thresholds
and models. Mirrors app/middleware/waf_middleware.py's decision logic
exactly (L1 -> escalation gate -> L2A -> L2B -> threat score) so a
"disagreement" genuinely reflects a change in what the live pipeline
would decide today, not a divergent implementation.

NOT used on the live request path — that stays in waf_middleware.py for
latency-critical inline scoring. This module is for offline/audit re-scoring
of already-logged traffic.
"""
import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
import app.services.threat_scorer as scorer
from app.services.feature_extractor import extract
from app.core.config import settings


def reaudit(url: str, method: str, body: str) -> dict:
    """
    Re-run the full WAF decision pipeline on a previously-seen request.

    Returns
    -------
    dict with keys: decision, label, confidence, l2a_score
    """
    l1_blocked, l1_reason = l1.check(url, body)
    if l1_blocked:
        return {"decision": "block", "label": l1_reason, "confidence": 1.0, "l2a_score": None}

    req_dict = {"url": url, "method": method, "headers": {}, "body": body}
    fvec, token_ids = extract(req_dict)

    l2a_score = l2a.score(fvec)

    # Selective escalation gate — identical to the live middleware
    if l2a_score < settings.ESCALATION_THRESHOLD:
        return {"decision": "allow", "label": "normal", "confidence": 1.0, "l2a_score": l2a_score}

    label, confidence, _ = l2b.infer(fvec, token_ids)
    score, decision = scorer.compute(l2a_score, label, confidence)

    return {"decision": decision, "label": label, "confidence": confidence, "l2a_score": l2a_score}
