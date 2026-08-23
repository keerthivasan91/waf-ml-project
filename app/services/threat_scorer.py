"""app/services/threat_scorer.py

Threat-score formula — CRC Decision 2 locked config.
Source: fork-of-06-end-to-end-eval.ipynb ("06-end-to-end-eval (2).ipynb"),
cell 39 (FINAL TEST EVALUATION). Selected on validation only, evaluated
once on the untouched test set:

    FPR=0.17%, TPR=88.64%, LFI block rate=45.41%, other_attack/CMDi=76.73%,
    SQLi=99.95%, XSS=98.21%, mean latency=3.84ms, P99=22.89ms

Formula (fixed per accepted paper, no stray floor()):
    s = min(100, c_L2A + c_L2B)
    c_L2A = min(50, l2a_score * L2A_SCORE_MULTIPLIER)
    c_L2B = 0 if label == "normal" else confidence * L2B_CONF_MULTIPLIER

NOTE: whether L2B even runs (the "selective escalation" gate) is decided
in app/middleware/waf_middleware.py using settings.ESCALATION_THRESHOLD,
NOT in this module. This module only computes the score for requests
that already reached L2B.
"""
from app.core.config import settings


def compute(l2a_score: float, label: str, confidence: float):
    """
    Match the locked CRC config exactly.

    Returns
    -------
    score, decision
    """
    l2a_contrib = min(50.0, l2a_score * settings.L2A_SCORE_MULTIPLIER)

    if label == "normal":
        l2b_contrib = 0.0
    else:
        l2b_contrib = confidence * settings.L2B_CONF_MULTIPLIER

    threat_score = min(100, int(l2a_contrib + l2b_contrib))

    if threat_score >= settings.SCORE_BLOCK_THRESHOLD:
        decision = "block"
    elif threat_score >= settings.SCORE_LOG_THRESHOLD:
        decision = "log"
    else:
        decision = "allow"

    return threat_score, decision
