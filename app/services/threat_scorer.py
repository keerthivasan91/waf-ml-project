"""app/services/threat_scorer.py"""

def compute(l2a_score: float, label: str, confidence: float):
    """
    Match training pipeline exactly.
    Returns
    -------
    score, decision
    """
    l2a_contrib = min(50.0, l2a_score * 15)

    if label == "normal":
        l2b_contrib = 0.0
    else:
        l2b_contrib = confidence * 50.0

    threat_score = min(100, int(l2a_contrib + l2b_contrib))

    if threat_score >= 70:
        decision = "block"
    elif threat_score >= 30:
        decision = "log"
    else:
        decision = "allow"

    return threat_score, decision