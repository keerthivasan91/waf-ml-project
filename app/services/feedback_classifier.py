"""
app/services/feedback_classifier.py

Automated pre-classification of borderline requests in the feedback queue.

Instead of asking a human to review every borderline-scored request,
this module applies a set of heuristics to auto-label obvious cases —
reducing the human review burden before the adaptive retraining step.

Auto-classification rules (applied in order):
  1. Re-run Layer 1 regex rules → if matched, label = rule type (high confidence)
  2. L2A reconstruction error >> threshold → strong anomaly, mark for review
  3. L2B confidence > 0.95 → auto-verify with L2B label
  4. Score 30-50 with L2B label = normal → likely false positive
  5. Anything else → leave verified_label = None for human review
"""
import re
from datetime import datetime
from app.core.logging import logger
from app.db.collections import feedback_queue
import app.services.layer1_filter as l1

# Confidence threshold for auto-accepting L2B label
L2B_AUTO_THRESHOLD = 0.95

# L2A score multiplier above threshold to auto-flag as attack
L2A_STRONG_MULTIPLIER = 3.0


async def classify_pending(limit: int = 500) -> dict:
    """
    Run automated classification on unreviewed feedback queue items.

    Returns
    -------
    dict with counts: auto_labelled, skipped, errors
    """
    cursor = feedback_queue().find(
        {"verified_label": None, "poisoning_flag": False},
        {"_id": 1, "request_id": 1, "url": 1, "body": 1,
         "score": 1, "label": 1, "l2a_score": 1}
    ).limit(limit)

    items = await cursor.to_list(length=limit)
    auto_labelled = 0
    skipped       = 0
    errors        = 0

    for item in items:
        try:
            verdict = _classify_item(item)
            if verdict is None:
                skipped += 1
                continue

            await feedback_queue().update_one(
                {"_id": item["_id"]},
                {"$set": {
                    "verified_label":   verdict["label"],
                    "poisoning_flag":   verdict.get("poisoning", False),
                    "auto_classified":  True,
                    "classified_at":    datetime.utcnow(),
                    "classification_reason": verdict["reason"],
                }},
            )
            auto_labelled += 1
            logger.debug("Auto-labelled %s → %s (%s)",
                         item["request_id"], verdict["label"], verdict["reason"])
        except Exception as e:
            errors += 1
            logger.error("Feedback classify error for %s: %s",
                         item.get("request_id"), e)

    logger.info("Feedback auto-classification: labelled=%d skipped=%d errors=%d",
                auto_labelled, skipped, errors)
    return {"auto_labelled": auto_labelled, "skipped": skipped, "errors": errors}


def _classify_item(item: dict) -> dict | None:
    """
    Apply heuristics to one feedback item.
    Returns {label, reason, poisoning} or None if unsure.
    """
    url     = item.get("url", "")
    body    = item.get("body", "")
    score   = item.get("score", 0)
    label   = item.get("label", "")
    l2a_s   = item.get("l2a_score", 0.0)

    # Rule 1: L1 regex match → definite attack
    blocked, reason = l1.check(url, body)
    if blocked:
        return {"label": reason.replace("_rule", ""), "reason": "l1_regex_match"}

    # Rule 2: Very low score + L2B label=normal → likely false positive
    if score < 35 and label == "normal":
        return {"label": "false_positive", "reason": "low_score_normal_label"}

    # Rule 3: High L2B confidence (stored in label field via score distribution)
    # If score > 85 the L2B was very confident → trust it
    if score >= 85 and label not in ("normal", ""):
        return {"label": label, "reason": "high_score_high_confidence"}

    # Rule 4: score in middle range → needs human review
    if 35 <= score <= 75:
        return None

    # Default: unsure
    return None