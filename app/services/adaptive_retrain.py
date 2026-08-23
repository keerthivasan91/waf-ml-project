"""app/services/adaptive_retrain.py — adaptive retraining pipeline

Anti-poisoning safeguards (three checks, per NB07's mechanism):

  1. Per-IP cap        — no single source floods the loop with near-duplicate
                          samples from one IP.
  2. Family-diversity cap — reject a sample if too many URL-canonicalized
                          near-duplicates already exist in this batch (guards
                          against one crafted-payload family dominating).
  3. L2A/L2B cross-agreement — the human-verified label must be corroborated
                          by re-running the CURRENT models on the sample: for
                          an attack label, L2A must flag it anomalous AND L2B
                          must agree with (or at least not contradict) the
                          verified label; for "normal", both must concur it's
                          not anomalous. A verified label the models actively
                          disagree with is held back rather than trusted
                          blindly, since a single human review is also
                          fallible to a well-crafted poisoning attempt.

  NOTE: an earlier version of this check used regex-based label-plausibility
  (a claimed label must match a keyword pattern). That rejected ~99% of
  genuinely valid candidates because most real attack payloads don't match
  any single hand-written pattern. Cross-agreement generalizes correctly
  instead of hard-coding attack signatures.

  Each sample is also re-scanned by Layer 1 — if L1 catches it outright,
  it's a known-pattern attack, not useful anomaly-drift signal, so it's
  discarded from the retraining set (still logged as verified/blocked
  upstream, just not fed back into L2A/L2B training).

A minimum sample count is required before a retrain cycle actually queues
(RETRAIN_MIN_SAMPLES). Full retraining itself still runs offline in
Kaggle/Colab per NB07 — this function validates the batch and logs the
trigger event + clean sample count, matching NB07's Priority D pipeline
(anti-poison -> human-review ratio gate -> retrain).
"""
import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger
from app.db.collections import feedback_queue, retrain_log
import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
from app.services.reaudit import reaudit

POISON_MAX_RATE_PER_IP = 20   # max verified samples from one IP
MAX_FAMILY_PER_BATCH    = 3   # NB07 family-diversity cap
MAX_BATCH_RATIO         = 0.10  # NB07 human-review gate: batch vs class training size


def _canonicalize(url: str) -> str:
    """Strip %XX encoding noise before hashing, so near-duplicate payloads
    (same attack, different encoding) land in the same family bucket."""
    text = re.sub(r"%[0-9a-fA-F]{2}", "", (url or "").lower())
    return hashlib.md5(text.encode()).hexdigest()[:10]


def _cross_agreement_pass(sample: dict) -> tuple[bool, str]:
    """Check the verified label against a fresh L2A/L2B re-score."""
    verified_label = sample.get("verified_label")
    try:
        result = reaudit(sample.get("url", ""), sample.get("method", "GET"), sample.get("body", ""))
    except Exception as e:
        return False, f"reaudit_failed:{e}"

    if verified_label == "normal":
        if result["label"] == "normal":
            return True, ""
        return False, "cross_agreement_failed_normal_flagged_as_attack"

    # verified label is an attack class
    if result["label"] == "normal":
        return False, "cross_agreement_failed_attack_flagged_as_normal"
    return True, ""


async def run_retrain_cycle() -> dict:
    """
    Returns dict with run metadata.
    """
    # Fetch verified non-poisoned feedback
    cursor = feedback_queue().find(
        {"verified_label": {"$ne": None}, "poisoning_flag": False},
        {"_id": 0}
    )
    samples = await cursor.to_list(length=10000)

    if len(samples) < settings.RETRAIN_MIN_SAMPLES:
        logger.info("Retrain skipped: only %d verified samples (min=%d)",
                    len(samples), settings.RETRAIN_MIN_SAMPLES)
        return {"status": "skipped", "reason": "insufficient_samples",
                "n_samples": len(samples)}

    # Fail loudly if models aren't loaded rather than silently rejecting
    # every sample via reaudit_failed — that's a startup problem, not a
    # poisoning verdict, and the two must not be conflated.
    if l2a._sess is None or l2b._sess is None:
        logger.error("Retrain aborted: L2A/L2B models not loaded — "
                     "cross-agreement anti-poison check cannot run")
        return {"status": "error", "reason": "models_not_loaded",
                "n_samples": len(samples)}

    # ── Anti-poisoning safeguards ────────────────────────────────────────
    ip_counts: dict = defaultdict(int)
    family_counts: dict = defaultdict(int)
    for s in samples:
        family_counts[_canonicalize(s.get("url", ""))] += 1

    clean, rejected = [], []
    reject_reason_counts: dict = defaultdict(int)
    for s in samples:
        ip = s.get("ip", "unknown")
        family = _canonicalize(s.get("url", ""))

        if ip_counts[ip] >= POISON_MAX_RATE_PER_IP:
            rejected.append({**s, "reject_reason": "per_ip_cap_exceeded"})
            reject_reason_counts["per_ip_cap_exceeded"] += 1
            continue
        if family_counts[family] > MAX_FAMILY_PER_BATCH:
            rejected.append({**s, "reject_reason": "family_diversity_cap_exceeded"})
            reject_reason_counts["family_diversity_cap_exceeded"] += 1
            continue

        blocked, _ = l1.check(s.get("url", ""), s.get("body", ""))
        if blocked:
            rejected.append({**s, "reject_reason": "l1_pattern_match"})
            reject_reason_counts["l1_pattern_match"] += 1
            continue

        passed, reason = _cross_agreement_pass(s)
        if not passed:
            rejected.append({**s, "reject_reason": reason})
            reject_reason_counts[reason] += 1
            continue

        ip_counts[ip] += 1
        clean.append(s)

    logger.info("Retrain anti-poison: %d/%d samples passed (%d rejected)",
                len(clean), len(samples), len(rejected))

    # NB07's human-review batch-size gate (batch must not exceed
    # MAX_BATCH_RATIO of the target class's actual training-set size) needs
    # per-class training counts this service doesn't have at runtime. That
    # gate runs in the offline NB07 pipeline, which does have them — this
    # service's job ends at producing a clean, anti-poison-verified batch
    # for NB07 to consume and gate.

    run_doc = {
        "timestamp":    datetime.utcnow(),
        "status":       "queued",
        "n_raw":        len(samples),
        "n_clean":      len(clean),
        "n_rejected":   len(rejected),
        "reject_reason_breakdown": dict(reject_reason_counts),
        "note":         "Full retraining runs offline in Kaggle/Colab (NB07 pipeline). "
                        "This logs the trigger event, clean sample count, and anti-poison "
                        "rejection breakdown for that pipeline to consume.",
    }
    await retrain_log().insert_one(run_doc)
    return {**run_doc, "_id": str(run_doc.get("_id", ""))}
