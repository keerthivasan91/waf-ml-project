"""app/db/queries.py — reusable async queries"""
from datetime import datetime, timedelta
from app.db.collections import (request_logs, threat_events,
                             feedback_queue, health_snapshots,
                             health_audit_log)

# NOTE: Motor's insert_one() mutates the dict you pass it in place,
# injecting a raw (non-JSON-serializable) ObjectId into doc["_id"].
# Every insert_* helper here takes dict(doc) — a shallow copy — so the
# CALLER's original dict is never silently mutated. Skipping this on a
# doc that later gets returned straight through an API response is
# exactly how CRC Decision 1's /api/health/trigger-audit broke: a raw
# ObjectId reached jsonable_encoder and FastAPI 500'd.

async def insert_request_log(doc: dict) -> None:
    await request_logs().insert_one(dict(doc))

async def insert_threat_event(doc: dict) -> None:
    await threat_events().insert_one(dict(doc))

async def insert_health_snapshot(doc: dict) -> None:
    await health_snapshots().insert_one(dict(doc))

async def get_recent_logs(limit: int = 100, decision_filter: str = None) -> list:
    query = {}
    if decision_filter:
        query["decision"] = decision_filter
    cursor = request_logs().find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

async def get_recent_threats(limit: int = 50) -> list:
    cursor = threat_events().find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

async def get_recent_allow_log_traffic(since: datetime, sample_pct: float = 100.0,
                                        limit: int = 5000) -> list:
    """
    CRC Decision 1 / NB08 "capture recent allow/log traffic". Only rows with
    a stored `body` are usable — that's every allow/log row since
    waf_middleware.py started capturing bodies for exactly this purpose.
    `sample_pct` mirrors NB08's CAPTURE_PERCENTAGE (100.0 = capture everything
    eligible; lower values sample, useful if breach-window volume is huge).
    """
    query = {"timestamp": {"$gte": since}, "decision": {"$in": ["allow", "log"]},
              "body": {"$exists": True}}
    cursor = request_logs().find(query, {"_id": 0}).limit(limit)
    rows = await cursor.to_list(length=limit)
    if sample_pct >= 100.0 or not rows:
        return rows
    import random
    k = max(1, int(len(rows) * sample_pct / 100.0))
    return random.sample(rows, k)


async def insert_health_audit(doc: dict) -> None:
    await health_audit_log().insert_one(dict(doc))


async def get_pending_feedback(limit: int = 200) -> list:
    cursor = feedback_queue().find(
        {"verified_label": None, "poisoning_flag": False},
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

async def get_dashboard_stats() -> dict:
    now  = datetime.utcnow()
    h24  = now - timedelta(hours=24)
    h1   = now - timedelta(hours=1)
    coll = request_logs()

    total_24h   = await coll.count_documents({"timestamp": {"$gte": h24}})
    blocked_24h = await coll.count_documents({"timestamp": {"$gte": h24}, "decision": "block"})
    allowed_24h = await coll.count_documents({"timestamp": {"$gte": h24}, "decision": "allow"})
    total_1h    = await coll.count_documents({"timestamp": {"$gte": h1}})
    blocked_1h  = await coll.count_documents({"timestamp": {"$gte": h1},  "decision": "block"})

    pipeline = [
        {"$match": {"timestamp": {"$gte": h24}, "decision": "block"}},
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
    ]
    attack_breakdown = {d["_id"]: d["count"]
                        async for d in threat_events().aggregate(pipeline)}

    pipeline_lat = [
        {"$match": {"timestamp": {"$gte": h1}}},
        {"$group": {"_id": None, "avg": {"$avg": "$latency_ms"},
                    "p99": {"$percentile": {"input": "$latency_ms",
                                            "p": [0.99], "method": "approximate"}}}},
    ]
    lat_result = await coll.aggregate(pipeline_lat).to_list(1)
    avg_latency = round(lat_result[0]["avg"], 2) if lat_result else 0
    p99_latency = round(lat_result[0]["p99"][0], 2) if lat_result else 0

    return {
        "total_24h":       total_24h,
        "blocked_24h":     blocked_24h,
        "allowed_24h":     allowed_24h,
        "block_rate_24h":  round(blocked_24h / total_24h * 100, 2) if total_24h else 0,
        "total_1h":        total_1h,
        "blocked_1h":      blocked_1h,
        "attack_breakdown": attack_breakdown,
        "avg_latency_ms":  avg_latency,
        "p99_latency_ms":  p99_latency,
    }