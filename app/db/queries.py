"""app/db/queries.py — reusable async queries"""
from datetime import datetime, timedelta
from app.db.collections import (request_logs, threat_events,
                             feedback_queue, health_snapshots)

async def insert_request_log(doc: dict) -> None:
    await request_logs().insert_one(doc)

async def insert_threat_event(doc: dict) -> None:
    await threat_events().insert_one(doc)

async def insert_health_snapshot(doc: dict) -> None:
    await health_snapshots().insert_one(doc)

async def get_recent_logs(limit: int = 100, decision_filter: str = None) -> list:
    query = {}
    if decision_filter:
        query["decision"] = decision_filter
    cursor = request_logs().find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

async def get_recent_threats(limit: int = 50) -> list:
    cursor = threat_events().find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

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