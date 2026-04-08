"""app/db/mongodb.py — async MongoDB connection using Motor"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings
from app.core.logging import logger

_client: AsyncIOMotorClient = None
_db:     AsyncIOMotorDatabase = None

async def connect_db() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    _db     = _client[settings.MONGO_DB]
    await _client.admin.command("ping")
    await _ensure_indexes()
    logger.info("MongoDB connected → %s", settings.MONGO_DB)

async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")

def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised. Call connect_db() first.")
    return _db

async def _ensure_indexes() -> None:
    db = get_db()
    await db.request_logs.create_index([("timestamp", -1)])
    await db.request_logs.create_index([("decision", 1)])
    await db.request_logs.create_index([("ip", 1)])
    await db.threat_events.create_index([("timestamp", -1)])
    await db.threat_events.create_index([("score", -1)])
    await db.feedback_queue.create_index([("timestamp", -1)])
    await db.feedback_queue.create_index([("verified_label", 1)])
    await db.health_snapshots.create_index([("timestamp", -1)])
    logger.debug("MongoDB indexes ensured")