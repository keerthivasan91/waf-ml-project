"""app/db/collections.py — typed collection accessors"""
from motor.motor_asyncio import AsyncIOMotorCollection
from app.db.mongodb import get_db

def request_logs()    -> AsyncIOMotorCollection: return get_db()["request_logs"]
def threat_events()   -> AsyncIOMotorCollection: return get_db()["threat_events"]
def feedback_queue()  -> AsyncIOMotorCollection: return get_db()["feedback_queue"]
def model_versions()  -> AsyncIOMotorCollection: return get_db()["model_versions"]
def health_snapshots()-> AsyncIOMotorCollection: return get_db()["health_snapshots"]
def retrain_log()     -> AsyncIOMotorCollection: return get_db()["retrain_log"]