"""app/models/schemas/threat.py"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ThreatResult(BaseModel):
    request_id:   str
    decision:     str           # allow | log | block
    score:        int           # 0-100
    label:        str           # normal | sqli | xss | lfi | other_attack | *_rule
    layer:        str           # L1 | L2A | L2B
    confidence:   Optional[float] = None
    l2a_score:    Optional[float] = None
    l2a_contrib:  Optional[float] = None
    l2b_contrib:  Optional[float] = None
    latency_ms:   float
    timestamp:    datetime = None

    def model_post_init(self, __context):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()