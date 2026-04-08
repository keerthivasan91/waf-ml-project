"""app/models/schemas/log.py"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class RequestLog(BaseModel):
    request_id:  str
    ip:          Optional[str]
    method:      str
    url:         str
    body_len:    int
    decision:    str
    score:       int
    label:       str
    layer:       str
    latency_ms:  float
    timestamp:   datetime