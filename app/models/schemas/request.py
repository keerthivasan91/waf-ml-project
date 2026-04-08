"""app/models/schemas/request.py"""
from pydantic import BaseModel
from typing import Optional

class IncomingRequest(BaseModel):
    url:     str
    method:  str = "GET"
    headers: dict = {}
    body:    str  = ""
    ip:      Optional[str] = None