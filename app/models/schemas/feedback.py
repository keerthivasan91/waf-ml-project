"""app/models/schemas/feedback.py"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FeedbackItem(BaseModel):
    request_id:     str
    url:            str
    method:         str
    body:           str
    score:          int
    label:          str
    l2a_score:      float
    verified_label: Optional[str] = None   # set by human reviewer
    poisoning_flag: bool = False
    timestamp:      datetime