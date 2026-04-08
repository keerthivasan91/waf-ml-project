"""app/middleware/request_parser.py"""
from fastapi import Request

async def parse_request(request: Request) -> dict:
    """Extract url, method, headers, body, ip from FastAPI Request."""
    body = ""
    try:
        raw = await request.body()
        body = raw.decode("utf-8", errors="replace")
    except Exception:
        pass

    return {
        "url":     str(request.url),
        "method":  request.method,
        "headers": dict(request.headers),
        "body":    body,
        "ip":      request.client.host if request.client else None,
    }