"""app/core/exceptions.py"""
from fastapi import Request
from fastapi.responses import JSONResponse

class ModelNotLoadedError(RuntimeError): pass
class DatabaseError(RuntimeError): pass

async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503,
        content={"error": "ML models not ready", "detail": str(exc)})

async def database_error_handler(request: Request, exc: DatabaseError):
    return JSONResponse(status_code=503,
        content={"error": "Database unavailable", "detail": str(exc)})