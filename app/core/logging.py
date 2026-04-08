"""app/core/logging.py"""
import logging, sys

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt)
    for lib in ("uvicorn.access", "motor", "pymongo"):
        logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("waf")