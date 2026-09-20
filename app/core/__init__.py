"""Central app settings export.

The canonical Settings definition lives in app.core.config.  Keep this
module as a re-export so importing app.core never creates a second,
divergent Settings instance.
"""
from app.core.config import Settings, settings

__all__ = ["Settings", "settings"]
