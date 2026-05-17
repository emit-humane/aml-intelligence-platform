from .health import router as health_router
from .stream import router as stream_router
from .alerts import router as alerts_router
from .graph import router as graph_router
from .analytics import router as analytics_router

__all__ = [
    "health_router", "stream_router", "alerts_router",
    "graph_router", "analytics_router",
]
