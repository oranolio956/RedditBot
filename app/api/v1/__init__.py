"""API v1 package and router configuration."""

from fastapi import APIRouter

from .users import router as users_router
from .telegram import router as telegram_router
from .sharing import router as sharing_router
from .consciousness import router as consciousness_router
from .memory_palace import router as memory_palace_router
from .temporal_archaeology import router as temporal_archaeology_router
from .emotional_intelligence import router as emotional_intelligence_router
from .synesthesia import router as synesthesia_router

# Create main API v1 router
router = APIRouter()

# Include sub-routers
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(telegram_router, prefix="/telegram", tags=["Telegram Bot"])
router.include_router(sharing_router, tags=["Viral Sharing"])
router.include_router(consciousness_router, prefix="/consciousness", tags=["Consciousness Mirroring"])
router.include_router(memory_palace_router, prefix="/memory-palace", tags=["Memory Palace"])
router.include_router(temporal_archaeology_router, prefix="/archaeology", tags=["Temporal Archaeology"])
router.include_router(emotional_intelligence_router, prefix="/emotional-intelligence", tags=["Emotional Intelligence"])
router.include_router(synesthesia_router, prefix="/synesthesia", tags=["Digital Synesthesia"])

__all__ = ["router"]