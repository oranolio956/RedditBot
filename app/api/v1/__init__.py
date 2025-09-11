"""API v1 package and router configuration."""

from fastapi import APIRouter

from .auth import router as auth_router
from .gdpr import router as gdpr_router
from .users import router as users_router
from .telegram import router as telegram_router
from .telegram_auth import router as telegram_auth_router
from .sharing import router as sharing_router
from .consciousness import router as consciousness_router
from .memory_palace import router as memory_palace_router
from .temporal_archaeology import router as temporal_archaeology_router
from .emotional_intelligence import router as emotional_intelligence_router
from .synesthesia import router as synesthesia_router
from .neural_dreams import router as neural_dreams_router
from .meta_reality import router as meta_reality_router
from .transcendence import router as transcendence_router
from .digital_telepathy import router as digital_telepathy_router
from .personality import router as personality_router
from .quantum_consciousness import router as quantum_consciousness_router
from .kelly import router as kelly_router
from .kelly_monitoring import router as kelly_monitoring_router
from .kelly_intervention import router as kelly_intervention_router
from .kelly_alerts import router as kelly_alerts_router
from .kelly_emergency import router as kelly_emergency_router
from .kelly_analytics import router as kelly_analytics_router
from .kelly_intelligence import router as kelly_intelligence_router
from .kelly_crm import router as kelly_crm_router

# Create main API v1 router
router = APIRouter()

# Include sub-routers - Core Features
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
router.include_router(gdpr_router, prefix="/gdpr", tags=["GDPR Compliance"])
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(telegram_router, prefix="/telegram", tags=["Telegram Bot"])
router.include_router(telegram_auth_router, prefix="/telegram", tags=["Telegram Authentication"])
router.include_router(sharing_router, tags=["Viral Sharing"])
router.include_router(kelly_router, prefix="/kelly", tags=["Kelly Brain System"])
router.include_router(kelly_monitoring_router, prefix="/kelly", tags=["Kelly Monitoring"])
router.include_router(kelly_intervention_router, prefix="/kelly", tags=["Kelly Intervention"])
router.include_router(kelly_alerts_router, prefix="/kelly", tags=["Kelly Alerts"])
router.include_router(kelly_emergency_router, prefix="/kelly", tags=["Kelly Emergency"])
router.include_router(kelly_analytics_router, prefix="/kelly/analytics", tags=["Kelly Analytics"])
router.include_router(kelly_intelligence_router, prefix="/kelly/intelligence", tags=["Kelly Intelligence"])
router.include_router(kelly_crm_router, prefix="/kelly/crm", tags=["Kelly CRM"])

# Revolutionary AI Features
router.include_router(consciousness_router, prefix="/consciousness", tags=["Consciousness Mirroring"])
router.include_router(memory_palace_router, prefix="/memory-palace", tags=["Memory Palace"])
router.include_router(temporal_archaeology_router, prefix="/archaeology", tags=["Temporal Archaeology"])
router.include_router(emotional_intelligence_router, prefix="/emotional-intelligence", tags=["Emotional Intelligence"])
router.include_router(synesthesia_router, prefix="/synesthesia", tags=["Digital Synesthesia"])
router.include_router(neural_dreams_router, tags=["Neural Dreams"])
router.include_router(meta_reality_router, prefix="/meta-reality", tags=["Meta Reality Engine"])
router.include_router(transcendence_router, prefix="/transcendence", tags=["Transcendence Protocol"])
router.include_router(digital_telepathy_router, prefix="/digital-telepathy", tags=["Digital Telepathy"])
router.include_router(personality_router, prefix="/personality", tags=["Personality Adaptation"])
router.include_router(quantum_consciousness_router, prefix="/quantum-consciousness", tags=["Quantum Consciousness"])

__all__ = ["router"]