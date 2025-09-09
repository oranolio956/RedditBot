"""API v1 package and router configuration."""

from fastapi import APIRouter

from .users import router as users_router
from .webhook import router as webhook_router
from .telegram import router as telegram_router

# Create main API v1 router
router = APIRouter()

# Include sub-routers
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(webhook_router, prefix="/webhook", tags=["Webhook"])
router.include_router(telegram_router, prefix="/telegram", tags=["Telegram Bot"])

__all__ = ["router"]