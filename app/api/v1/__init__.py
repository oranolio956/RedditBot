"""API v1 package and router configuration."""

from fastapi import APIRouter

from .users import router as users_router
from .telegram import router as telegram_router
from .sharing import router as sharing_router

# Create main API v1 router
router = APIRouter()

# Include sub-routers
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(telegram_router, prefix="/telegram", tags=["Telegram Bot"])
router.include_router(sharing_router, tags=["Viral Sharing"])

__all__ = ["router"]