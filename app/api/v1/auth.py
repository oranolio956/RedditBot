"""
Authentication API Endpoints

Provides secure login, logout, token refresh, and session management
with comprehensive security features including rate limiting and audit logging.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.models.user import User
from app.core.auth import (
    get_auth_manager, get_current_user, AuthenticatedUser,
    UserRole, Permission
)
from app.core.secrets_manager import get_secret
from app.config.settings import get_settings
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request model."""
    telegram_id: int = Field(..., description="Telegram user ID")
    username: Optional[str] = Field(None, description="Telegram username")
    first_name: Optional[str] = Field(None, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    auth_date: int = Field(..., description="Authentication timestamp")
    hash: str = Field(..., description="Telegram auth hash")
    
    @validator("telegram_id")
    def validate_telegram_id(cls, v):
        if v <= 0:
            raise ValueError("Invalid Telegram ID")
        return v


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    role: UserRole
    permissions: list[Permission]
    session_id: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class LogoutRequest(BaseModel):
    """Logout request model."""
    session_id: Optional[str] = None
    logout_all_sessions: bool = False


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """
    Authenticate user with Telegram auth data and return JWT token.
    
    This endpoint validates Telegram authentication data and creates
    a secure session with proper audit logging.
    """
    try:
        settings = get_settings()
        auth_manager = await get_auth_manager()
        
        # Rate limiting for login attempts
        client_ip = request.client.host if request.client else "unknown"
        is_rate_limited = await auth_manager.rate_limit_check(
            f"login:{client_ip}",
            limit=5,  # 5 attempts per minute
            window=60
        )
        
        if is_rate_limited:
            await auth_manager.track_failed_attempt(f"login:{client_ip}")
            logger.warning(
                "Login rate limit exceeded",
                ip=client_ip,
                telegram_id=login_data.telegram_id
            )
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Please try again later."
            )
        
        # Check if IP is blocked
        if await auth_manager.is_blocked(f"login:{client_ip}"):
            logger.warning(
                "Login attempt from blocked IP",
                ip=client_ip,
                telegram_id=login_data.telegram_id
            )
            raise HTTPException(
                status_code=403,
                detail="Access temporarily blocked due to failed attempts"
            )
        
        # Verify Telegram authentication data
        if not await verify_telegram_auth(login_data):
            await auth_manager.track_failed_attempt(f"login:{client_ip}")
            logger.warning(
                "Invalid Telegram authentication",
                ip=client_ip,
                telegram_id=login_data.telegram_id
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication data"
            )
        
        # Find or create user
        query = select(User).where(User.telegram_id == login_data.telegram_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            user = User(
                telegram_id=login_data.telegram_id,
                username=login_data.username,
                first_name=login_data.first_name,
                last_name=login_data.last_name,
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info(
                "New user created via login",
                user_id=str(user.id),
                telegram_id=login_data.telegram_id
            )
        
        if not user.is_active:
            logger.warning(
                "Login attempt by inactive user",
                user_id=str(user.id),
                telegram_id=login_data.telegram_id
            )
            raise HTTPException(
                status_code=403,
                detail="Account is deactivated"
            )
        
        # Determine user role (implement your logic here)
        user_role = UserRole.ADMIN if user.telegram_id in settings.security.admin_users else UserRole.USER
        
        # Create session
        session_id = await auth_manager.create_session(
            user.id,
            user_role,
            request,
            telegram_id=user.telegram_id
        )
        
        # Create access token
        token_data = {
            "user_id": str(user.id),
            "telegram_id": user.telegram_id,
            "role": user_role,
            "session_id": session_id
        }
        
        access_token = await auth_manager.create_access_token(
            token_data,
            expires_delta=timedelta(seconds=settings.security.jwt_expiration)
        )
        
        # Get user permissions
        from app.core.auth import ROLE_PERMISSIONS
        permissions = list(ROLE_PERMISSIONS.get(user_role, set()))
        
        logger.info(
            "User login successful",
            user_id=str(user.id),
            telegram_id=user.telegram_id,
            role=user_role,
            ip=client_ip,
            session_id=session_id
        )
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.security.jwt_expiration,
            user_id=str(user.id),
            role=user_role,
            permissions=permissions,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Login failed",
            error=str(e),
            telegram_id=login_data.telegram_id,
            ip=request.client.host if request.client else "unknown"
        )
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.post("/logout")
async def logout(
    logout_data: LogoutRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout user and invalidate session(s).
    """
    try:
        auth_manager = await get_auth_manager()
        
        if logout_data.logout_all_sessions:
            # Invalidate all user sessions (implement as needed)
            # For now, just invalidate current session
            session_invalidated = await auth_manager.invalidate_session(
                current_user.session_id
            )
        else:
            # Invalidate specific session
            session_id = logout_data.session_id or current_user.session_id
            session_invalidated = await auth_manager.invalidate_session(session_id)
        
        if session_invalidated:
            logger.info(
                "User logout successful",
                user_id=str(current_user.user_id),
                session_id=current_user.session_id
            )
            return {"message": "Logout successful"}
        else:
            logger.warning(
                "Session invalidation failed",
                user_id=str(current_user.user_id),
                session_id=current_user.session_id
            )
            return {"message": "Logout completed (session may have already expired)"}
        
    except Exception as e:
        logger.error(
            "Logout failed",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get current authenticated user information.
    """
    try:
        # Get full user data from database
        query = select(User).where(User.id == current_user.user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": str(user.id),
            "telegram_id": user.telegram_id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": current_user.role,
            "permissions": list(current_user.permissions),
            "is_admin": current_user.is_admin,
            "created_at": user.created_at.isoformat(),
            "last_active": user.last_activity.isoformat() if user.last_activity else None,
            "session_id": current_user.session_id,
            "two_factor_verified": current_user.two_factor_verified
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get user info",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve user information")


@router.post("/verify-session")
async def verify_session(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify current session validity.
    """
    try:
        auth_manager = await get_auth_manager()
        
        # Get session data
        session_data = await auth_manager.get_session(current_user.session_id)
        
        if not session_data:
            raise HTTPException(status_code=401, detail="Session not found")
        
        return {
            "valid": True,
            "user_id": str(current_user.user_id),
            "role": current_user.role,
            "session_id": current_user.session_id,
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Session verification failed",
            error=str(e),
            user_id=str(current_user.user_id)
        )
        raise HTTPException(status_code=500, detail="Session verification failed")


@router.get("/permissions")
async def get_user_permissions(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user's permissions and capabilities.
    """
    return {
        "role": current_user.role,
        "permissions": list(current_user.permissions),
        "is_admin": current_user.is_admin,
        "capabilities": {
            "can_read_users": Permission.READ_USERS in current_user.permissions,
            "can_write_users": Permission.WRITE_USERS in current_user.permissions,
            "can_delete_users": Permission.DELETE_USERS in current_user.permissions,
            "can_access_admin": Permission.ADMIN_PANEL in current_user.permissions,
            "can_manage_system": Permission.WRITE_SYSTEM in current_user.permissions,
            "can_view_analytics": Permission.READ_ANALYTICS in current_user.permissions
        }
    }


async def verify_telegram_auth(login_data: LoginRequest) -> bool:
    """
    Verify Telegram authentication data using bot token.
    
    Implements the Telegram authentication verification algorithm:
    https://core.telegram.org/widgets/login#checking-authorization
    """
    try:
        # Get bot token
        bot_token = await get_secret("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            settings = get_settings()
            bot_token = settings.telegram.bot_token
        
        # Create data check string
        check_hash = login_data.hash
        data_check_arr = []
        
        # Add all fields except hash
        data_dict = login_data.model_dump(exclude={"hash"})
        for key, value in sorted(data_dict.items()):
            if value is not None:
                data_check_arr.append(f"{key}={value}")
        
        data_check_string = "\n".join(data_check_arr)
        
        # Create secret key from bot token
        secret_key = hashlib.sha256(bot_token.encode()).digest()
        
        # Generate HMAC hash
        import hmac
        computed_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare hashes
        is_valid = hmac.compare_digest(computed_hash, check_hash)
        
        # Check auth_date (should be within last 24 hours for security)
        current_timestamp = int(datetime.utcnow().timestamp())
        auth_age = current_timestamp - login_data.auth_date
        
        if auth_age > 86400:  # 24 hours
            logger.warning(
                "Telegram auth data too old",
                auth_age=auth_age,
                telegram_id=login_data.telegram_id
            )
            return False
        
        if not is_valid:
            logger.warning(
                "Telegram auth hash mismatch",
                telegram_id=login_data.telegram_id,
                expected_hash=computed_hash,
                received_hash=check_hash
            )
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Telegram auth verification failed: {e}")
        return False


# Export router
__all__ = ["router"]
