"""
Comprehensive Authentication and Authorization Module

Provides JWT authentication, RBAC, session management, and security middleware
for the Telegram ML Bot application with enterprise-grade security features.
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Any
from uuid import UUID, uuid4
from enum import Enum

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.secrets_manager import get_secret
from app.database import get_db_session
from app.models.user import User
import structlog

logger = structlog.get_logger(__name__)


class UserRole(str, Enum):
    """User roles for RBAC system."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    BOT = "bot"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions for fine-grained access control."""
    # User management
    READ_USERS = "read_users"
    WRITE_USERS = "write_users"
    DELETE_USERS = "delete_users"
    
    # System administration
    READ_SYSTEM = "read_system"
    WRITE_SYSTEM = "write_system"
    ADMIN_PANEL = "admin_panel"
    
    # Bot management
    MANAGE_BOT = "manage_bot"
    BOT_COMMANDS = "bot_commands"
    
    # Data access
    READ_ANALYTICS = "read_analytics"
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    
    # Webhooks
    MANAGE_WEBHOOKS = "manage_webhooks"
    RECEIVE_WEBHOOKS = "receive_webhooks"


class SessionData(BaseModel):
    """Session data model."""
    user_id: UUID
    telegram_id: Optional[int] = None
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    two_factor_verified: bool = False
    session_metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthenticatedUser(BaseModel):
    """Authenticated user context."""
    user_id: UUID
    telegram_id: Optional[int] = None
    role: UserRole
    permissions: Set[Permission]
    session_id: str
    is_admin: bool = False
    two_factor_verified: bool = False


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        Permission.READ_USERS, Permission.WRITE_USERS, Permission.DELETE_USERS,
        Permission.READ_SYSTEM, Permission.WRITE_SYSTEM, Permission.ADMIN_PANEL,
        Permission.MANAGE_BOT, Permission.BOT_COMMANDS,
        Permission.READ_ANALYTICS, Permission.EXPORT_DATA, Permission.DELETE_DATA,
        Permission.MANAGE_WEBHOOKS, Permission.RECEIVE_WEBHOOKS
    },
    UserRole.MODERATOR: {
        Permission.READ_USERS, Permission.WRITE_USERS,
        Permission.READ_SYSTEM,
        Permission.MANAGE_BOT, Permission.BOT_COMMANDS,
        Permission.READ_ANALYTICS,
        Permission.RECEIVE_WEBHOOKS
    },
    UserRole.USER: {
        Permission.BOT_COMMANDS
    },
    UserRole.BOT: {
        Permission.RECEIVE_WEBHOOKS, Permission.BOT_COMMANDS
    },
    UserRole.GUEST: set()
}


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_default_headers() -> Dict[str, str]:
        """Get default security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:; "
                "font-src 'self' https:; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-src 'none';"
            ),
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            )
        }


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.session_prefix = "session:"
        self.rate_limit_prefix = "rate_limit:"
        self.failed_attempts_prefix = "failed_attempts:"
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis.url,
                decode_responses=True,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("AuthManager Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AuthManager Redis: {e}")
            raise
    
    async def create_access_token(
        self, 
        user_data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a secure JWT access token."""
        try:
            # Get JWT secret from secure storage
            jwt_secret = await get_secret("JWT_SECRET")
            if not jwt_secret:
                jwt_secret = self.settings.security.jwt_secret
                
            if len(jwt_secret) < 32:
                raise ValueError("JWT secret must be at least 32 characters")
            
            # Set expiration
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(
                    seconds=self.settings.security.jwt_expiration
                )
            
            # Create token payload
            payload = {
                "sub": str(user_data["user_id"]),
                "telegram_id": user_data.get("telegram_id"),
                "role": user_data["role"],
                "session_id": user_data.get("session_id", str(uuid4())),
                "exp": expire,
                "iat": datetime.utcnow(),
                "jti": str(uuid4())  # JWT ID for token blacklisting
            }
            
            # Create token
            token = jwt.encode(
                payload,
                jwt_secret,
                algorithm=self.settings.security.jwt_algorithm
            )
            
            logger.info(
                "Access token created",
                user_id=user_data["user_id"],
                expires_at=expire.isoformat()
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise HTTPException(status_code=500, detail="Token creation failed")
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            # Get JWT secret
            jwt_secret = await get_secret("JWT_SECRET")
            if not jwt_secret:
                jwt_secret = self.settings.security.jwt_secret
            
            # Decode token
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=[self.settings.security.jwt_algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": False,
                    "verify_iss": False
                }
            )
            
            # Validate required claims
            required_claims = ["sub", "role", "exp", "iat"]
            for claim in required_claims:
                if claim not in payload:
                    raise HTTPException(
                        status_code=401, 
                        detail=f"Missing required claim: {claim}"
                    )
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and await self.is_token_blacklisted(jti):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def create_session(
        self, 
        user_id: UUID, 
        role: UserRole,
        request: Request,
        telegram_id: Optional[int] = None
    ) -> str:
        """Create a new user session."""
        try:
            session_id = str(uuid4())
            
            # Get user permissions
            permissions = ROLE_PERMISSIONS.get(role, set())
            
            # Create session data
            session_data = SessionData(
                user_id=user_id,
                telegram_id=telegram_id,
                role=role,
                permissions=permissions,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                device_fingerprint=self._generate_device_fingerprint(request)
            )
            
            # Store session in Redis
            if self.redis_client:
                session_key = f"{self.session_prefix}{session_id}"
                await self.redis_client.setex(
                    session_key,
                    self.settings.security.admin_session_timeout,
                    session_data.model_dump_json()
                )
            
            logger.info(
                "Session created",
                user_id=str(user_id),
                session_id=session_id,
                role=role
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise HTTPException(status_code=500, detail="Session creation failed")
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by session ID."""
        try:
            if not self.redis_client:
                return None
            
            session_key = f"{self.session_prefix}{session_id}"
            session_json = await self.redis_client.get(session_key)
            
            if not session_json:
                return None
            
            session_data = SessionData.model_validate_json(session_json)
            
            # Update last activity
            session_data.last_activity = datetime.utcnow()
            await self.redis_client.setex(
                session_key,
                self.settings.security.admin_session_timeout,
                session_data.model_dump_json()
            )
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        try:
            if not self.redis_client:
                return False
            
            session_key = f"{self.session_prefix}{session_id}"
            result = await self.redis_client.delete(session_key)
            
            logger.info(f"Session invalidated: {session_id}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to invalidate session {session_id}: {e}")
            return False
    
    async def blacklist_token(self, jti: str, exp: datetime) -> bool:
        """Add token to blacklist."""
        try:
            if not self.redis_client:
                return False
            
            # Calculate TTL until token expires
            ttl = int((exp - datetime.utcnow()).total_seconds())
            if ttl > 0:
                await self.redis_client.setex(f"blacklist:{jti}", ttl, "1")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to blacklist token {jti}: {e}")
            return False
    
    async def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        try:
            if not self.redis_client:
                return False
            
            result = await self.redis_client.get(f"blacklist:{jti}")
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to check blacklist for {jti}: {e}")
            return False
    
    async def rate_limit_check(
        self, 
        identifier: str, 
        limit: int, 
        window: int
    ) -> bool:
        """Check if rate limit is exceeded."""
        try:
            if not self.redis_client:
                return False
            
            key = f"{self.rate_limit_prefix}{identifier}"
            current = await self.redis_client.get(key)
            
            if current and int(current) >= limit:
                return True
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            await pipe.execute()
            
            return False
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {identifier}: {e}")
            return False
    
    async def track_failed_attempt(self, identifier: str) -> int:
        """Track failed authentication attempt."""
        try:
            if not self.redis_client:
                return 0
            
            key = f"{self.failed_attempts_prefix}{identifier}"
            count = await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # 1 hour window
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to track attempt for {identifier}: {e}")
            return 0
    
    async def is_blocked(self, identifier: str) -> bool:
        """Check if identifier is blocked due to failed attempts."""
        try:
            if not self.redis_client:
                return False
            
            key = f"{self.failed_attempts_prefix}{identifier}"
            count = await self.redis_client.get(key)
            
            if count and int(count) >= self.settings.security.max_failed_attempts:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check if {identifier} is blocked: {e}")
            return False
    
    def _generate_device_fingerprint(self, request: Request) -> str:
        """Generate device fingerprint from request."""
        try:
            # Combine various request attributes
            fingerprint_data = {
                "user_agent": request.headers.get("user-agent", ""),
                "accept_language": request.headers.get("accept-language", ""),
                "accept_encoding": request.headers.get("accept-encoding", ""),
                "host": request.headers.get("host", "")
            }
            
            # Create hash
            fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
            return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
            
        except Exception:
            return "unknown"
    
    def verify_webhook_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Verify webhook signature."""
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return False


# Global auth manager instance
_auth_manager: Optional[AuthManager] = None


async def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
        await _auth_manager.initialize()
    return _auth_manager


# FastAPI Security dependency
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db_session)
) -> AuthenticatedUser:
    """Get current authenticated user."""
    try:
        auth_manager = await get_auth_manager()
        
        # Verify token
        payload = await auth_manager.verify_token(credentials.credentials)
        
        # Get user data
        user_id = UUID(payload["sub"])
        role = UserRole(payload["role"])
        session_id = payload.get("session_id", "")
        telegram_id = payload.get("telegram_id")
        
        # Validate session if exists
        session_data = None
        if session_id:
            session_data = await auth_manager.get_session(session_id)
            if not session_data:
                raise HTTPException(status_code=401, detail="Session expired")
        
        # Get user permissions
        permissions = ROLE_PERMISSIONS.get(role, set())
        
        # Check if user exists in database
        query = select(User).where(User.id == user_id)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        return AuthenticatedUser(
            user_id=user_id,
            telegram_id=telegram_id,
            role=role,
            permissions=permissions,
            session_id=session_id,
            is_admin=role == UserRole.ADMIN,
            two_factor_verified=session_data.two_factor_verified if session_data else False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(user: AuthenticatedUser = Depends(get_current_user)):
        if permission not in user.permissions:
            raise HTTPException(
                status_code=403, 
                detail=f"Permission required: {permission}"
            )
        return user
    return decorator


def require_role(role: UserRole):
    """Decorator to require specific role."""
    def decorator(user: AuthenticatedUser = Depends(get_current_user)):
        if user.role != role:
            raise HTTPException(
                status_code=403, 
                detail=f"Role required: {role}"
            )
        return user
    return decorator


def require_admin(user: AuthenticatedUser = Depends(get_current_user)):
    """Require admin role."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def verify_telegram_webhook(
    request: Request,
    require_signature: bool = True
) -> bool:
    """Verify Telegram webhook signature."""
    try:
        if not require_signature:
            return True
        
        # Get bot token for signature verification
        bot_token = await get_secret("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            settings = get_settings()
            bot_token = settings.telegram.bot_token
        
        # Get signature from headers
        signature = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not signature:
            logger.warning("Missing Telegram webhook signature")
            return False
        
        # Get request body
        body = await request.body()
        
        # Verify signature
        auth_manager = await get_auth_manager()
        return auth_manager.verify_webhook_signature(body, signature, bot_token)
        
    except Exception as e:
        logger.error(f"Telegram webhook verification failed: {e}")
        return False


# Export main components
__all__ = [
    "AuthManager",
    "UserRole",
    "Permission",
    "AuthenticatedUser",
    "SecurityHeaders",
    "get_auth_manager",
    "get_current_user",
    "require_permission",
    "require_role",
    "require_admin",
    "verify_telegram_webhook"
]
