"""
Centralized Security Module

Provides secure JWT verification and authentication utilities
to prevent algorithm substitution attacks and ensure consistent security.
"""

import jwt
from typing import Dict, Optional, Union
from uuid import UUID
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import get_settings
import structlog

logger = structlog.get_logger(__name__)

security = HTTPBearer()


def verify_token(token: str) -> Dict[str, any]:
    """
    Securely verify JWT token with explicit algorithm validation.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload dictionary
        
    Raises:
        HTTPException: If token is invalid or verification fails
    """
    try:
        # Explicitly validate algorithm to prevent algorithm substitution attacks
        settings = get_settings()
        allowed_algorithms = ['HS256', 'RS256']
        
        if not hasattr(settings.security, 'jwt_algorithm') or settings.security.jwt_algorithm not in allowed_algorithms:
            logger.error("Invalid JWT algorithm configuration", algorithm=getattr(settings, 'JWT_ALGORITHM', None))
            raise HTTPException(status_code=500, detail="Invalid JWT algorithm configuration")
        
        if not hasattr(settings.security, 'jwt_secret') or not settings.security.jwt_secret:
            logger.error("JWT secret key not configured")
            raise HTTPException(status_code=500, detail="JWT configuration error")
        
        # Decode with strict validation
        payload = jwt.decode(
            token,
            settings.security.jwt_secret,
            algorithms=[settings.security.jwt_algorithm],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": False,
                "verify_iss": False
            }
        )
        
        # Validate required claims
        if 'sub' not in payload:
            logger.warning("JWT token missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token claims")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        logger.error("Unexpected error during JWT verification", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


async def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> UUID:
    """
    Verify JWT token from HTTP Authorization header and return user_id.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        User ID as UUID
        
    Raises:
        HTTPException: If authentication fails
    """
    payload = verify_token(credentials.credentials)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    try:
        return UUID(user_id)
    except ValueError:
        logger.warning("Invalid user ID format in JWT", user_id=user_id)
        raise HTTPException(status_code=401, detail="Invalid user identifier")


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """
    Create a secure JWT access token.
    
    Args:
        data: Payload data to encode
        expires_delta: Token expiration in seconds (optional)
        
    Returns:
        Encoded JWT token
    """
    from datetime import datetime, timedelta
    
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + timedelta(seconds=expires_delta)
    else:
        expire = datetime.utcnow() + timedelta(
            seconds=getattr(settings.security, 'jwt_expiration', 3600)
        )
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    return jwt.encode(
        to_encode,
        settings.security.jwt_secret,
        algorithm=settings.security.jwt_algorithm
    )


def validate_jwt_configuration() -> bool:
    """
    Validate JWT configuration on startup.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    settings = get_settings()
    
    if not hasattr(settings.security, 'jwt_secret') or not settings.security.jwt_secret:
        raise ValueError("JWT_SECRET_KEY must be configured")
    
    if len(settings.security.jwt_secret) < 32:
        raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
    
    allowed_algorithms = ['HS256', 'RS256']
    if not hasattr(settings.security, 'jwt_algorithm') or settings.security.jwt_algorithm not in allowed_algorithms:
        raise ValueError(f"JWT_ALGORITHM must be one of: {allowed_algorithms}")
    
    return True