"""
Rate Limiting Middleware

Implements request rate limiting using Redis to prevent abuse
and ensure fair usage of the API endpoints.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.config import settings
from app.core.redis import redis_manager

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis for distributed rate limiting.
    
    Implements both per-minute and per-hour rate limits with
    different limits for different types of endpoints.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.enabled = settings.security.rate_limit_enabled
        self.per_minute_limit = settings.security.rate_limit_per_minute
        self.per_hour_limit = settings.security.rate_limit_per_hour
        
        # Different limits for different endpoint types
        self.endpoint_limits = {
            "/health": {"per_minute": 120, "per_hour": 7200},  # Health checks - more lenient
            "/metrics": {"per_minute": 60, "per_hour": 3600},  # Metrics - moderate
            "/api/v1/webhook": {"per_minute": 300, "per_hour": 18000},  # Webhook - high limit
            "/api/v1/users": {"per_minute": 30, "per_hour": 1000},  # User API - standard
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        
        if not self.enabled:
            return await call_next(request)
        
        # Get client identifier (IP address or user ID)
        client_id = self._get_client_identifier(request)
        
        # Get rate limits for this endpoint
        per_minute, per_hour = self._get_rate_limits(request.url.path)
        
        try:
            # Check rate limits
            minute_check = await self._check_rate_limit(
                client_id, "minute", per_minute, 60
            )
            hour_check = await self._check_rate_limit(
                client_id, "hour", per_hour, 3600
            )
            
            # If either limit is exceeded, return 429
            if not minute_check["allowed"]:
                return self._create_rate_limit_response(minute_check, "minute")
            
            if not hour_check["allowed"]:
                return self._create_rate_limit_response(hour_check, "hour")
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit-Minute"] = str(per_minute)
            response.headers["X-RateLimit-Remaining-Minute"] = str(minute_check["remaining"])
            response.headers["X-RateLimit-Reset-Minute"] = str(int(time.time()) + minute_check["reset_time"])
            
            response.headers["X-RateLimit-Limit-Hour"] = str(per_hour)
            response.headers["X-RateLimit-Remaining-Hour"] = str(hour_check["remaining"])
            response.headers["X-RateLimit-Reset-Hour"] = str(int(time.time()) + hour_check["reset_time"])
            
            return response
            
        except Exception as e:
            logger.error("Rate limiting error", error=str(e), client_id=client_id)
            # On error, allow the request to proceed
            return await call_next(request)
    
    def _get_client_identifier(self, request: Request) -> str:
        """
        Get unique identifier for the client.
        
        Priority order:
        1. User ID from authentication (if available)
        2. X-Forwarded-For header (for proxies)
        3. Client IP address
        """
        # Check for authenticated user (implement based on your auth system)
        # user_id = getattr(request.state, 'user_id', None)
        # if user_id:
        #     return f"user:{user_id}"
        
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP (client IP)
            client_ip = forwarded_for.split(",")[0].strip()
            return f"ip:{client_ip}"
        
        # Fall back to direct client IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_rate_limits(self, path: str) -> tuple[int, int]:
        """
        Get rate limits for specific endpoint.
        
        Args:
            path: Request path
            
        Returns:
            Tuple of (per_minute_limit, per_hour_limit)
        """
        # Check for specific endpoint limits
        for endpoint_pattern, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return limits["per_minute"], limits["per_hour"]
        
        # Return default limits
        return self.per_minute_limit, self.per_hour_limit
    
    async def _check_rate_limit(
        self,
        identifier: str,
        window_type: str,
        limit: int,
        window_seconds: int
    ) -> dict:
        """
        Check rate limit for identifier and window.
        
        Args:
            identifier: Client identifier
            window_type: Type of window ("minute" or "hour")
            limit: Request limit for this window
            window_seconds: Window duration in seconds
            
        Returns:
            Rate limit check result
        """
        key = f"rate_limit:{identifier}:{window_type}"
        
        try:
            return await redis_manager.rate_limit_check(
                key, limit, window_seconds, increment=True
            )
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e), identifier=identifier)
            # Return permissive result on error
            return {
                "allowed": True,
                "remaining": limit,
                "reset_time": window_seconds,
                "current_count": 0,
                "limit": limit,
            }
    
    def _create_rate_limit_response(self, rate_limit_info: dict, window_type: str) -> Response:
        """
        Create HTTP 429 rate limit exceeded response.
        
        Args:
            rate_limit_info: Rate limit check result
            window_type: Type of window that was exceeded
            
        Returns:
            HTTP 429 response
        """
        retry_after = rate_limit_info["reset_time"]
        
        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + retry_after),
        }
        
        error_detail = (
            f"Rate limit exceeded for {window_type}. "
            f"Limit: {rate_limit_info['limit']} requests per {window_type}. "
            f"Try again in {retry_after} seconds."
        )
        
        content = {
            "error": "Rate limit exceeded",
            "detail": error_detail,
            "retry_after": retry_after,
            "limit": rate_limit_info["limit"],
            "window": window_type,
        }
        
        return Response(
            content=content,
            status_code=429,
            headers=headers,
            media_type="application/json"
        )