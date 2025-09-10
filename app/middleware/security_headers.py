"""
Security Headers Middleware

Adds essential security headers to all HTTP responses to protect against
common web vulnerabilities:
- XSS attacks
- Clickjacking
- MIME type confusion
- Information leakage
- CSRF attacks
"""

from typing import Callable, Dict, Any
from datetime import datetime, timedelta

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

logger = structlog.get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds comprehensive security headers to all responses.
    
    Headers added:
    - X-Frame-Options: Prevents clickjacking
    - X-Content-Type-Options: Prevents MIME sniffing
    - X-XSS-Protection: XSS protection for older browsers
    - Strict-Transport-Security: Forces HTTPS
    - Content-Security-Policy: Prevents XSS and data injection
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    - X-Permitted-Cross-Domain-Policies: Adobe Flash/PDF protection
    - X-Download-Options: IE download security
    - Cross-Origin-Embedder-Policy: Isolates browsing context
    - Cross-Origin-Opener-Policy: Prevents cross-origin attacks
    - Cross-Origin-Resource-Policy: Controls resource sharing
    """
    
    def __init__(
        self, 
        app,
        hsts_max_age: int = 31536000,  # 1 year
        enable_hsts: bool = True,
        csp_policy: str = None,
        enable_debug_headers: bool = False
    ):
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.enable_hsts = enable_hsts
        self.enable_debug_headers = enable_debug_headers
        self.csp_policy = csp_policy or self._default_csp_policy()
    
    def _default_csp_policy(self) -> str:
        """
        Generate a strict default Content Security Policy.
        
        This policy is very restrictive and may need adjustment
        based on your application's specific needs.
        """
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://js.stripe.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' "
            "https://api.openai.com https://api.anthropic.com "
            "https://api.stripe.com wss:; "
            "media-src 'self' blob:; "
            "object-src 'none'; "
            "frame-src 'self' https://js.stripe.com; "
            "worker-src 'self' blob:; "
            "child-src 'self' blob:; "
            "form-action 'self'; "
            "frame-ancestors 'self'; "
            "manifest-src 'self'; "
            "base-uri 'self'"
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        headers = self._get_security_headers(request)
        
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        
        # Add debug headers if enabled
        if self.enable_debug_headers:
            response.headers["X-Security-Headers-Applied"] = str(datetime.utcnow().isoformat())
            response.headers["X-Request-ID"] = getattr(request.state, 'correlation_id', 'unknown')
        
        return response
    
    def _get_security_headers(self, request: Request) -> Dict[str, str]:
        """Generate all security headers."""
        headers = {}
        
        # Prevent clickjacking
        headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type confusion
        headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy but still useful)
        headers["X-XSS-Protection"] = "1; mode=block"
        
        # Force HTTPS (only if enabled and on HTTPS)
        if self.enable_hsts and self._is_https_request(request):
            headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )
        
        # Content Security Policy
        headers["Content-Security-Policy"] = self.csp_policy
        
        # Referrer policy - limit referrer information leakage
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (formerly Feature Policy)
        # Disable potentially dangerous browser features
        headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=(), "
            "interest-cohort=()"  # Disable FLoC
        )
        
        # Adobe Flash/PDF security
        headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # IE download security
        headers["X-Download-Options"] = "noopen"
        
        # Cross-Origin policies for better isolation
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Server information hiding
        headers["Server"] = "AI-Conversation-Bot"
        
        # Cache control for sensitive endpoints
        if self._is_sensitive_endpoint(request):
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
            headers["Pragma"] = "no-cache"
            headers["Expires"] = "0"
        
        return headers
    
    def _is_https_request(self, request: Request) -> bool:
        """Check if request is over HTTPS."""
        return (
            request.url.scheme == "https" or
            request.headers.get("x-forwarded-proto") == "https" or
            request.headers.get("x-forwarded-ssl") == "on"
        )
    
    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Check if endpoint handles sensitive data."""
        sensitive_paths = [
            "/api/v1/auth/",
            "/api/v1/users/",
            "/api/v1/payments/",
            "/api/v1/admin/",
            "/webhook",
            "/health"  # Health endpoints can leak system info
        ]
        
        path = request.url.path
        return any(path.startswith(sensitive) for sensitive in sensitive_paths)


class SecurityEventLogger:
    """Log security-related events for monitoring."""
    
    def __init__(self):
        self.logger = structlog.get_logger("security")
    
    async def log_security_violation(
        self, 
        request: Request, 
        violation_type: str, 
        details: Dict[str, Any]
    ):
        """Log security violations for monitoring."""
        self.logger.warning(
            "Security violation detected",
            violation_type=violation_type,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            path=str(request.url.path),
            method=request.method,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def log_blocked_request(
        self, 
        request: Request, 
        reason: str,
        rule_triggered: str = None
    ):
        """Log blocked requests."""
        self.logger.error(
            "Request blocked by security policy",
            reason=reason,
            rule_triggered=rule_triggered,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            path=str(request.url.path),
            method=request.method,
            timestamp=datetime.utcnow().isoformat()
        )


class RateLimitSecurityMiddleware(BaseHTTPMiddleware):
    """
    Additional rate limiting middleware focused on security.
    
    This complements the main rate limiting and focuses on
    preventing brute force attacks and abuse.
    """
    
    def __init__(
        self,
        app,
        suspicious_threshold: int = 100,  # requests per minute
        block_duration: int = 3600,  # 1 hour
        enable_progressive_delays: bool = True
    ):
        super().__init__(app)
        self.suspicious_threshold = suspicious_threshold
        self.block_duration = block_duration
        self.enable_progressive_delays = enable_progressive_delays
        self.request_counts = {}
        self.blocked_ips = {}
        self.security_logger = SecurityEventLogger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply security-focused rate limiting."""
        client_ip = self._get_client_ip(request)
        
        # Check if IP is currently blocked
        if await self._is_ip_blocked(client_ip):
            await self.security_logger.log_blocked_request(
                request, 
                "IP blocked due to suspicious activity"
            )
            return Response("Too Many Requests", status_code=429)
        
        # Track request count
        await self._track_request(client_ip, request)
        
        # Check if IP should be blocked
        if await self._should_block_ip(client_ip, request):
            await self._block_ip(client_ip, request)
            return Response("Too Many Requests", status_code=429)
        
        # Add progressive delay for suspicious IPs
        if self.enable_progressive_delays:
            delay = await self._calculate_progressive_delay(client_ip)
            if delay > 0:
                import asyncio
                await asyncio.sleep(delay)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies."""
        # Check common proxy headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _track_request(self, client_ip: str, request: Request):
        """Track request count for IP."""
        now = datetime.utcnow()
        minute_key = now.strftime("%Y-%m-%d-%H-%M")
        ip_minute_key = f"{client_ip}:{minute_key}"
        
        if ip_minute_key not in self.request_counts:
            self.request_counts[ip_minute_key] = 0
        
        self.request_counts[ip_minute_key] += 1
        
        # Clean old entries (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        old_keys = [
            key for key in self.request_counts.keys()
            if datetime.strptime(key.split(":", 1)[1], "%Y-%m-%d-%H-%M") < cutoff
        ]
        for key in old_keys:
            del self.request_counts[key]
    
    async def _should_block_ip(self, client_ip: str, request: Request) -> bool:
        """Check if IP should be blocked."""
        now = datetime.utcnow()
        minute_key = now.strftime("%Y-%m-%d-%H-%M")
        ip_minute_key = f"{client_ip}:{minute_key}"
        
        current_count = self.request_counts.get(ip_minute_key, 0)
        
        if current_count > self.suspicious_threshold:
            await self.security_logger.log_security_violation(
                request,
                "rate_limit_exceeded",
                {
                    "requests_per_minute": current_count,
                    "threshold": self.suspicious_threshold,
                    "client_ip": client_ip
                }
            )
            return True
        
        return False
    
    async def _block_ip(self, client_ip: str, request: Request):
        """Block IP for specified duration."""
        block_until = datetime.utcnow() + timedelta(seconds=self.block_duration)
        self.blocked_ips[client_ip] = block_until
        
        await self.security_logger.log_blocked_request(
            request,
            f"IP blocked for {self.block_duration} seconds due to rate limit violation",
            "suspicious_activity_threshold"
        )
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        if client_ip not in self.blocked_ips:
            return False
        
        block_until = self.blocked_ips[client_ip]
        if datetime.utcnow() > block_until:
            del self.blocked_ips[client_ip]
            return False
        
        return True
    
    async def _calculate_progressive_delay(self, client_ip: str) -> float:
        """Calculate progressive delay based on request frequency."""
        now = datetime.utcnow()
        minute_key = now.strftime("%Y-%m-%d-%H-%M")
        ip_minute_key = f"{client_ip}:{minute_key}"
        
        current_count = self.request_counts.get(ip_minute_key, 0)
        
        # Add delay if approaching threshold
        threshold_ratio = current_count / self.suspicious_threshold
        
        if threshold_ratio > 0.7:  # 70% of threshold
            # Progressive delay from 0.1 to 2 seconds
            return min(2.0, (threshold_ratio - 0.7) * 6.67)
        
        return 0.0


# Export components
__all__ = [
    'SecurityHeadersMiddleware',
    'RateLimitSecurityMiddleware', 
    'SecurityEventLogger'
]