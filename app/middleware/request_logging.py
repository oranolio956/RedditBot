"""
Request Logging Middleware

Comprehensive request/response logging with correlation IDs,
performance tracking, and structured logging for observability.
"""

import time
import json
import uuid
from typing import Callable, Dict, Any, Optional
import logging

import structlog
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging.
    
    Features:
    - Correlation ID generation and tracking
    - Request/response body logging (with sensitive data masking)
    - Performance metrics (latency, throughput)
    - User identification and tracking
    - Error logging with full context
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = True,
        log_response_body: bool = False,
        max_body_length: int = 10000,
        exclude_paths: Optional[list] = None,
        sensitive_fields: Optional[list] = None
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length
        self.exclude_paths = exclude_paths or ['/health', '/metrics', '/docs', '/redoc', '/openapi.json']
        self.sensitive_fields = sensitive_fields or [
            'password', 'token', 'secret', 'api_key', 'authorization',
            'credit_card', 'ssn', 'pin', 'cvv'
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process and log HTTP requests."""
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Prepare request log data
        request_log = {
            'correlation_id': correlation_id,
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client_host': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent'),
            'content_type': request.headers.get('content-type'),
        }
        
        # Extract user info if available
        if hasattr(request.state, 'user'):
            request_log['user_id'] = getattr(request.state.user, 'id', None)
            request_log['user_email'] = getattr(request.state.user, 'email', None)
        
        # Log request body if enabled
        if self.log_request_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    # Store body for later use by the application
                    request._body = body
                    
                    # Parse and mask sensitive data
                    parsed_body = self._parse_body(body, request.headers.get('content-type', ''))
                    masked_body = self._mask_sensitive_data(parsed_body)
                    
                    # Truncate if too long
                    body_str = json.dumps(masked_body) if isinstance(masked_body, dict) else str(masked_body)
                    if len(body_str) > self.max_body_length:
                        body_str = body_str[:self.max_body_length] + '... [truncated]'
                    
                    request_log['body'] = body_str
            except Exception as e:
                logger.warning(f"Failed to log request body: {e}")
        
        # Log incoming request
        logger.info(
            "Incoming request",
            **request_log
        )
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            
            # Capture response body if needed
            if self.log_response_body and response.status_code >= 400:
                response_body = b''
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                # Parse and log response body
                try:
                    parsed_response = json.loads(response_body.decode())
                    masked_response = self._mask_sensitive_data(parsed_response)
                    response_log_body = json.dumps(masked_response)[:self.max_body_length]
                except:
                    response_log_body = response_body.decode()[:self.max_body_length]
                
                # Create new response with captured body
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
                
                request_log['response_body'] = response_log_body
            
        except Exception as e:
            error = e
            logger.error(
                "Request processing error",
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                method=request.method
            )
            raise
        
        finally:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            response_log = {
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code if response else 500,
                'processing_time_ms': round(processing_time * 1000, 2),
                'response_size': len(response.body) if response and hasattr(response, 'body') else 0,
            }
            
            # Add correlation ID to response headers
            if response:
                response.headers['X-Correlation-ID'] = correlation_id
                response.headers['X-Processing-Time'] = str(processing_time)
            
            # Determine log level based on status code
            if response:
                if response.status_code >= 500:
                    logger.error("Request completed with error", **response_log)
                elif response.status_code >= 400:
                    logger.warning("Request completed with client error", **response_log)
                else:
                    logger.info("Request completed successfully", **response_log)
            else:
                logger.error("Request failed", **response_log, error=str(error) if error else "Unknown error")
        
        return response
    
    def _parse_body(self, body: bytes, content_type: str) -> Any:
        """Parse request/response body based on content type."""
        if not body:
            return None
        
        try:
            if 'application/json' in content_type:
                return json.loads(body.decode())
            elif 'application/x-www-form-urlencoded' in content_type:
                from urllib.parse import parse_qs
                return parse_qs(body.decode())
            else:
                # Return as string for other content types
                return body.decode()
        except Exception:
            # If parsing fails, return as string
            try:
                return body.decode()
            except:
                return str(body)
    
    def _mask_sensitive_data(self, data: Any, depth: int = 0) -> Any:
        """Recursively mask sensitive fields in data."""
        if depth > 10:  # Prevent infinite recursion
            return data
        
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                # Check if key contains sensitive field name
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    masked[key] = '***MASKED***'
                else:
                    masked[key] = self._mask_sensitive_data(value, depth + 1)
            return masked
        
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item, depth + 1) for item in data]
        
        else:
            return data


class LoggingRoute(APIRoute):
    """
    Custom APIRoute that logs request/response details.
    
    Alternative to middleware for more granular control.
    """
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Generate correlation ID if not present
            if not hasattr(request.state, 'correlation_id'):
                request.state.correlation_id = str(uuid.uuid4())
            
            # Log request
            logger.info(
                f"API call: {request.method} {request.url.path}",
                correlation_id=request.state.correlation_id,
                method=request.method,
                path=request.url.path,
                query_params=dict(request.query_params)
            )
            
            # Call original handler
            response = await original_route_handler(request)
            
            # Add correlation ID to response
            response.headers['X-Correlation-ID'] = request.state.correlation_id
            
            return response
        
        return custom_route_handler


# Structured logging configuration
def configure_structured_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    log_file: Optional[str] = None
):
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output logs in JSON format
        log_file: Optional log file path
    """
    import sys
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)


# Correlation ID context manager
class CorrelationContext:
    """Context manager for correlation ID tracking."""
    
    _correlation_id: Optional[str] = None
    
    @classmethod
    def set_id(cls, correlation_id: str):
        """Set correlation ID for current context."""
        cls._correlation_id = correlation_id
    
    @classmethod
    def get_id(cls) -> Optional[str]:
        """Get current correlation ID."""
        return cls._correlation_id
    
    @classmethod
    def clear(cls):
        """Clear correlation ID."""
        cls._correlation_id = None


# Export main components
__all__ = [
    'RequestLoggingMiddleware',
    'LoggingRoute',
    'configure_structured_logging',
    'CorrelationContext'
]