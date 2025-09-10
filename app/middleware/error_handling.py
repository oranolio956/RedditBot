"""
Error Handling Middleware

Comprehensive error handling with recovery strategies, circuit breakers,
and graceful degradation for production resilience.
"""

import time
import traceback
from typing import Dict, Any, Optional, Callable, Type
from datetime import datetime, timedelta
import asyncio
from enum import Enum

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

logger = structlog.get_logger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for handling strategies."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    INTERNAL = "internal"
    TIMEOUT = "timeout"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware with recovery strategies.
    
    Features:
    - Structured error responses
    - Circuit breaker pattern
    - Graceful degradation
    - Error tracking and metrics
    - Recovery strategies
    - User-friendly error messages
    """
    
    def __init__(
        self,
        app,
        debug: bool = False,
        include_traceback: bool = False,
        circuit_breaker_enabled: bool = True,
        custom_handlers: Optional[Dict[Type[Exception], Callable]] = None
    ):
        super().__init__(app)
        self.debug = debug
        self.include_traceback = include_traceback
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.custom_handlers = custom_handlers or {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_path': {},
            'last_error_time': None
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error handling."""
        try:
            # Check circuit breaker for the path
            if self.circuit_breaker_enabled:
                breaker = self._get_circuit_breaker(request.url.path)
                if not breaker.is_closed():
                    return self._circuit_breaker_response(request)
            
            # Process request
            response = await call_next(request)
            
            # Reset circuit breaker on success
            if self.circuit_breaker_enabled and response.status_code < 500:
                breaker = self._get_circuit_breaker(request.url.path)
                breaker.record_success()
            
            return response
            
        except Exception as exc:
            # Record error
            self._record_error(request, exc)
            
            # Get correlation ID if available
            correlation_id = getattr(request.state, 'correlation_id', None)
            
            # Handle specific exception types
            if isinstance(exc, RequestValidationError):
                return await self._handle_validation_error(exc, request, correlation_id)
            
            elif isinstance(exc, (HTTPException, StarletteHTTPException)):
                return await self._handle_http_exception(exc, request, correlation_id)
            
            elif isinstance(exc, ValidationError):
                return await self._handle_pydantic_validation_error(exc, request, correlation_id)
            
            # Check for custom handlers
            for exc_type, handler in self.custom_handlers.items():
                if isinstance(exc, exc_type):
                    return await handler(exc, request, correlation_id)
            
            # Handle database errors
            if self._is_database_error(exc):
                return await self._handle_database_error(exc, request, correlation_id)
            
            # Handle external service errors
            if self._is_external_service_error(exc):
                return await self._handle_external_service_error(exc, request, correlation_id)
            
            # Handle timeout errors
            if isinstance(exc, asyncio.TimeoutError):
                return await self._handle_timeout_error(exc, request, correlation_id)
            
            # Default internal server error
            return await self._handle_internal_error(exc, request, correlation_id)
    
    async def _handle_validation_error(
        self,
        exc: RequestValidationError,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle request validation errors."""
        errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            errors.append({
                'field': field_path,
                'message': error['msg'],
                'type': error['type']
            })
        
        logger.warning(
            "Validation error",
            correlation_id=correlation_id,
            path=request.url.path,
            errors=errors
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': 'The request contains invalid data',
                    'category': ErrorCategory.VALIDATION,
                    'details': errors,
                    'correlation_id': correlation_id
                }
            }
        )
    
    async def _handle_http_exception(
        self,
        exc: HTTPException,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        # Determine error category
        category = self._categorize_http_error(exc.status_code)
        
        # Log appropriately based on status code
        if exc.status_code >= 500:
            logger.error(
                "HTTP exception",
                correlation_id=correlation_id,
                path=request.url.path,
                status_code=exc.status_code,
                detail=exc.detail
            )
        else:
            logger.info(
                "HTTP exception",
                correlation_id=correlation_id,
                path=request.url.path,
                status_code=exc.status_code,
                detail=exc.detail
            )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                'error': {
                    'code': f'HTTP_{exc.status_code}',
                    'message': exc.detail or self._get_default_message(exc.status_code),
                    'category': category,
                    'correlation_id': correlation_id
                }
            }
        )
    
    async def _handle_pydantic_validation_error(
        self,
        exc: ValidationError,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        errors = []
        for error in exc.errors():
            errors.append({
                'field': ' -> '.join(str(loc) for loc in error['loc']),
                'message': error['msg'],
                'type': error['type']
            })
        
        logger.warning(
            "Pydantic validation error",
            correlation_id=correlation_id,
            path=request.url.path,
            errors=errors
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'error': {
                    'code': 'DATA_VALIDATION_ERROR',
                    'message': 'Data validation failed',
                    'category': ErrorCategory.VALIDATION,
                    'details': errors,
                    'correlation_id': correlation_id
                }
            }
        )
    
    async def _handle_database_error(
        self,
        exc: Exception,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle database errors with circuit breaker."""
        logger.error(
            "Database error",
            correlation_id=correlation_id,
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__
        )
        
        # Trip circuit breaker for database errors
        if self.circuit_breaker_enabled:
            breaker = self._get_circuit_breaker('database')
            breaker.record_failure()
        
        # User-friendly message
        message = "A database error occurred. Please try again later."
        if self.debug:
            message = f"Database error: {str(exc)}"
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'error': {
                    'code': 'DATABASE_ERROR',
                    'message': message,
                    'category': ErrorCategory.DATABASE,
                    'correlation_id': correlation_id,
                    'retry_after': 30  # Suggest retry after 30 seconds
                }
            }
        )
    
    async def _handle_external_service_error(
        self,
        exc: Exception,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle external service errors with graceful degradation."""
        service_name = self._extract_service_name(exc)
        
        logger.error(
            "External service error",
            correlation_id=correlation_id,
            path=request.url.path,
            service=service_name,
            error=str(exc)
        )
        
        # Trip circuit breaker for specific service
        if self.circuit_breaker_enabled:
            breaker = self._get_circuit_breaker(f'service_{service_name}')
            breaker.record_failure()
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'error': {
                    'code': 'EXTERNAL_SERVICE_ERROR',
                    'message': f"The {service_name} service is temporarily unavailable",
                    'category': ErrorCategory.EXTERNAL_SERVICE,
                    'correlation_id': correlation_id,
                    'retry_after': 60
                }
            }
        )
    
    async def _handle_timeout_error(
        self,
        exc: asyncio.TimeoutError,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle timeout errors."""
        logger.error(
            "Request timeout",
            correlation_id=correlation_id,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                'error': {
                    'code': 'REQUEST_TIMEOUT',
                    'message': 'The request took too long to process',
                    'category': ErrorCategory.TIMEOUT,
                    'correlation_id': correlation_id
                }
            }
        )
    
    async def _handle_internal_error(
        self,
        exc: Exception,
        request: Request,
        correlation_id: Optional[str]
    ) -> JSONResponse:
        """Handle internal server errors."""
        # Log full error details
        logger.error(
            "Internal server error",
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc() if self.include_traceback else None
        )
        
        # Prepare error response
        error_response = {
            'code': 'INTERNAL_SERVER_ERROR',
            'message': 'An unexpected error occurred',
            'category': ErrorCategory.INTERNAL,
            'correlation_id': correlation_id
        }
        
        # Include error details in debug mode
        if self.debug:
            error_response['message'] = str(exc)
            error_response['type'] = type(exc).__name__
            if self.include_traceback:
                error_response['traceback'] = traceback.format_exc().split('\n')
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': error_response}
        )
    
    def _get_circuit_breaker(self, identifier: str) -> 'CircuitBreaker':
        """Get or create circuit breaker for identifier."""
        if identifier not in self.circuit_breakers:
            self.circuit_breakers[identifier] = CircuitBreaker(
                identifier=identifier,
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
        return self.circuit_breakers[identifier]
    
    def _circuit_breaker_response(self, request: Request) -> JSONResponse:
        """Return circuit breaker open response."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'error': {
                    'code': 'CIRCUIT_BREAKER_OPEN',
                    'message': 'Service temporarily unavailable due to high error rate',
                    'category': ErrorCategory.INTERNAL,
                    'retry_after': 60
                }
            }
        )
    
    def _record_error(self, request: Request, exc: Exception):
        """Record error statistics."""
        self.error_stats['total_errors'] += 1
        self.error_stats['last_error_time'] = datetime.utcnow()
        
        # Record by category
        category = self._categorize_error(exc)
        if category not in self.error_stats['errors_by_category']:
            self.error_stats['errors_by_category'][category] = 0
        self.error_stats['errors_by_category'][category] += 1
        
        # Record by path
        path = request.url.path
        if path not in self.error_stats['errors_by_path']:
            self.error_stats['errors_by_path'][path] = 0
        self.error_stats['errors_by_path'][path] += 1
    
    def _categorize_error(self, exc: Exception) -> str:
        """Categorize error for statistics."""
        if isinstance(exc, (RequestValidationError, ValidationError)):
            return ErrorCategory.VALIDATION
        elif isinstance(exc, HTTPException):
            return self._categorize_http_error(exc.status_code)
        elif self._is_database_error(exc):
            return ErrorCategory.DATABASE
        elif self._is_external_service_error(exc):
            return ErrorCategory.EXTERNAL_SERVICE
        elif isinstance(exc, asyncio.TimeoutError):
            return ErrorCategory.TIMEOUT
        else:
            return ErrorCategory.INTERNAL
    
    def _categorize_http_error(self, status_code: int) -> str:
        """Categorize HTTP error by status code."""
        if status_code == 401:
            return ErrorCategory.AUTHENTICATION
        elif status_code == 403:
            return ErrorCategory.AUTHORIZATION
        elif status_code == 404:
            return ErrorCategory.NOT_FOUND
        elif status_code == 429:
            return ErrorCategory.RATE_LIMIT
        elif 400 <= status_code < 500:
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.INTERNAL
    
    def _is_database_error(self, exc: Exception) -> bool:
        """Check if exception is database-related."""
        db_error_types = [
            'OperationalError',
            'IntegrityError',
            'DataError',
            'DatabaseError',
            'InterfaceError',
            'ProgrammingError'
        ]
        return any(error_type in type(exc).__name__ for error_type in db_error_types)
    
    def _is_external_service_error(self, exc: Exception) -> bool:
        """Check if exception is from external service."""
        service_error_indicators = [
            'ConnectionError',
            'TimeoutError',
            'HTTPError',
            'RequestException',
            'APIError',
            'ServiceError'
        ]
        return any(indicator in type(exc).__name__ for indicator in service_error_indicators)
    
    def _extract_service_name(self, exc: Exception) -> str:
        """Extract service name from exception."""
        # Try to extract from exception message or type
        exc_str = str(exc).lower()
        services = ['telegram', 'openai', 'anthropic', 'stripe', 'redis', 'database']
        
        for service in services:
            if service in exc_str:
                return service.capitalize()
        
        return 'external'
    
    def _get_default_message(self, status_code: int) -> str:
        """Get default message for status code."""
        messages = {
            400: "Bad request",
            401: "Authentication required",
            403: "Access forbidden",
            404: "Resource not found",
            405: "Method not allowed",
            409: "Conflict with current state",
            422: "Unprocessable entity",
            429: "Too many requests",
            500: "Internal server error",
            502: "Bad gateway",
            503: "Service unavailable",
            504: "Gateway timeout"
        }
        return messages.get(status_code, "An error occurred")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        identifier: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.identifier = identifier
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
    
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        if self.state == 'OPEN':
            # Check if recovery timeout has passed
            if self.last_failure_time:
                if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                    self.state = 'HALF_OPEN'
                    return True
            return False
        return True
    
    def record_success(self):
        """Record successful request."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
        self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(
                f"Circuit breaker opened for {self.identifier}",
                failure_count=self.failure_count
            )


# Global error handlers for FastAPI
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Global handler for validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Global handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail
            }
        }
    )


# Export main components
__all__ = [
    'ErrorHandlingMiddleware',
    'CircuitBreaker',
    'ErrorCategory',
    'validation_exception_handler',
    'http_exception_handler'
]