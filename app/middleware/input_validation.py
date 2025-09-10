"""
Input Validation and Sanitization Middleware

Provides comprehensive input validation, sanitization, and protection
against common injection attacks including:
- SQL injection attempts
- XSS payloads
- Command injection
- Path traversal
- File upload validation
- JSON structure validation
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Set
from urllib.parse import unquote
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.datastructures import FormData, UploadFile
import structlog

logger = structlog.get_logger(__name__)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive input validation middleware.
    
    Features:
    - SQL injection detection
    - XSS payload detection
    - Path traversal prevention
    - Command injection detection
    - File upload validation
    - JSON structure validation
    - Request size limits
    - Content type validation
    """
    
    def __init__(
        self,
        app,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        max_json_size: int = 1 * 1024 * 1024,      # 1MB
        max_string_length: int = 10000,             # 10k chars
        allowed_file_types: Set[str] = None,
        enable_sql_injection_detection: bool = True,
        enable_xss_detection: bool = True,
        enable_command_injection_detection: bool = True,
        enable_path_traversal_detection: bool = True,
        strict_json_validation: bool = True
    ):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.max_json_size = max_json_size
        self.max_string_length = max_string_length
        self.allowed_file_types = allowed_file_types or {
            '.jpg', '.jpeg', '.png', '.gif', '.webp',  # Images
            '.mp3', '.wav', '.ogg', '.m4a',            # Audio
            '.pdf', '.txt', '.csv'                      # Documents
        }
        self.enable_sql_injection_detection = enable_sql_injection_detection
        self.enable_xss_detection = enable_xss_detection
        self.enable_command_injection_detection = enable_command_injection_detection
        self.enable_path_traversal_detection = enable_path_traversal_detection
        self.strict_json_validation = strict_json_validation
        
        # Initialize detection patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for various attack detection."""
        
        # SQL injection patterns
        self.sql_patterns = [
            re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)", re.IGNORECASE),
            re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"(\bOR\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"(1=1|1=0)", re.IGNORECASE),
            re.compile(r"(\'\s*(OR|AND)\s*\'\w*\'\s*=\s*\'\w*)", re.IGNORECASE),
            re.compile(r"(--|#|/\*)", re.IGNORECASE),
            re.compile(r"(\bxp_cmdshell\b|\bsp_executesql\b)", re.IGNORECASE),
        ]
        
        # XSS patterns
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>", re.IGNORECASE),
            re.compile(r"<object[^>]*>", re.IGNORECASE),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
            re.compile(r"<link[^>]*>", re.IGNORECASE),
            re.compile(r"<meta[^>]*>", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"data:text/html", re.IGNORECASE),
        ]
        
        # Command injection patterns
        self.command_patterns = [
            re.compile(r"[;&|`$(){}[\]<>]"),
            re.compile(r"\b(cat|ls|pwd|whoami|id|uname|ps|kill|rm|mv|cp|chmod|chown)\b"),
            re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
            re.compile(r"(cmd\.exe|powershell\.exe|bash|sh)", re.IGNORECASE),
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
            re.compile(r"(\/etc\/|\\windows\\|\/proc\/)", re.IGNORECASE),
            re.compile(r"(%2e%2e%2f|%2e%2e%5c)", re.IGNORECASE),
            re.compile(r"(\.\.%2f|\.\.%5c)", re.IGNORECASE),
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware entry point."""
        try:
            # Check request size
            if not await self._validate_request_size(request):
                return self._create_error_response("Request too large", 413)
            
            # Validate content type
            if not await self._validate_content_type(request):
                return self._create_error_response("Invalid content type", 415)
            
            # Validate URL and query parameters
            if not await self._validate_url_and_params(request):
                return self._create_error_response("Invalid request parameters", 400)
            
            # Validate request body based on content type
            if not await self._validate_request_body(request):
                return self._create_error_response("Invalid request body", 400)
            
            # If all validations pass, proceed with request
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error("Input validation error", error=str(e), path=request.url.path)
            return self._create_error_response("Validation error", 500)
    
    async def _validate_request_size(self, request: Request) -> bool:
        """Validate request size limits."""
        try:
            content_length = request.headers.get("content-length")
            if content_length:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(
                        "Request size exceeded",
                        size=size,
                        limit=self.max_request_size,
                        path=request.url.path
                    )
                    return False
            return True
        except (ValueError, TypeError):
            return False
    
    async def _validate_content_type(self, request: Request) -> bool:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "")
        
        # Allow common content types
        allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/octet-stream"
        ]
        
        if request.method in ["POST", "PUT", "PATCH"]:
            if not any(allowed_type in content_type for allowed_type in allowed_types):
                logger.warning(
                    "Invalid content type",
                    content_type=content_type,
                    path=request.url.path
                )
                return False
        
        return True
    
    async def _validate_url_and_params(self, request: Request) -> bool:
        """Validate URL path and query parameters."""
        # Check URL path for path traversal
        if self.enable_path_traversal_detection:
            url_path = unquote(str(request.url.path))
            for pattern in self.path_traversal_patterns:
                if pattern.search(url_path):
                    logger.warning(
                        "Path traversal attempt detected in URL",
                        path=url_path,
                        client_ip=request.client.host if request.client else "unknown"
                    )
                    return False
        
        # Check query parameters
        for key, value in request.query_params.items():
            if not self._validate_string_value(key, f"query_param_key:{key}"):
                return False
            if not self._validate_string_value(value, f"query_param_value:{key}"):
                return False
        
        return True
    
    async def _validate_request_body(self, request: Request) -> bool:
        """Validate request body based on content type."""
        content_type = request.headers.get("content-type", "")
        
        try:
            if "application/json" in content_type:
                return await self._validate_json_body(request)
            elif "multipart/form-data" in content_type:
                return await self._validate_form_data(request)
            elif "application/x-www-form-urlencoded" in content_type:
                return await self._validate_form_body(request)
            
            return True
        except Exception as e:
            logger.error("Body validation error", error=str(e))
            return False
    
    async def _validate_json_body(self, request: Request) -> bool:
        """Validate JSON request body."""
        try:
            # Get the body without consuming it
            body = await request.body()
            
            if len(body) > self.max_json_size:
                logger.warning("JSON body too large", size=len(body))
                return False
            
            if not body:
                return True
            
            # Parse and validate JSON structure
            try:
                json_data = json.loads(body)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON", error=str(e))
                return False
            
            # Recursively validate JSON content
            return self._validate_json_content(json_data, "root")
            
        except Exception as e:
            logger.error("JSON validation error", error=str(e))
            return False
    
    def _validate_json_content(self, data: Any, path: str) -> bool:
        """Recursively validate JSON content."""
        if isinstance(data, dict):
            for key, value in data.items():
                if not self._validate_string_value(str(key), f"{path}.{key}"):
                    return False
                if not self._validate_json_content(value, f"{path}.{key}"):
                    return False
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if not self._validate_json_content(item, f"{path}[{i}]"):
                    return False
        
        elif isinstance(data, str):
            if not self._validate_string_value(data, path):
                return False
        
        return True
    
    async def _validate_form_data(self, request: Request) -> bool:
        """Validate multipart form data."""
        try:
            form = await request.form()
            
            for key, value in form.items():
                if isinstance(value, UploadFile):
                    if not self._validate_file_upload(value, key):
                        return False
                else:
                    if not self._validate_string_value(str(key), f"form_key:{key}"):
                        return False
                    if not self._validate_string_value(str(value), f"form_value:{key}"):
                        return False
            
            return True
        except Exception as e:
            logger.error("Form data validation error", error=str(e))
            return False
    
    async def _validate_form_body(self, request: Request) -> bool:
        """Validate URL-encoded form body."""
        try:
            form = await request.form()
            
            for key, value in form.items():
                if not self._validate_string_value(str(key), f"form_key:{key}"):
                    return False
                if not self._validate_string_value(str(value), f"form_value:{key}"):
                    return False
            
            return True
        except Exception as e:
            logger.error("Form body validation error", error=str(e))
            return False
    
    def _validate_file_upload(self, file: UploadFile, field_name: str) -> bool:
        """Validate uploaded file."""
        # Check file extension
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.allowed_file_types:
                logger.warning(
                    "Invalid file type uploaded",
                    filename=file.filename,
                    extension=file_ext,
                    field=field_name
                )
                return False
        
        # Check file size (basic check)
        if hasattr(file, 'size') and file.size:
            if file.size > self.max_request_size:
                logger.warning(
                    "Uploaded file too large",
                    filename=file.filename,
                    size=file.size,
                    field=field_name
                )
                return False
        
        # Check filename for path traversal
        if file.filename and self.enable_path_traversal_detection:
            for pattern in self.path_traversal_patterns:
                if pattern.search(file.filename):
                    logger.warning(
                        "Path traversal in filename",
                        filename=file.filename,
                        field=field_name
                    )
                    return False
        
        return True
    
    def _validate_string_value(self, value: str, context: str) -> bool:
        """Validate string value against various attack patterns."""
        if not value:
            return True
        
        # Check string length
        if len(value) > self.max_string_length:
            logger.warning(
                "String too long",
                length=len(value),
                context=context
            )
            return False
        
        # URL decode for proper validation
        decoded_value = unquote(value)
        
        # SQL injection detection
        if self.enable_sql_injection_detection:
            for pattern in self.sql_patterns:
                if pattern.search(decoded_value):
                    logger.warning(
                        "SQL injection attempt detected",
                        value=value[:100],  # Log first 100 chars
                        context=context
                    )
                    return False
        
        # XSS detection
        if self.enable_xss_detection:
            for pattern in self.xss_patterns:
                if pattern.search(decoded_value):
                    logger.warning(
                        "XSS attempt detected",
                        value=value[:100],
                        context=context
                    )
                    return False
        
        # Command injection detection
        if self.enable_command_injection_detection:
            for pattern in self.command_patterns:
                if pattern.search(decoded_value):
                    logger.warning(
                        "Command injection attempt detected",
                        value=value[:100],
                        context=context
                    )
                    return False
        
        # Path traversal detection
        if self.enable_path_traversal_detection:
            for pattern in self.path_traversal_patterns:
                if pattern.search(decoded_value):
                    logger.warning(
                        "Path traversal attempt detected",
                        value=value[:100],
                        context=context
                    )
                    return False
        
        return True
    
    def _create_error_response(self, message: str, status_code: int) -> Response:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "status_code": status_code,
                "type": "validation_error"
            }
        )


class ContentSecurityValidator:
    """Additional content security validations."""
    
    @staticmethod
    def validate_telegram_update(data: Dict[str, Any]) -> bool:
        """Validate Telegram webhook update structure."""
        required_fields = ['update_id']
        
        if not isinstance(data, dict):
            return False
        
        if 'update_id' not in data:
            return False
        
        # Basic structure validation
        if not isinstance(data['update_id'], int):
            return False
        
        # Check for suspicious nested depth
        if ContentSecurityValidator._get_max_depth(data) > 10:
            logger.warning("Telegram update has suspicious nesting depth")
            return False
        
        return True
    
    @staticmethod
    def _get_max_depth(obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of object."""
        if current_depth > 20:  # Prevent infinite recursion
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                ContentSecurityValidator._get_max_depth(v, current_depth + 1)
                for v in obj.values()
            )
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(
                ContentSecurityValidator._get_max_depth(item, current_depth + 1)
                for item in obj
            )
        else:
            return current_depth


# Export components
__all__ = [
    'InputValidationMiddleware',
    'ContentSecurityValidator'
]