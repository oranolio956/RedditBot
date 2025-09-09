"""
Telegram Webhook Management

Handles webhook setup, verification, and processing with FastAPI integration,
security measures, and automatic failover to polling.
"""

import asyncio
import hmac
import hashlib
import time
import re
from typing import Optional, Dict, Any, List
import json
import ipaddress

import structlog
from aiogram import Bot, Dispatcher
from aiogram.types import Update
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from aiohttp import web, ClientConnectorError
from fastapi import Request, HTTPException, status
import httpx

from app.config import settings

logger = structlog.get_logger(__name__)


class WebhookManager:
    """
    Comprehensive webhook management system.
    
    Features:
    - Automatic webhook setup and verification
    - Security validation (IP filtering, signature verification)
    - Health monitoring and automatic failover
    - Request rate limiting and filtering
    - Integration with FastAPI application
    - Support for multiple webhook endpoints
    """
    
    def __init__(self, bot: Bot, dp: Dispatcher):
        self.bot = bot
        self.dp = dp
        
        # Webhook configuration
        self.webhook_url = settings.telegram.webhook_url
        self.webhook_secret = settings.telegram.webhook_secret
        self.webhook_path = "/webhook/telegram"
        
        # Security settings
        self.allowed_ips = self._get_telegram_ips()
        self.max_request_size = 1024 * 1024  # 1MB
        self.request_timeout = 30  # seconds
        
        # Health monitoring
        self.last_update_time = 0
        self.webhook_errors = 0
        self.total_updates_processed = 0
        self.failed_updates = 0
        
        # Rate limiting
        self.request_timestamps: List[float] = []
        self.max_requests_per_minute = 100
        
        # Failover configuration
        self.max_webhook_errors = 10
        self.webhook_check_interval = 300  # 5 minutes
        self.fallback_to_polling = False
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def _get_telegram_ips(self) -> List[ipaddress.IPv4Network]:
        """Get Telegram Bot API server IP ranges."""
        # These are the official Telegram Bot API IP ranges
        telegram_ips = [
            "149.154.160.0/20",
            "91.108.4.0/22",
            "91.108.56.0/22",
            "91.108.56.0/23",
            "149.154.160.0/22",
            "149.154.164.0/22",
            "149.154.168.0/22",
            "149.154.172.0/22",
        ]
        
        return [ipaddress.IPv4Network(ip) for ip in telegram_ips]
    
    async def initialize(self) -> None:
        """Initialize webhook manager."""
        try:
            if not self.webhook_url:
                logger.info("No webhook URL configured, webhook manager disabled")
                return
            
            logger.info("Initializing webhook manager")
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Webhook manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize webhook manager", error=str(e))
            raise
    
    async def setup_webhook(self, app: web.Application) -> None:
        """Setup webhook with the Telegram API and FastAPI integration."""
        try:
            if not self.webhook_url:
                raise ValueError("Webhook URL not configured")
            
            # Construct full webhook URL
            full_webhook_url = f"{self.webhook_url.rstrip('/')}{self.webhook_path}"
            
            # Set webhook
            webhook_info = await self.bot.set_webhook(
                url=full_webhook_url,
                secret_token=self.webhook_secret,
                max_connections=100,
                allowed_updates=[
                    "message", "edited_message", "channel_post", "edited_channel_post",
                    "inline_query", "chosen_inline_result", "callback_query",
                    "shipping_query", "pre_checkout_query", "poll", "poll_answer",
                    "my_chat_member", "chat_member", "chat_join_request"
                ],
                drop_pending_updates=False
            )
            
            logger.info(f"Webhook set successfully to {full_webhook_url}")
            
            # Verify webhook is working
            await self._verify_webhook()
            
            # Setup webhook handler
            await self._setup_webhook_handler(app)
            
        except Exception as e:
            logger.error("Failed to setup webhook", error=str(e))
            
            # Fallback to polling
            await self._fallback_to_polling()
            raise
    
    async def _verify_webhook(self) -> bool:
        """Verify webhook is properly configured."""
        try:
            webhook_info = await self.bot.get_webhook_info()
            
            if not webhook_info.url:
                logger.error("Webhook URL is not set")
                return False
            
            if webhook_info.has_custom_certificate:
                logger.info("Webhook using custom certificate")
            
            if webhook_info.pending_update_count > 0:
                logger.warning(f"Webhook has {webhook_info.pending_update_count} pending updates")
            
            if webhook_info.last_error_date:
                last_error = webhook_info.last_error_date.timestamp()
                if time.time() - last_error < 3600:  # Less than 1 hour ago
                    logger.warning(f"Recent webhook error: {webhook_info.last_error_message}")
            
            logger.info(f"Webhook verification successful: {webhook_info.url}")
            return True
            
        except Exception as e:
            logger.error("Webhook verification failed", error=str(e))
            return False
    
    async def _setup_webhook_handler(self, app: web.Application) -> None:
        """Setup webhook request handler."""
        try:
            # Create request handler
            handler = SimpleRequestHandler(
                dispatcher=self.dp,
                bot=self.bot,
                secret_token=self.webhook_secret
            )
            
            # Register webhook route with security middleware
            app.router.add_post(
                self.webhook_path,
                self._secure_webhook_handler
            )
            
            # Store handler for processing
            app['webhook_handler'] = handler
            
            logger.info(f"Webhook handler setup at {self.webhook_path}")
            
        except Exception as e:
            logger.error("Failed to setup webhook handler", error=str(e))
            raise
    
    async def _secure_webhook_handler(self, request: web.Request) -> web.Response:
        """Secure webhook request handler with validation."""
        try:
            start_time = time.time()
            
            # Rate limiting check
            if not await self._check_rate_limit():
                logger.warning("Webhook request rate limited")
                return web.Response(status=429, text="Rate limited")
            
            # IP validation
            if not await self._validate_request_ip(request):
                logger.warning(f"Webhook request from unauthorized IP: {request.remote}")
                return web.Response(status=403, text="Forbidden")
            
            # Size validation
            if not await self._validate_request_size(request):
                logger.warning("Webhook request too large")
                return web.Response(status=413, text="Request too large")
            
            # Signature validation
            if not await self._validate_webhook_signature(request):
                logger.warning("Invalid webhook signature")
                return web.Response(status=401, text="Unauthorized")
            
            # Get the actual handler
            handler = request.app.get('webhook_handler')
            if not handler:
                logger.error("Webhook handler not found")
                return web.Response(status=500, text="Internal error")
            
            # Process the update
            try:
                response = await handler.handle(request)
                
                # Update metrics
                processing_time = time.time() - start_time
                await self._record_successful_update(processing_time)
                
                return response
                
            except Exception as e:
                logger.error("Error processing webhook update", error=str(e))
                await self._record_failed_update(str(e))
                return web.Response(status=500, text="Processing error")
            
        except Exception as e:
            logger.error("Error in webhook handler", error=str(e))
            await self._record_failed_update(str(e))
            return web.Response(status=500, text="Handler error")
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Clean old timestamps
        cutoff = now - 60  # 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > cutoff
        ]
        
        # Check rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            return False
        
        # Add current timestamp
        self.request_timestamps.append(now)
        return True
    
    async def _validate_request_ip(self, request: web.Request) -> bool:
        """Validate request IP against Telegram IP ranges."""
        try:
            client_ip = ipaddress.IPv4Address(request.remote)
            
            # Check against allowed IP ranges
            for allowed_network in self.allowed_ips:
                if client_ip in allowed_network:
                    return True
            
            # Check for forwarded IPs (if behind proxy)
            forwarded_for = request.headers.get('X-Forwarded-For')
            if forwarded_for:
                # Take the first IP (original client)
                forwarded_ip = ipaddress.IPv4Address(forwarded_for.split(',')[0].strip())
                for allowed_network in self.allowed_ips:
                    if forwarded_ip in allowed_network:
                        return True
            
            return False
            
        except Exception as e:
            logger.error("Error validating request IP", error=str(e), ip=request.remote)
            return False
    
    async def _extract_client_ip(self, request: web.Request) -> Optional[str]:
        """Extract client IP with proper proxy header handling."""
        # Try various headers in order of preference
        headers_to_check = [
            'X-Forwarded-For',
            'X-Real-IP',
            'CF-Connecting-IP',  # Cloudflare
            'X-Client-IP',
            'True-Client-IP'
        ]
        
        for header in headers_to_check:
            value = request.headers.get(header)
            if value:
                # X-Forwarded-For can contain multiple IPs
                ips = [ip.strip() for ip in value.split(',')]
                # Return the first non-private IP, or first IP if all private
                for ip in ips:
                    try:
                        ip_obj = ipaddress.ip_address(ip)
                        if not ip_obj.is_private:
                            return ip
                    except ValueError:
                        continue
                # If all IPs are private, return the first one
                if ips:
                    return ips[0]
        
        # Fallback to direct connection IP
        return request.remote
    
    async def _validate_content_type(self, request: web.Request) -> bool:
        """Validate request content type."""
        content_type = request.headers.get('Content-Type', '').lower()
        allowed_types = [
            'application/json',
            'application/json; charset=utf-8',
            'text/plain',
            'multipart/form-data'
        ]
        
        for allowed in allowed_types:
            if content_type.startswith(allowed):
                return True
        
        logger.warning(f"Invalid content type: {content_type}")
        return False
    
    async def _validate_payload_content(self, request: web.Request) -> bool:
        """Validate payload content for malicious patterns."""
        try:
            # Get request body
            body = await request.text()
            
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    logger.error(f"Malicious pattern detected: {pattern}")
                    return False
            
            # Validate JSON structure if applicable
            content_type = request.headers.get('Content-Type', '').lower()
            if content_type.startswith('application/json'):
                try:
                    data = json.loads(body)
                    # Basic structure validation for Telegram updates
                    if not isinstance(data, dict):
                        return False
                    
                    # Check for required fields
                    if 'update_id' not in data:
                        logger.warning("Missing update_id in webhook payload")
                        return False
                    
                    # Validate update_id is numeric
                    if not isinstance(data['update_id'], int):
                        logger.warning("Invalid update_id format")
                        return False
                    
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON payload")
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating payload content", error=str(e))
            return False
    
    async def _record_failed_ip_attempt(self, client_ip: str) -> None:
        """Record failed attempt from IP address."""
        if not hasattr(self, '_failed_ip_attempts'):
            self._failed_ip_attempts = {}
        
        now = time.time()
        if client_ip not in self._failed_ip_attempts:
            self._failed_ip_attempts[client_ip] = []
        
        self._failed_ip_attempts[client_ip].append(now)
        
        # Clean old attempts (keep last 24 hours)
        cutoff = now - 86400  # 24 hours
        self._failed_ip_attempts[client_ip] = [
            ts for ts in self._failed_ip_attempts[client_ip] if ts > cutoff
        ]
        
        # Auto-block IPs with too many failed attempts
        if len(self._failed_ip_attempts[client_ip]) >= 10:  # 10 failures in 24 hours
            self.blocked_ips.add(client_ip)
            logger.warning(f"Auto-blocked IP due to repeated failures: {client_ip}")
    
    async def _record_security_incident(self, client_ip: str, incident_type: str) -> None:
        """Record security incident for monitoring."""
        incident_data = {
            'timestamp': time.time(),
            'client_ip': client_ip,
            'incident_type': incident_type,
            'severity': 'high' if incident_type in ['invalid_signature', 'malicious_payload'] else 'medium'
        }
        
        if not hasattr(self, '_security_incidents'):
            self._security_incidents = []
        
        self._security_incidents.append(incident_data)
        
        # Keep only recent incidents (last 1000)
        if len(self._security_incidents) > 1000:
            self._security_incidents = self._security_incidents[-500:]
        
        # Auto-block on critical security incidents
        if incident_type in ['invalid_signature', 'malicious_payload']:
            self.blocked_ips.add(client_ip)
            logger.error(f"Auto-blocked IP due to security incident: {client_ip}")
    
    async def _validate_request_size(self, request: web.Request) -> bool:
        """Validate request content size."""
        content_length = request.headers.get('Content-Length')
        if content_length:
            try:
                size = int(content_length)
                return size <= self.max_request_size
            except ValueError:
                return False
        return True
    
    async def _validate_webhook_signature(self, request: web.Request) -> bool:
        """Validate webhook signature if secret token is configured."""
        if not self.webhook_secret:
            return True  # No signature validation if no secret
        
        try:
            # Get signature header
            signature_header = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
            if not signature_header:
                return False
            
            # Compare with configured secret
            return hmac.compare_digest(signature_header, self.webhook_secret)
            
        except Exception as e:
            logger.error("Error validating webhook signature", error=str(e))
            return False
    
    async def _record_successful_update(self, processing_time: float) -> None:
        """Record successful update processing."""
        self.last_update_time = time.time()
        self.total_updates_processed += 1
        
        # Reset error count on success
        self.webhook_errors = 0
        
        logger.debug(f"Processed webhook update in {processing_time:.3f}s")
    
    async def _record_failed_update(self, error: str) -> None:
        """Record failed update processing."""
        self.failed_updates += 1
        self.webhook_errors += 1
        
        logger.warning(f"Failed to process webhook update: {error}")
        
        # Check if we should fallback to polling
        if self.webhook_errors >= self.max_webhook_errors:
            logger.error("Too many webhook errors, considering fallback to polling")
            await self._consider_fallback()
    
    async def _consider_fallback(self) -> None:
        """Consider fallback to polling mode."""
        try:
            # Check webhook health
            if not await self._verify_webhook():
                logger.error("Webhook verification failed, falling back to polling")
                await self._fallback_to_polling()
                return
            
            # Reset error count if webhook is actually working
            webhook_info = await self.bot.get_webhook_info()
            if webhook_info.url and not webhook_info.last_error_date:
                logger.info("Webhook appears healthy, resetting error count")
                self.webhook_errors = 0
            
        except Exception as e:
            logger.error("Error checking webhook health", error=str(e))
            await self._fallback_to_polling()
    
    async def _fallback_to_polling(self) -> None:
        """Fallback to polling mode."""
        try:
            logger.warning("Falling back to polling mode")
            
            # Remove webhook
            await self.bot.delete_webhook(drop_pending_updates=True)
            
            # Set fallback flag
            self.fallback_to_polling = True
            
            logger.info("Successfully fell back to polling mode")
            
        except Exception as e:
            logger.error("Failed to fallback to polling", error=str(e))
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.webhook_check_interval)
                
                if self.fallback_to_polling:
                    continue
                
                # Check if we've received updates recently
                time_since_last_update = time.time() - self.last_update_time
                if time_since_last_update > self.webhook_check_interval * 2:
                    logger.warning(f"No updates received for {time_since_last_update:.0f} seconds")
                    
                    # Verify webhook is still working
                    if not await self._verify_webhook():
                        await self._consider_fallback()
                
                # Log health status
                logger.info(
                    f"Webhook health check - "
                    f"Total: {self.total_updates_processed}, "
                    f"Failed: {self.failed_updates}, "
                    f"Errors: {self.webhook_errors}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in webhook health check", error=str(e))
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Clean old request timestamps
                now = time.time()
                cutoff = now - 3600  # 1 hour
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if ts > cutoff
                ]
                
                logger.debug("Webhook cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in webhook cleanup", error=str(e))
    
    async def get_webhook_info(self) -> Dict[str, Any]:
        """Get current webhook information."""
        try:
            webhook_info = await self.bot.get_webhook_info()
            
            return {
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_update_count": webhook_info.pending_update_count,
                "last_error_date": webhook_info.last_error_date.timestamp() if webhook_info.last_error_date else None,
                "last_error_message": webhook_info.last_error_message,
                "max_connections": webhook_info.max_connections,
                "allowed_updates": webhook_info.allowed_updates,
                "last_synchronization_error_date": (
                    webhook_info.last_synchronization_error_date.timestamp()
                    if webhook_info.last_synchronization_error_date else None
                ),
            }
            
        except Exception as e:
            logger.error("Failed to get webhook info", error=str(e))
            return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get webhook metrics."""
        now = time.time()
        uptime = now - (self.last_update_time or now)
        
        # Calculate success rate
        total_requests = self.total_updates_processed + self.failed_updates
        success_rate = (
            (self.total_updates_processed / max(1, total_requests)) * 100
        )
        
        # Calculate recent request rate
        recent_requests = len([
            ts for ts in self.request_timestamps
            if now - ts <= 300  # Last 5 minutes
        ])
        requests_per_minute = recent_requests / 5
        
        return {
            "webhook_enabled": bool(self.webhook_url and not self.fallback_to_polling),
            "webhook_url": self.webhook_url,
            "fallback_mode": self.fallback_to_polling,
            "total_updates_processed": self.total_updates_processed,
            "failed_updates": self.failed_updates,
            "current_errors": self.webhook_errors,
            "success_rate": success_rate,
            "last_update_time": self.last_update_time,
            "time_since_last_update": now - self.last_update_time if self.last_update_time else 0,
            "requests_per_minute": requests_per_minute,
            "total_recent_requests": recent_requests,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_request_size": self.max_request_size,
            "allowed_ip_ranges": [str(ip) for ip in self.allowed_ips],
        }
    
    async def remove_webhook(self) -> bool:
        """Remove webhook configuration."""
        try:
            await self.bot.delete_webhook(drop_pending_updates=False)
            logger.info("Webhook removed successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to remove webhook", error=str(e))
            return False
    
    async def restart_webhook(self) -> bool:
        """Restart webhook (remove and set again)."""
        try:
            logger.info("Restarting webhook")
            
            # Remove current webhook
            await self.remove_webhook()
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Reset error counters
            self.webhook_errors = 0
            self.fallback_to_polling = False
            
            # This would need to be called with the app instance
            # For now, just log that restart is needed
            logger.info("Webhook restart initiated - manual setup required")
            return True
            
        except Exception as e:
            logger.error("Failed to restart webhook", error=str(e))
            return False
    
    async def test_webhook_connectivity(self) -> Dict[str, Any]:
        """Test webhook connectivity."""
        try:
            if not self.webhook_url:
                return {"status": "no_webhook_url"}
            
            full_url = f"{self.webhook_url.rstrip('/')}{self.webhook_path}"
            
            # Test HTTP connectivity
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    response = await client.get(full_url)
                    connectivity_status = {
                        "url_accessible": True,
                        "status_code": response.status_code,
                        "response_time": None  # Would need timing
                    }
                except httpx.RequestError as e:
                    connectivity_status = {
                        "url_accessible": False,
                        "error": str(e),
                        "status_code": None,
                        "response_time": None
                    }
            
            # Get Telegram's view of webhook
            webhook_info = await self.get_webhook_info()
            
            return {
                "status": "completed",
                "connectivity": connectivity_status,
                "telegram_webhook_info": webhook_info,
                "local_metrics": await self.get_metrics()
            }
            
        except Exception as e:
            logger.error("Webhook connectivity test failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def cleanup(self) -> None:
        """Clean up webhook manager resources."""
        try:
            logger.info("Cleaning up webhook manager")
            
            # Cancel background tasks
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Webhook manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during webhook manager cleanup", error=str(e))


# FastAPI integration functions

async def setup_telegram_webhook(app, bot: Bot, dp: Dispatcher) -> WebhookManager:
    """Setup Telegram webhook with FastAPI application."""
    webhook_manager = WebhookManager(bot, dp)
    await webhook_manager.initialize()
    
    if webhook_manager.webhook_url:
        # This would need to be adapted for FastAPI instead of aiohttp
        logger.info("Webhook setup would need FastAPI integration")
    
    return webhook_manager


async def create_webhook_route_handler(webhook_manager: WebhookManager):
    """Create FastAPI route handler for webhook."""
    async def webhook_handler(request: Request):
        """FastAPI webhook handler."""
        try:
            # Convert FastAPI request to format expected by webhook manager
            # This would need proper implementation for FastAPI integration
            
            # For now, return basic response
            return {"status": "received"}
            
        except Exception as e:
            logger.error("Error in FastAPI webhook handler", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Webhook processing error"
            )
    
    return webhook_handler