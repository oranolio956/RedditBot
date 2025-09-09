"""
Advanced Telegram Bot Middleware

Comprehensive middleware for Telegram bot including:
- Sophisticated rate limiting with anti-spam detection
- Session management and user tracking
- Performance monitoring and health checks
- Security and abuse prevention
- Intelligent error handling and recovery
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
import structlog
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update, Message, CallbackQuery, InlineQuery

from app.core.redis import redis_manager
from app.telegram.rate_limiter import (
    rate_limiter, check_user_rate_limit, check_global_rate_limit, 
    check_chat_rate_limit, RateLimitType, Priority
)
from app.telegram.session import session_manager

logger = structlog.get_logger(__name__)


@dataclass
class MiddlewareMetrics:
    """Middleware performance metrics."""
    total_updates: int = 0
    successful_updates: int = 0
    blocked_updates: int = 0
    error_updates: int = 0
    total_processing_time: float = 0.0
    peak_processing_time: float = 0.0
    rate_limit_blocks: int = 0
    session_operations: int = 0
    last_reset: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_updates == 0:
            return 100.0
        return (self.successful_updates / self.total_updates) * 100
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        if self.successful_updates == 0:
            return 0.0
        return self.total_processing_time / self.successful_updates


class TelegramRateLimitMiddleware(BaseMiddleware):
    """
    Advanced rate limiting middleware for Telegram bot updates.
    
    Features:
    - Multiple rate limiting algorithms (sliding window, token bucket, fixed window)
    - User-specific limits based on priority and trust score
    - Anti-spam detection and mitigation
    - Dynamic rate adjustments
    - Emergency fallback mode
    - Comprehensive monitoring and statistics
    """
    
    def __init__(self):
        self.rate_limiter = rate_limiter
        self.metrics = MiddlewareMetrics()
        
        # Emergency mode settings
        self.emergency_mode = False
        self.emergency_threshold = 0.1  # 10% success rate triggers emergency mode
        self.emergency_check_interval = 100  # Check every 100 requests
        
        # Performance optimization
        self.batch_size = 50
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        """Process Telegram update with rate limiting."""
        start_time = time.time()
        self.metrics.total_updates += 1
        
        try:
            # Extract event information
            user_id = self._extract_user_id(event)
            chat_id = self._extract_chat_id(event)
            update_type = self._detect_update_type(event)
            content = self._extract_content(event)
            
            # Emergency mode check
            if self._should_check_emergency_mode():
                await self._check_emergency_mode()
            
            if self.emergency_mode:
                logger.warning("Emergency mode active - bypassing rate limits")
                return await handler(event, data)
            
            # Global rate limit check
            global_allowed, global_info = await check_global_rate_limit()
            if not global_allowed:
                self.metrics.blocked_updates += 1
                self.metrics.rate_limit_blocks += 1
                await self._handle_rate_limit_exceeded(
                    event, "global", "Global rate limit exceeded", global_info
                )
                return
            
            # User-specific rate limit check
            if user_id:
                user_allowed, user_info = await check_user_rate_limit(
                    user_id=user_id,
                    action_type=update_type,
                    content=content
                )
                
                if not user_allowed:
                    self.metrics.blocked_updates += 1
                    self.metrics.rate_limit_blocks += 1
                    await self._handle_rate_limit_exceeded(
                        event, "user", f"User rate limit exceeded for {update_type}", user_info
                    )
                    return
            
            # Chat-specific rate limit check for group chats
            if chat_id and chat_id < 0 and user_id:
                chat_allowed, chat_info = await check_chat_rate_limit(
                    chat_id=chat_id,
                    user_id=user_id
                )
                
                if not chat_allowed:
                    self.metrics.blocked_updates += 1
                    self.metrics.rate_limit_blocks += 1
                    await self._handle_rate_limit_exceeded(
                        event, "chat", "Chat rate limit exceeded", chat_info
                    )
                    return
            
            # All checks passed, proceed with handler
            result = await handler(event, data)
            
            # Update success metrics
            self.metrics.successful_updates += 1
            processing_time = time.time() - start_time
            self.metrics.total_processing_time += processing_time
            self.metrics.peak_processing_time = max(
                self.metrics.peak_processing_time, processing_time
            )
            
            # Periodic cleanup
            await self._maybe_cleanup()
            
            return result
            
        except Exception as e:
            logger.error("Rate limiting middleware error", error=str(e))
            self.metrics.error_updates += 1\n            \n            # Emergency mode activation on repeated failures\n            await self._handle_middleware_error()\n            \n            # Allow the request to proceed on error\n            return await handler(event, data)
    
    def _extract_user_id(self, event: TelegramObject) -> Optional[int]:
        """Extract user ID from Telegram event."""
        try:
            if isinstance(event, Update):
                if event.message and event.message.from_user:
                    return event.message.from_user.id
                elif event.callback_query and event.callback_query.from_user:
                    return event.callback_query.from_user.id
                elif event.inline_query and event.inline_query.from_user:
                    return event.inline_query.from_user.id
                elif event.chosen_inline_result and event.chosen_inline_result.from_user:
                    return event.chosen_inline_result.from_user.id
            
            # Direct event types
            if hasattr(event, 'from_user') and event.from_user:
                return event.from_user.id
                
            return None
            
        except Exception as e:
            logger.error("Error extracting user ID", error=str(e))
            return None
    
    def _extract_chat_id(self, event: TelegramObject) -> Optional[int]:
        """Extract chat ID from Telegram event."""
        try:
            if isinstance(event, Update):
                if event.message and event.message.chat:
                    return event.message.chat.id
                elif event.callback_query and event.callback_query.message:
                    return event.callback_query.message.chat.id
            
            # Direct event types
            if hasattr(event, 'chat') and event.chat:
                return event.chat.id
                
            return None
            
        except Exception as e:
            logger.error("Error extracting chat ID", error=str(e))
            return None
    
    def _detect_update_type(self, event: TelegramObject) -> str:
        """Detect the type of Telegram update."""
        try:
            if isinstance(event, Update):
                if event.message:
                    if event.message.text and event.message.text.startswith('/'):
                        return "command"
                    return "message"
                elif event.callback_query:
                    return "callback"
                elif event.inline_query:
                    return "inline"
                elif event.chosen_inline_result:
                    return "inline_result"
                elif event.edited_message:
                    return "edit"
                elif event.channel_post:
                    return "channel_post"
                elif event.edited_channel_post:
                    return "channel_edit"
            
            # Direct event types
            if isinstance(event, Message):
                if event.text and event.text.startswith('/'):
                    return "command"
                return "message"
            elif isinstance(event, CallbackQuery):
                return "callback"
            elif isinstance(event, InlineQuery):
                return "inline"
            
            return "unknown"
            
        except Exception as e:
            logger.error("Error detecting update type", error=str(e))
            return "unknown"
    
    def _extract_content(self, event: TelegramObject) -> Optional[str]:
        """Extract content from Telegram event for spam detection."""
        try:
            if isinstance(event, Update):
                if event.message:
                    return event.message.text or event.message.caption
                elif event.callback_query:
                    return event.callback_query.data
                elif event.inline_query:
                    return event.inline_query.query
            
            # Direct event types
            if isinstance(event, Message):
                return event.text or event.caption
            elif isinstance(event, CallbackQuery):
                return event.data
            elif isinstance(event, InlineQuery):
                return event.query
            
            return None
            
        except Exception as e:
            logger.error("Error extracting content", error=str(e))
            return None
    
    def _should_check_emergency_mode(self) -> bool:
        """Check if emergency mode should be evaluated."""
        return self.metrics.total_updates % self.emergency_check_interval == 0
    
    async def _check_emergency_mode(self) -> None:
        """Check and potentially activate emergency mode."""
        try:
            if self.metrics.total_updates > 100:  # Only check after some requests
                success_rate = self.metrics.success_rate
                
                if success_rate < self.emergency_threshold * 100:
                    if not self.emergency_mode:
                        self.emergency_mode = True
                        logger.critical(
                            "Emergency mode activated",
                            success_rate=success_rate,
                            total_updates=self.metrics.total_updates,
                            blocked_updates=self.metrics.blocked_updates
                        )
                elif success_rate > 50 and self.emergency_mode:
                    # Deactivate emergency mode when things improve
                    self.emergency_mode = False
                    logger.info(
                        "Emergency mode deactivated",
                        success_rate=success_rate
                    )
                    
        except Exception as e:
            logger.error("Error checking emergency mode", error=str(e))
    
    async def _handle_middleware_error(self) -> None:
        """Handle middleware errors and potentially activate emergency mode."""
        try:
            error_rate = self.metrics.error_updates / max(1, self.metrics.total_updates)
            
            if error_rate > 0.1 and not self.emergency_mode:  # More than 10% errors
                self.emergency_mode = True
                logger.critical(
                    "Emergency mode activated due to high error rate",
                    error_rate=error_rate,
                    error_updates=self.metrics.error_updates,
                    total_updates=self.metrics.total_updates
                )
                
        except Exception as e:
            logger.error("Error handling middleware error", error=str(e))
    
    async def _handle_rate_limit_exceeded(
        self,
        event: TelegramObject,
        limit_type: str,
        message: str,
        rate_info: Dict[str, Any]
    ) -> None:
        """Handle rate limit exceeded scenarios."""
        try:
            user_id = self._extract_user_id(event)
            chat_id = self._extract_chat_id(event)
            
            logger.warning(
                "Rate limit exceeded in middleware",
                user_id=user_id,
                chat_id=chat_id,
                limit_type=limit_type,
                message=message,
                remaining=rate_info.get("remaining", 0),
                reset_time=rate_info.get("reset_time", 0),
                spam_penalty=rate_info.get("spam_penalty", 0),
                trust_score=rate_info.get("trust_score", 1.0)
            )
            
            # Optional: Send rate limit notification
            # This would be implemented based on specific bot requirements
            
        except Exception as e:
            logger.error("Error handling rate limit exceeded", error=str(e))
    
    async def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup if needed."""
        try:
            current_time = time.time()
            
            if current_time - self.last_cleanup > self.cleanup_interval:
                self.last_cleanup = current_time
                
                # Cleanup expired rate limit data
                cleaned = await self.rate_limiter.cleanup_expired_data()
                
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired rate limit entries")
                
                # Reset metrics if they're getting too large
                if self.metrics.total_updates > 1000000:  # 1M requests
                    await self._reset_metrics()
                    
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
    
    async def _reset_metrics(self) -> None:
        """Reset metrics counters."""
        try:
            logger.info(
                "Resetting middleware metrics",
                total_updates=self.metrics.total_updates,
                success_rate=self.metrics.success_rate
            )
            
            self.metrics = MiddlewareMetrics()
            
        except Exception as e:
            logger.error("Error resetting metrics", error=str(e))
    
    # Management and monitoring methods
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get middleware metrics."""
        try:
            return {
                "total_updates": self.metrics.total_updates,
                "successful_updates": self.metrics.successful_updates,
                "blocked_updates": self.metrics.blocked_updates,
                "error_updates": self.metrics.error_updates,
                "rate_limit_blocks": self.metrics.rate_limit_blocks,
                "session_operations": self.metrics.session_operations,
                "success_rate": round(self.metrics.success_rate, 2),
                "average_processing_time_ms": round(self.metrics.average_processing_time * 1000, 2),
                "peak_processing_time_ms": round(self.metrics.peak_processing_time * 1000, 2),
                "emergency_mode": self.emergency_mode,
                "last_reset": self.metrics.last_reset,
                "uptime_seconds": time.time() - self.metrics.last_reset
            }
        except Exception as e:
            logger.error("Error getting metrics", error=str(e))
            return {}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the middleware."""
        try:
            metrics = await self.get_metrics()
            
            # Determine health status
            success_rate = metrics.get("success_rate", 0)
            error_rate = self.metrics.error_updates / max(1, self.metrics.total_updates) * 100
            
            if self.emergency_mode:
                status = "critical"
            elif success_rate < 50:
                status = "unhealthy"
            elif error_rate > 10:
                status = "degraded"
            elif success_rate < 90:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "success_rate": success_rate,
                "error_rate": round(error_rate, 2),
                "emergency_mode": self.emergency_mode,
                "total_updates": self.metrics.total_updates,
                "redis_healthy": redis_manager.is_healthy
            }
            
        except Exception as e:
            logger.error("Error getting health status", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def reset_emergency_mode(self) -> bool:
        """Manually reset emergency mode."""
        try:
            self.emergency_mode = False
            logger.info("Emergency mode manually reset")
            return True
        except Exception as e:
            logger.error("Error resetting emergency mode", error=str(e))
            return False


class TelegramSessionMiddleware(BaseMiddleware):
    """
    Session management middleware for Telegram bot.
    
    Handles user session creation, updates, and cleanup.
    """
    
    def __init__(self):
        self.session_manager = session_manager
        self.session_operations = 0
    
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        """Process Telegram update with session management."""
        try:
            # Extract user and chat information
            user_id = self._extract_user_id(event)
            chat_id = self._extract_chat_id(event)
            
            if user_id and chat_id:
                # Get or create session
                session = await self._get_or_create_session(user_id, chat_id, event)
                
                if session:
                    # Add session to handler data
                    data["session"] = session
                    
                    # Update session activity
                    await self.session_manager.update_session_activity(session)
                    self.session_operations += 1
            
            # Call the handler
            return await handler(event, data)
            
        except Exception as e:
            logger.error("Session middleware error", error=str(e))
            # Continue without session on error
            return await handler(event, data)
    
    def _extract_user_id(self, event: TelegramObject) -> Optional[int]:
        """Extract user ID from event."""
        # Same implementation as in rate limit middleware
        try:
            if isinstance(event, Update):
                if event.message and event.message.from_user:
                    return event.message.from_user.id
                elif event.callback_query and event.callback_query.from_user:
                    return event.callback_query.from_user.id
                elif event.inline_query and event.inline_query.from_user:
                    return event.inline_query.from_user.id
            
            if hasattr(event, 'from_user') and event.from_user:
                return event.from_user.id
                
            return None
            
        except Exception:
            return None
    
    def _extract_chat_id(self, event: TelegramObject) -> Optional[int]:
        """Extract chat ID from event."""
        # Same implementation as in rate limit middleware
        try:
            if isinstance(event, Update):
                if event.message and event.message.chat:
                    return event.message.chat.id
                elif event.callback_query and event.callback_query.message:
                    return event.callback_query.message.chat.id
            
            if hasattr(event, 'chat') and event.chat:
                return event.chat.id
                
            return None
            
        except Exception:
            return None
    
    async def _get_or_create_session(
        self,
        user_id: int,
        chat_id: int,
        event: TelegramObject
    ) -> Optional[Any]:
        """Get existing session or create new one."""
        try:
            # Try to get existing session
            session = await self.session_manager.get_user_session(user_id, chat_id)
            
            if not session:
                # Extract user data from event
                user_data = self._extract_user_data(event)
                
                # Create new session
                session = await self.session_manager.create_session(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_data=user_data
                )
            
            return session
            
        except Exception as e:
            logger.error("Error getting or creating session", error=str(e))
            return None
    
    def _extract_user_data(self, event: TelegramObject) -> Dict[str, Any]:
        """Extract user data from Telegram event."""
        try:
            user_data = {}
            
            user = None
            if isinstance(event, Update):
                if event.message and event.message.from_user:
                    user = event.message.from_user
                elif event.callback_query and event.callback_query.from_user:
                    user = event.callback_query.from_user
                elif event.inline_query and event.inline_query.from_user:
                    user = event.inline_query.from_user
            elif hasattr(event, 'from_user'):
                user = event.from_user
            
            if user:
                user_data.update({
                    "username": user.username,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "language_code": user.language_code,
                    "is_bot": user.is_bot,
                    "is_premium": getattr(user, 'is_premium', False)
                })
            
            return user_data
            
        except Exception as e:
            logger.error("Error extracting user data", error=str(e))
            return {}


# Global middleware instances
telegram_rate_limit_middleware = TelegramRateLimitMiddleware()
telegram_session_middleware = TelegramSessionMiddleware()


# Convenience functions for external use
async def get_middleware_metrics() -> Dict[str, Any]:
    """Get comprehensive middleware metrics."""
    try:
        rate_metrics = await telegram_rate_limit_middleware.get_metrics()
        session_metrics = {
            "session_operations": telegram_session_middleware.session_operations
        }
        
        return {
            "rate_limiting": rate_metrics,
            "session_management": session_metrics
        }
        
    except Exception as e:
        logger.error("Error getting middleware metrics", error=str(e))
        return {}


async def get_middleware_health() -> Dict[str, Any]:
    """Get middleware health status."""
    try:
        return await telegram_rate_limit_middleware.get_health_status()
    except Exception as e:
        logger.error("Error getting middleware health", error=str(e))
        return {"status": "error", "error": str(e)}


async def reset_emergency_mode() -> bool:
    """Reset emergency mode."""
    try:
        return await telegram_rate_limit_middleware.reset_emergency_mode()
    except Exception as e:
        logger.error("Error resetting emergency mode", error=str(e))
        return False