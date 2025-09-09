"""
Main Telegram Bot Implementation

Production-ready Telegram bot with aiogram 3.x, advanced anti-ban measures,
circuit breakers, and comprehensive error handling.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import time
import random
from dataclasses import dataclass

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import Update, BotCommand, MenuButton, MenuButtonCommands
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import aioredis

from app.config import settings
from app.core.redis import redis_manager
from .handlers import setup_handlers
from .middleware import setup_middleware
from .webhook import WebhookManager
from .session import SessionManager
from .rate_limiter import AdvancedRateLimiter
from .anti_ban import AntiBanManager
from .circuit_breaker import CircuitBreakerManager
from .metrics import TelegramMetrics

logger = structlog.get_logger(__name__)


@dataclass
class BotStatus:
    """Bot status tracking."""
    is_running: bool = False
    start_time: Optional[float] = None
    message_count: int = 0
    error_count: int = 0
    last_update_id: Optional[int] = None
    current_connections: int = 0


class TelegramBot:
    """
    High-performance Telegram bot with advanced anti-ban measures.
    
    Features:
    - aiogram 3.x with async patterns
    - Redis-based session storage
    - Advanced rate limiting with sliding windows
    - Natural typing simulation
    - Circuit breaker pattern for resilience
    - Comprehensive error handling
    - Anti-detection measures
    - Real-time metrics
    """
    
    def __init__(self):
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.storage: Optional[RedisStorage] = None
        self.webhook_manager: Optional[WebhookManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.rate_limiter: Optional[AdvancedRateLimiter] = None
        self.anti_ban: Optional[AntiBanManager] = None
        self.circuit_breaker: Optional[CircuitBreakerManager] = None
        self.metrics: Optional[TelegramMetrics] = None
        
        self.status = BotStatus()
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Anti-ban configuration
        self.typing_delays = {
            'min': 0.5,
            'max': 2.0,
            'chars_per_second': 150  # Realistic typing speed
        }
        
        # Message processing queue for natural timing
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_semaphore = asyncio.Semaphore(50)  # Max concurrent processing
    
    async def initialize(self) -> None:
        """
        Initialize bot with all components.
        
        Sets up Redis storage, middleware, handlers, and monitoring.
        """
        try:
            logger.info("Initializing Telegram bot")
            
            # Initialize Redis connection for FSM storage
            redis_client = await aioredis.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                retry_on_timeout=settings.redis.retry_on_timeout,
                decode_responses=True
            )
            
            # Create Redis storage for FSM
            self.storage = RedisStorage(
                redis=redis_client,
                key_builder=lambda chat_id, user_id: f"fsm:{chat_id}:{user_id}",
                state_ttl=settings.redis.session_ttl,
                data_ttl=settings.redis.session_ttl
            )
            
            # Initialize bot with optimized settings
            self.bot = Bot(
                token=settings.telegram.bot_token,
                default=DefaultBotProperties(
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    protect_content=False,
                    allow_sending_without_reply=True,
                )
            )
            
            # Initialize dispatcher
            self.dp = Dispatcher(storage=self.storage)
            
            # Initialize core components
            await self._initialize_components()
            
            # Setup middleware (order matters!)
            await setup_middleware(self.dp, self)
            
            # Setup handlers
            await setup_handlers(self.dp, self)
            
            # Set bot commands
            await self._setup_bot_commands()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.status.is_running = True
            self.status.start_time = time.time()
            
            logger.info("Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize bot", error=str(e), exc_info=True)
            await self.cleanup()
            raise
    
    async def _initialize_components(self) -> None:
        """Initialize all bot components."""
        try:
            # Initialize metrics first (needed by other components)
            self.metrics = TelegramMetrics()
            await self.metrics.initialize()
            
            # Initialize session manager
            self.session_manager = SessionManager()
            await self.session_manager.initialize()
            
            # Initialize advanced rate limiter
            self.rate_limiter = AdvancedRateLimiter()
            await self.rate_limiter.initialize()
            
            # Initialize anti-ban manager
            self.anti_ban = AntiBanManager(self.rate_limiter, self.metrics)
            await self.anti_ban.initialize()
            
            # Initialize circuit breaker
            self.circuit_breaker = CircuitBreakerManager()
            await self.circuit_breaker.initialize()
            
            # Initialize webhook manager
            if settings.telegram.webhook_url:
                self.webhook_manager = WebhookManager(self.bot, self.dp)
                await self.webhook_manager.initialize()
            
            logger.info("All bot components initialized")
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e))
            raise
    
    async def _setup_bot_commands(self) -> None:
        """Setup bot command menu."""
        try:
            commands = [
                BotCommand(command="start", description="Start the bot"),
                BotCommand(command="help", description="Get help"),
                BotCommand(command="status", description="Check bot status"),
                BotCommand(command="settings", description="User settings"),
            ]
            
            await self.bot.set_my_commands(commands)
            
            # Set menu button
            await self.bot.set_chat_menu_button(
                menu_button=MenuButtonCommands(type="commands")
            )
            
            logger.info("Bot commands configured")
            
        except Exception as e:
            logger.error("Failed to setup bot commands", error=str(e))
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        try:
            # Message queue processor
            self._background_tasks.append(
                asyncio.create_task(self._message_queue_processor())
            )
            
            # Metrics updater
            self._background_tasks.append(
                asyncio.create_task(self._metrics_updater())
            )
            
            # Session cleaner
            self._background_tasks.append(
                asyncio.create_task(self._session_cleaner())
            )
            
            # Health checker
            self._background_tasks.append(
                asyncio.create_task(self._health_checker())
            )
            
            # Anti-ban maintenance
            self._background_tasks.append(
                asyncio.create_task(self._anti_ban_maintenance())
            )
            
            logger.info(f"Started {len(self._background_tasks)} background tasks")
            
        except Exception as e:
            logger.error("Failed to start background tasks", error=str(e))
            raise
    
    async def _message_queue_processor(self) -> None:
        """Process message queue with natural timing."""
        while not self._shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    message_data = await asyncio.wait_for(
                        self._message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                async with self._processing_semaphore:
                    await self._process_queued_message(message_data)
                
            except Exception as e:
                logger.error("Error in message queue processor", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_queued_message(self, message_data: Dict[str, Any]) -> None:
        """Process a single queued message with anti-ban measures."""
        try:
            chat_id = message_data['chat_id']
            message_text = message_data['message']
            delay_config = message_data.get('delay_config', {})
            
            # Apply natural typing delay
            if delay_config.get('simulate_typing', True):
                typing_delay = self.calculate_typing_delay(message_text)
                await self.bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(typing_delay)
            
            # Send message through circuit breaker
            await self.circuit_breaker.execute(
                self.bot.send_message,
                chat_id=chat_id,
                text=message_text,
                **message_data.get('send_params', {})
            )
            
            # Update metrics
            await self.metrics.increment_messages_sent()
            
        except Exception as e:
            logger.error("Failed to process queued message", error=str(e))
            await self.metrics.increment_errors()
    
    async def _metrics_updater(self) -> None:
        """Update metrics periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if self.metrics:
                    await self.metrics.update_system_metrics()
                    await self.metrics.update_bot_status(self.status)
                
            except Exception as e:
                logger.error("Error updating metrics", error=str(e))
    
    async def _session_cleaner(self) -> None:
        """Clean expired sessions periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                
                if self.session_manager:
                    cleaned = await self.session_manager.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(f"Cleaned {cleaned} expired sessions")
                
            except Exception as e:
                logger.error("Error cleaning sessions", error=str(e))
    
    async def _health_checker(self) -> None:
        """Perform health checks periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check bot health
                bot_healthy = await self._check_bot_health()
                
                # Check Redis health
                redis_healthy = await redis_manager.health_check()
                
                # Update metrics
                if self.metrics:
                    await self.metrics.update_health_status(
                        bot_healthy=bot_healthy,
                        redis_healthy=redis_healthy
                    )
                
                if not (bot_healthy and redis_healthy):
                    logger.warning(
                        "Health check failed",
                        bot_healthy=bot_healthy,
                        redis_healthy=redis_healthy
                    )
                
            except Exception as e:
                logger.error("Error in health checker", error=str(e))
    
    async def _anti_ban_maintenance(self) -> None:
        """Perform anti-ban maintenance tasks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(120)  # Every 2 minutes
                
                if self.anti_ban:
                    # Update behavior patterns
                    await self.anti_ban.update_patterns()
                    
                    # Clean rate limit data
                    await self.anti_ban.cleanup_old_data()
                
            except Exception as e:
                logger.error("Error in anti-ban maintenance", error=str(e))
    
    async def _check_bot_health(self) -> bool:
        """Check if bot is healthy."""
        try:
            me = await self.bot.get_me()
            return me is not None and me.id is not None
        except Exception as e:
            logger.error("Bot health check failed", error=str(e))
            return False
    
    def calculate_typing_delay(self, text: str) -> float:
        """Calculate natural typing delay based on text length.
        
        Note: This is a fallback method. The enhanced typing system
        provides much more sophisticated simulation through the
        typing integration service.
        """
        chars = len(text)
        base_delay = chars / self.typing_delays['chars_per_second']
        
        # Add random variation
        variation = random.uniform(-0.3, 0.7)  # -30% to +70%
        delay = base_delay * (1 + variation)
        
        # Clamp to reasonable bounds
        return max(
            self.typing_delays['min'],
            min(delay, self.typing_delays['max'])
        )
    
    async def send_message_queued(
        self,
        chat_id: int,
        text: str,
        simulate_typing: bool = True,
        user_id: Optional[int] = None,
        enhanced_typing: bool = True,
        **kwargs
    ) -> None:
        """
        Queue message for sending with natural timing.
        
        Args:
            chat_id: Target chat ID
            text: Message text
            simulate_typing: Whether to simulate typing
            user_id: User ID for enhanced typing simulation
            enhanced_typing: Whether to use advanced typing simulation
            **kwargs: Additional parameters for send_message
        """
        try:
            # Try to use enhanced typing simulation if available
            if enhanced_typing and user_id and simulate_typing:
                try:
                    from app.services.typing_integration import typing_integration
                    
                    if typing_integration:
                        # Start realistic typing session
                        session_id = await typing_integration.start_realistic_typing_session(
                            text=text,
                            user_id=user_id,
                            chat_id=chat_id,
                            bot=self.bot,
                            send_callback=lambda: self.bot.send_message(chat_id, text, **kwargs)
                        )
                        
                        logger.debug(
                            "Started enhanced typing session",
                            session_id=session_id,
                            user_id=user_id,
                            text_length=len(text)
                        )
                        return  # Message will be sent after typing simulation
                        
                except Exception as e:
                    logger.debug("Enhanced typing unavailable, using fallback", error=str(e))
            
            # Fallback to original queuing system
            message_data = {
                'chat_id': chat_id,
                'message': text,
                'delay_config': {
                    'simulate_typing': simulate_typing
                },
                'send_params': kwargs
            }
            
            await self._message_queue.put(message_data)
            
        except Exception as e:
            logger.error("Failed to queue message", error=str(e))
            # Final fallback to direct send
            await self.bot.send_message(chat_id, text, **kwargs)
    
    async def start_polling(self) -> None:
        """Start bot polling."""
        if not self.dp or not self.bot:
            raise RuntimeError("Bot not initialized")
        
        try:
            logger.info("Starting bot polling")
            await self.dp.start_polling(
                self.bot,
                polling_timeout=30,
                handle_as_tasks=True,
                fast=True,
                skip_updates=False
            )
        except Exception as e:
            logger.error("Error in polling", error=str(e))
            raise
    
    async def start_webhook(self, app: web.Application) -> None:
        """Start webhook server."""
        if not self.webhook_manager:
            raise RuntimeError("Webhook manager not initialized")
        
        await self.webhook_manager.setup_webhook(app)
        logger.info("Webhook server started")
    
    async def stop_webhook(self) -> None:
        """Stop webhook."""
        if self.webhook_manager:
            await self.webhook_manager.remove_webhook()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status."""
        return {
            'is_running': self.status.is_running,
            'start_time': self.status.start_time,
            'uptime': time.time() - self.status.start_time if self.status.start_time else 0,
            'message_count': self.status.message_count,
            'error_count': self.status.error_count,
            'current_connections': self.status.current_connections,
            'queue_size': self._message_queue.qsize(),
            'background_tasks': len(self._background_tasks),
            'components': {
                'session_manager': self.session_manager is not None,
                'rate_limiter': self.rate_limiter is not None,
                'anti_ban': self.anti_ban is not None,
                'circuit_breaker': self.circuit_breaker is not None,
                'metrics': self.metrics is not None,
                'webhook_manager': self.webhook_manager is not None,
            },
            'health': {
                'bot': await self._check_bot_health(),
                'redis': await redis_manager.health_check() if redis_manager else False,
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up bot resources."""
        try:
            logger.info("Cleaning up bot resources")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Cleanup components
            if self.webhook_manager:
                await self.webhook_manager.cleanup()
            
            if self.metrics:
                await self.metrics.cleanup()
            
            if self.session_manager:
                await self.session_manager.cleanup()
            
            if self.rate_limiter:
                await self.rate_limiter.cleanup()
            
            if self.anti_ban:
                await self.anti_ban.cleanup()
            
            if self.circuit_breaker:
                await self.circuit_breaker.cleanup()
            
            # Close storage
            if self.storage:
                await self.storage.close()
            
            # Close bot session
            if self.bot:
                await self.bot.session.close()
            
            self.status.is_running = False
            
            logger.info("Bot cleanup completed")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))


# Global bot instance
telegram_bot: Optional[TelegramBot] = None


async def get_bot() -> TelegramBot:
    """Get bot instance (singleton)."""
    global telegram_bot
    
    if telegram_bot is None:
        telegram_bot = TelegramBot()
        await telegram_bot.initialize()
    
    return telegram_bot


async def cleanup_bot() -> None:
    """Cleanup global bot instance."""
    global telegram_bot
    
    if telegram_bot:
        await telegram_bot.cleanup()
        telegram_bot = None