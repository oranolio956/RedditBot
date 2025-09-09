"""
Typing Simulator Initialization Service

Handles proper initialization and integration of the advanced typing simulator
with the existing Telegram bot infrastructure, ensuring seamless compatibility
and optimal performance.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import structlog
from app.config import settings
from app.core.redis import redis_manager
from app.services.typing_simulator import AdvancedTypingSimulator, get_typing_simulator, cleanup_typing_simulator
from app.services.typing_integration import EnhancedTypingIntegration, get_typing_integration, cleanup_typing_integration
from app.services.personality_manager import get_personality_manager
from app.telegram.anti_ban import AntiBanManager
from app.telegram.session import SessionManager

logger = structlog.get_logger(__name__)


class TypingSystemInitializer:
    """
    Handles initialization and lifecycle management of the typing simulator system.
    """
    
    def __init__(self):
        self.typing_simulator: Optional[AdvancedTypingSimulator] = None
        self.typing_integration: Optional[EnhancedTypingIntegration] = None
        self.is_initialized = False
        self.initialization_error: Optional[Exception] = None
        
        # Configuration from settings
        self.enable_advanced_typing = getattr(settings, 'ENABLE_ADVANCED_TYPING', True)
        self.enable_typing_caching = getattr(settings, 'ENABLE_TYPING_CACHING', True)
        self.max_concurrent_sessions = getattr(settings, 'MAX_TYPING_SESSIONS', 1000)
        self.typing_cache_ttl = getattr(settings, 'TYPING_CACHE_TTL', 300)
    
    async def initialize(
        self,
        anti_ban_manager: Optional[AntiBanManager] = None,
        session_manager: Optional[SessionManager] = None,
        force_reinit: bool = False
    ) -> bool:
        """
        Initialize the typing simulator system with proper dependency injection.
        
        Args:
            anti_ban_manager: Anti-ban manager instance for integration
            session_manager: Session manager instance for context
            force_reinit: Force reinitialization even if already initialized
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized and not force_reinit:
            logger.debug("Typing system already initialized")
            return True
        
        try:
            logger.info("Initializing Advanced Typing Simulator System")
            
            # Check if system is enabled
            if not self.enable_advanced_typing:
                logger.info("Advanced typing disabled in configuration")
                return True
            
            # Initialize core components
            await self._initialize_core_simulator()
            
            # Initialize integration layer if dependencies available
            if anti_ban_manager and session_manager:
                await self._initialize_integration_layer(anti_ban_manager, session_manager)
            else:
                logger.warning(
                    "Integration dependencies not provided - running in standalone mode",
                    has_anti_ban=anti_ban_manager is not None,
                    has_session_manager=session_manager is not None
                )
            
            # Perform system validation
            await self._validate_system()
            
            self.is_initialized = True
            self.initialization_error = None
            
            logger.info(
                "Advanced Typing Simulator System initialized successfully",
                standalone_mode=self.typing_integration is None,
                caching_enabled=self.enable_typing_caching,
                max_sessions=self.max_concurrent_sessions
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize typing simulator system", error=str(e), exc_info=True)
            self.initialization_error = e
            self.is_initialized = False
            
            # Attempt cleanup of partial initialization
            await self._cleanup_partial_init()
            
            return False
    
    async def _initialize_core_simulator(self) -> None:
        """Initialize the core typing simulator."""
        try:
            # Get personality manager for integration
            personality_manager = await get_personality_manager()
            
            # Initialize the advanced typing simulator
            self.typing_simulator = await get_typing_simulator()
            
            logger.debug("Core typing simulator initialized")
            
        except Exception as e:
            logger.error("Failed to initialize core simulator", error=str(e))
            raise
    
    async def _initialize_integration_layer(
        self,
        anti_ban_manager: AntiBanManager,
        session_manager: SessionManager
    ) -> None:
        """Initialize the integration layer with existing systems."""
        try:
            # Get personality manager
            personality_manager = await get_personality_manager()
            
            # Initialize integration service
            self.typing_integration = await get_typing_integration(
                anti_ban_manager=anti_ban_manager,
                session_manager=session_manager,
                personality_manager=personality_manager
            )
            
            # Configure integration settings
            if self.typing_integration:
                await self._configure_integration_settings()
            
            logger.debug("Integration layer initialized")
            
        except Exception as e:
            logger.error("Failed to initialize integration layer", error=str(e))
            raise
    
    async def _configure_integration_settings(self) -> None:
        """Configure integration layer settings."""
        if not self.typing_integration:
            return
        
        try:
            # Configure session manager
            session_mgr = self.typing_integration.session_manager_typing
            session_mgr.max_concurrent_sessions = self.max_concurrent_sessions
            session_mgr.cache_ttl = self.typing_cache_ttl if self.enable_typing_caching else 0
            
            # Configure integration settings
            self.typing_integration.enable_advanced_simulation = True
            self.typing_integration.fallback_to_simple = True
            self.typing_integration.performance_monitoring = True
            
            logger.debug("Integration settings configured")
            
        except Exception as e:
            logger.warning("Error configuring integration settings", error=str(e))
    
    async def _validate_system(self) -> None:
        """Validate that the typing system is working correctly."""
        try:
            if self.typing_simulator:
                # Test basic simulation
                test_delay = await self.typing_simulator.get_typing_delay(
                    text="System validation test",
                    user_id=0  # Special test user ID
                )
                
                if not isinstance(test_delay, (int, float)) or test_delay <= 0:
                    raise ValueError(f"Invalid test delay: {test_delay}")
                
                logger.debug("Core simulator validation passed", test_delay=test_delay)
            
            if self.typing_integration:
                # Test integration layer
                test_integration_delay = await self.typing_integration.calculate_typing_delay_enhanced(
                    text="Integration validation test",
                    user_id=0
                )
                
                if not isinstance(test_integration_delay, (int, float)) or test_integration_delay <= 0:
                    raise ValueError(f"Invalid integration delay: {test_integration_delay}")
                
                logger.debug("Integration layer validation passed", test_delay=test_integration_delay)
            
            logger.debug("System validation completed successfully")
            
        except Exception as e:
            logger.error("System validation failed", error=str(e))
            raise
    
    async def _cleanup_partial_init(self) -> None:
        """Clean up partial initialization in case of errors."""
        try:
            if self.typing_integration:
                await cleanup_typing_integration()
                self.typing_integration = None
            
            if self.typing_simulator:
                await cleanup_typing_simulator()
                self.typing_simulator = None
            
            logger.debug("Partial initialization cleaned up")
            
        except Exception as e:
            logger.warning("Error during partial cleanup", error=str(e))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        try:
            status = {
                'initialized': self.is_initialized,
                'enabled': self.enable_advanced_typing,
                'initialization_error': str(self.initialization_error) if self.initialization_error else None,
                'core_simulator_available': self.typing_simulator is not None,
                'integration_available': self.typing_integration is not None,
                'settings': {
                    'caching_enabled': self.enable_typing_caching,
                    'max_concurrent_sessions': self.max_concurrent_sessions,
                    'cache_ttl': self.typing_cache_ttl
                }
            }
            
            # Get performance metrics if available
            if self.typing_integration:
                try:
                    performance_metrics = await self.typing_integration.get_performance_metrics()
                    status['performance_metrics'] = performance_metrics
                except Exception as e:
                    status['performance_metrics_error'] = str(e)
            
            return status
            
        except Exception as e:
            logger.error("Error getting system status", error=str(e))
            return {
                'initialized': False,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the typing system."""
        health_status = {
            'healthy': True,
            'checks': {}
        }
        
        try:
            # Check if system is initialized
            health_status['checks']['initialized'] = self.is_initialized
            if not self.is_initialized:
                health_status['healthy'] = False
            
            # Check core simulator
            if self.typing_simulator:
                try:
                    test_delay = await self.typing_simulator.get_typing_delay(
                        text="Health check test",
                        user_id=-1  # Special health check user ID
                    )
                    health_status['checks']['core_simulator'] = True
                    health_status['checks']['test_delay'] = test_delay
                except Exception as e:
                    health_status['checks']['core_simulator'] = False
                    health_status['checks']['core_simulator_error'] = str(e)
                    health_status['healthy'] = False
            else:
                health_status['checks']['core_simulator'] = False
            
            # Check integration layer
            if self.typing_integration:
                try:
                    integration_delay = await self.typing_integration.calculate_typing_delay_enhanced(
                        text="Health check integration test",
                        user_id=-1
                    )
                    health_status['checks']['integration_layer'] = True
                    health_status['checks']['integration_delay'] = integration_delay
                    
                    # Check performance metrics
                    metrics = await self.typing_integration.get_performance_metrics()
                    health_status['checks']['performance_metrics'] = True
                    health_status['metrics'] = {
                        'active_sessions': metrics.get('active_sessions', 0),
                        'error_rate': metrics.get('error_rate', 0.0),
                        'cache_hit_rate': metrics.get('cache_hit_rate', 0.0)
                    }
                    
                    # Check if error rate is too high
                    if metrics.get('error_rate', 0.0) > 0.05:  # >5% error rate
                        health_status['healthy'] = False
                        health_status['checks']['error_rate_warning'] = True
                    
                except Exception as e:
                    health_status['checks']['integration_layer'] = False
                    health_status['checks']['integration_error'] = str(e)
                    health_status['healthy'] = False
            else:
                health_status['checks']['integration_layer'] = False
            
            # Check Redis connectivity (if caching enabled)
            if self.enable_typing_caching:
                try:
                    await redis_manager.set("typing_health_check", "ok", ttl=60)
                    test_value = await redis_manager.get("typing_health_check")
                    health_status['checks']['redis_connectivity'] = test_value == "ok"
                    if not health_status['checks']['redis_connectivity']:
                        health_status['healthy'] = False
                except Exception as e:
                    health_status['checks']['redis_connectivity'] = False
                    health_status['checks']['redis_error'] = str(e)
                    health_status['healthy'] = False
            
            return health_status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def enable_system(self, enabled: bool = True) -> bool:
        """Enable or disable the typing system."""
        try:
            if self.typing_integration:
                await self.typing_integration.enable_advanced_simulation(enabled)
            
            self.enable_advanced_typing = enabled
            
            logger.info(
                "Typing system toggled",
                enabled=enabled,
                has_integration=self.typing_integration is not None
            )
            
            return True
            
        except Exception as e:
            logger.error("Error toggling typing system", error=str(e), enabled=enabled)
            return False
    
    async def cleanup(self) -> None:
        """Clean up the typing system."""
        try:
            logger.info("Cleaning up typing system")
            
            if self.typing_integration:
                await cleanup_typing_integration()
                self.typing_integration = None
            
            if self.typing_simulator:
                await cleanup_typing_simulator()
                self.typing_simulator = None
            
            self.is_initialized = False
            
            logger.info("Typing system cleanup completed")
            
        except Exception as e:
            logger.error("Error during typing system cleanup", error=str(e))


# Global initializer instance
_typing_initializer: Optional[TypingSystemInitializer] = None


async def initialize_typing_system(
    anti_ban_manager: Optional[AntiBanManager] = None,
    session_manager: Optional[SessionManager] = None,
    force_reinit: bool = False
) -> bool:
    """
    Initialize the global typing system.
    
    Args:
        anti_ban_manager: Anti-ban manager for integration
        session_manager: Session manager for context
        force_reinit: Force reinitialization
        
    Returns:
        bool: True if successful
    """
    global _typing_initializer
    
    if _typing_initializer is None:
        _typing_initializer = TypingSystemInitializer()
    
    return await _typing_initializer.initialize(
        anti_ban_manager=anti_ban_manager,
        session_manager=session_manager,
        force_reinit=force_reinit
    )


async def get_typing_system_status() -> Dict[str, Any]:
    """Get typing system status."""
    if _typing_initializer is None:
        return {
            'initialized': False,
            'error': 'System not initialized'
        }
    
    return await _typing_initializer.get_system_status()


async def typing_system_health_check() -> Dict[str, Any]:
    """Perform health check on typing system."""
    if _typing_initializer is None:
        return {
            'healthy': False,
            'error': 'System not initialized'
        }
    
    return await _typing_initializer.health_check()


async def enable_typing_system(enabled: bool = True) -> bool:
    """Enable or disable typing system."""
    if _typing_initializer is None:
        return False
    
    return await _typing_initializer.enable_system(enabled)


async def cleanup_typing_system() -> None:
    """Clean up typing system."""
    global _typing_initializer
    
    if _typing_initializer:
        await _typing_initializer.cleanup()
        _typing_initializer = None


@asynccontextmanager
async def typing_system_context(
    anti_ban_manager: Optional[AntiBanManager] = None,
    session_manager: Optional[SessionManager] = None
):
    """
    Context manager for typing system lifecycle.
    
    Usage:
        async with typing_system_context(anti_ban, session_mgr):
            # Use typing system
            delay = await get_typing_delay(...)
    """
    try:
        success = await initialize_typing_system(anti_ban_manager, session_manager)
        if not success:
            logger.warning("Typing system initialization failed in context manager")
        
        yield _typing_initializer
        
    finally:
        # Don't cleanup automatically - let global cleanup handle it
        pass


# Convenience functions for backward compatibility
async def get_typing_delay_simple(text: str, user_id: int) -> float:
    """
    Get typing delay with automatic fallback.
    
    This provides a simple interface that automatically uses the advanced
    typing system if available, or falls back to basic calculation.
    """
    try:
        if _typing_initializer and _typing_initializer.is_initialized:
            if _typing_initializer.typing_integration:
                return await _typing_initializer.typing_integration.calculate_typing_delay_enhanced(
                    text=text,
                    user_id=user_id
                )
            elif _typing_initializer.typing_simulator:
                return await _typing_initializer.typing_simulator.get_typing_delay(
                    text=text,
                    user_id=user_id
                )
        
        # Fallback to simple calculation
        import random
        char_count = len(text)
        base_delay = char_count / 120  # ~120 chars per minute
        variation = random.uniform(0.7, 1.4)
        return max(0.5, min(base_delay * variation, 10.0))
        
    except Exception as e:
        logger.warning("Error in typing delay calculation, using fallback", error=str(e))
        import random
        return random.uniform(1.0, 3.0)


async def start_typing_session_simple(
    text: str,
    user_id: int,
    chat_id: int,
    bot: Any,
    send_callback: Optional[Any] = None
) -> Optional[str]:
    """
    Start typing session with automatic fallback.
    
    Returns session ID if successful, None if fallback used.
    """
    try:
        if (_typing_initializer and 
            _typing_initializer.is_initialized and 
            _typing_initializer.typing_integration):
            
            return await _typing_initializer.typing_integration.start_realistic_typing_session(
                text=text,
                user_id=user_id,
                chat_id=chat_id,
                bot=bot,
                send_callback=send_callback
            )
        
        # Fallback: just use simple delay
        if send_callback:
            delay = await get_typing_delay_simple(text, user_id)
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(delay)
            await send_callback()
        
        return None
        
    except Exception as e:
        logger.warning("Error starting typing session, using fallback", error=str(e))
        
        # Final fallback
        if send_callback:
            try:
                await bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(2.0)  # Simple 2-second delay
                await send_callback()
            except:
                pass  # Don't let callback errors propagate
        
        return None