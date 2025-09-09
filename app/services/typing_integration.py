"""
Typing Simulator Integration Service

Integrates the advanced typing simulator with the existing Telegram bot infrastructure,
providing seamless compatibility with anti-ban systems, personality handlers, and
real-time typing indicators while maintaining performance for 1000+ concurrent users.
"""

import asyncio
import time
import json
import random
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque

import structlog
from aiogram import Bot
from aiogram.types import Message, Chat

from app.core.redis import redis_manager
from app.models.personality import PersonalityProfile
from app.telegram.anti_ban import AntiBanManager, RiskLevel
from app.telegram.session import SessionManager, MessageContext
from app.services.typing_simulator import AdvancedTypingSimulator, get_typing_simulator
from app.services.personality_manager import PersonalityManager

logger = structlog.get_logger(__name__)


class TypingIndicatorState(Enum):
    """Typing indicator states for Telegram."""
    IDLE = "idle"
    TYPING = "typing"
    PAUSED = "paused"
    THINKING = "thinking"
    CORRECTING = "correcting"


@dataclass
class TypingSession:
    """Active typing session data."""
    user_id: int
    chat_id: int
    message_text: str
    start_time: float
    total_duration: float
    current_position: float = 0.0
    indicators: List[Dict[str, Any]] = None
    is_active: bool = True
    personality_context: Optional[Dict[str, Any]] = None
    conversation_context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance tracking for concurrent typing sessions."""
    active_sessions: int = 0
    completed_sessions: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0


class TypingSessionManager:
    """Manages concurrent typing sessions with performance optimization."""
    
    def __init__(self):
        self.active_sessions: Dict[str, TypingSession] = {}
        self.session_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.background_tasks: List[asyncio.Task] = []
        self.performance_metrics = PerformanceMetrics()
        
        # Performance optimization
        self.session_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_concurrent_sessions = 1000
        self.batch_size = 50
        
        # Rate limiting for typing indicators
        self.indicator_rate_limits: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    
    async def start_typing_session(
        self,
        user_id: int,
        chat_id: int,
        message_text: str,
        bot: Bot,
        personality_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Start a new typing session with realistic indicators."""
        session_id = f"{user_id}_{chat_id}_{int(time.time() * 1000)}"
        
        try:
            # Check concurrent session limits
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                logger.warning(
                    "Max concurrent typing sessions reached",
                    active_sessions=len(self.active_sessions),
                    user_id=user_id
                )
                # Use fallback quick typing
                return await self._start_fallback_session(session_id, message_text)
            
            # Check cache for similar typing patterns
            cache_key = self._generate_cache_key(message_text, user_id, personality_context)
            cached_simulation = await self._get_cached_simulation(cache_key)
            
            if cached_simulation:
                # Use cached simulation with variations
                simulation_data = await self._apply_cache_variations(cached_simulation)
                self.performance_metrics.cache_hit_rate += 0.1
            else:
                # Generate new simulation
                typing_simulator = await get_typing_simulator()
                
                # Get personality profile if available
                personality_profile = None
                if personality_context and 'profile_id' in personality_context:
                    try:
                        # This would typically fetch from database
                        personality_profile = personality_context.get('profile')
                    except Exception as e:
                        logger.debug("Could not load personality profile", error=str(e))
                
                # Generate comprehensive typing simulation
                simulation_data = await typing_simulator.simulate_human_typing(
                    text=message_text,
                    user_id=user_id,
                    personality_profile=personality_profile,
                    context=conversation_context,
                    conversation_state=conversation_context
                )
                
                # Cache the simulation
                await self._cache_simulation(cache_key, simulation_data)
            
            # Create typing session
            session = TypingSession(
                user_id=user_id,
                chat_id=chat_id,
                message_text=message_text,
                start_time=time.time(),
                total_duration=simulation_data.get('total_time', 2.0),
                indicators=simulation_data.get('typing_events', []),
                personality_context=personality_context,
                conversation_context=conversation_context
            )
            
            self.active_sessions[session_id] = session
            
            # Start typing indicator task
            indicator_task = asyncio.create_task(
                self._run_typing_indicators(session_id, bot, simulation_data)
            )
            self.background_tasks.append(indicator_task)
            
            self.performance_metrics.active_sessions = len(self.active_sessions)
            
            logger.debug(
                "Started typing session",
                session_id=session_id,
                user_id=user_id,
                duration=session.total_duration,
                events=len(session.indicators)
            )
            
            return session_id
            
        except Exception as e:
            logger.error("Error starting typing session", error=str(e), user_id=user_id)
            return await self._start_fallback_session(session_id, message_text)
    
    async def _run_typing_indicators(
        self,
        session_id: str,
        bot: Bot,
        simulation_data: Dict[str, Any]
    ) -> None:
        """Run typing indicators for a session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            chat_id = session.chat_id
            user_id = session.user_id
            
            # Check rate limits for typing indicators
            if not self._check_indicator_rate_limit(user_id):
                logger.debug("Typing indicator rate limited", user_id=user_id)
                await asyncio.sleep(session.total_duration)
                return
            
            typing_events = simulation_data.get('typing_events', [])
            
            if not typing_events:
                # Simple typing indicator
                await bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(session.total_duration)
                return
            
            current_time = 0.0
            last_indicator_time = 0.0
            
            for event_data in typing_events:
                if not session.is_active:
                    break
                
                event_type = event_data.get('event_type', 'keypress')
                duration = event_data.get('duration', 0.1)
                
                # Determine appropriate chat action
                chat_action = self._get_chat_action(event_type, current_time - last_indicator_time)
                
                # Send typing indicator if needed
                if chat_action and (current_time - last_indicator_time) >= 5.0:
                    try:
                        await bot.send_chat_action(chat_id, chat_action)
                        last_indicator_time = current_time
                    except Exception as e:
                        logger.debug("Error sending chat action", error=str(e))
                
                # Wait for event duration
                if duration > 0:
                    await asyncio.sleep(min(duration, 10.0))  # Cap at 10 seconds
                
                current_time += duration
                session.current_position = current_time
            
        except asyncio.CancelledError:
            logger.debug("Typing indicator task cancelled", session_id=session_id)
        except Exception as e:
            logger.error("Error in typing indicators", error=str(e), session_id=session_id)
        finally:
            # Clean up session
            await self._cleanup_session(session_id)
    
    def _get_chat_action(self, event_type: str, time_since_last: float) -> Optional[str]:
        """Determine appropriate Telegram chat action."""
        # Only send typing indicators periodically to avoid spam
        if time_since_last < 4.0:
            return None
        
        if event_type in ['keypress', 'correction_keypress']:
            return "typing"
        elif event_type in ['pause', 'thinking_pause']:
            # Don't send action for pauses (shows as not typing)
            return None
        elif event_type in ['error_detection', 'backspace']:
            return "typing"  # Still typing during corrections
        
        return "typing"  # Default
    
    def _check_indicator_rate_limit(self, user_id: int) -> bool:
        """Check if user is within rate limits for typing indicators."""
        now = time.time()
        user_indicators = self.indicator_rate_limits[user_id]
        
        # Remove old entries
        while user_indicators and user_indicators[0] < now - 60:  # 1 minute window
            user_indicators.popleft()
        
        # Check limit (max 10 typing sessions per minute)
        if len(user_indicators) >= 10:
            return False
        
        user_indicators.append(now)
        return True
    
    async def _start_fallback_session(self, session_id: str, message_text: str) -> str:
        """Start simple fallback typing session."""
        # Simple calculation
        char_count = len(message_text)
        duration = max(1.0, min(char_count / 120, 10.0))  # 1-10 seconds
        
        # No complex simulation, just wait
        await asyncio.sleep(duration)
        
        return session_id
    
    def _generate_cache_key(
        self,
        message_text: str,
        user_id: int,
        personality_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for typing simulation."""
        # Normalize text for caching
        text_length = len(message_text)
        text_complexity = len(set(message_text.lower())) / len(message_text) if message_text else 0
        
        # Include personality hash if available
        personality_hash = ""
        if personality_context:
            personality_hash = str(hash(json.dumps(personality_context, sort_keys=True)))[:8]
        
        # Group similar length/complexity texts
        length_bucket = (text_length // 20) * 20  # Bucket by 20 chars
        complexity_bucket = int(text_complexity * 10) / 10  # Bucket by 0.1
        
        return f"typing_cache:{length_bucket}:{complexity_bucket}:{personality_hash}"
    
    async def _get_cached_simulation(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached typing simulation."""
        try:
            cached_data = await redis_manager.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug("Error getting cached simulation", error=str(e))
        
        return None
    
    async def _cache_simulation(self, cache_key: str, simulation_data: Dict[str, Any]) -> None:
        """Cache typing simulation data."""
        try:
            # Only cache successful simulations
            if simulation_data.get('realism_score', 0) > 0.7:
                await redis_manager.set(
                    cache_key,
                    json.dumps(simulation_data),
                    ttl=self.cache_ttl
                )
        except Exception as e:
            logger.debug("Error caching simulation", error=str(e))
    
    async def _apply_cache_variations(self, cached_simulation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply variations to cached simulation to avoid detection."""
        simulation = cached_simulation.copy()
        
        # Vary total time by ±15%
        if 'total_time' in simulation:
            variation = random.uniform(0.85, 1.15)
            simulation['total_time'] *= variation
        
        # Vary typing events
        if 'typing_events' in simulation:
            events = simulation['typing_events'].copy()
            for event in events:
                if 'duration' in event:
                    event_variation = random.uniform(0.8, 1.2)
                    event['duration'] *= event_variation
            
            simulation['typing_events'] = events
        
        # Update effective WPM
        if 'effective_wpm' in simulation:
            wpm_variation = random.uniform(0.9, 1.1)
            simulation['effective_wpm'] *= wpm_variation
        
        return simulation
    
    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up completed typing session."""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.is_active = False
                del self.active_sessions[session_id]
                
                self.performance_metrics.completed_sessions += 1
                self.performance_metrics.active_sessions = len(self.active_sessions)
        
        except Exception as e:
            logger.debug("Error cleaning up session", error=str(e), session_id=session_id)
    
    async def stop_session(self, session_id: str) -> None:
        """Stop an active typing session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].is_active = False
            await self._cleanup_session(session_id)
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a typing session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        progress = session.current_position / session.total_duration if session.total_duration > 0 else 1.0
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'is_active': session.is_active,
            'progress': min(1.0, progress),
            'elapsed_time': time.time() - session.start_time,
            'estimated_remaining': max(0, session.total_duration - session.current_position)
        }
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired or stuck sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Mark as expired if running too long (max 60 seconds)
            if current_time - session.start_time > 60:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self._cleanup_session(session_id)
        
        return len(expired_sessions)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            'active_sessions': self.performance_metrics.active_sessions,
            'completed_sessions': self.performance_metrics.completed_sessions,
            'cache_hit_rate': min(1.0, self.performance_metrics.cache_hit_rate / max(1, self.performance_metrics.completed_sessions)),
            'error_rate': self.performance_metrics.error_rate,
            'background_tasks': len([t for t in self.background_tasks if not t.done()]),
            'session_queue_size': self.session_queue.qsize()
        }


class EnhancedTypingIntegration:
    """
    Enhanced typing integration that connects advanced simulation with existing systems.
    
    Provides seamless integration with:
    - Anti-ban systems
    - Personality handlers  
    - Session management
    - Performance monitoring
    - Real-time indicators
    """
    
    def __init__(
        self,
        anti_ban_manager: AntiBanManager,
        session_manager: SessionManager,
        personality_manager: PersonalityManager
    ):
        self.anti_ban_manager = anti_ban_manager
        self.session_manager = session_manager
        self.personality_manager = personality_manager
        self.typing_simulator: Optional[AdvancedTypingSimulator] = None
        self.session_manager_typing = TypingSessionManager()
        
        # Integration settings
        self.enable_advanced_simulation = True
        self.fallback_to_simple = True
        self.max_simulation_time = 30.0  # Max 30 seconds
        self.performance_monitoring = True
        
        # Metrics
        self.integration_metrics = {
            'total_requests': 0,
            'advanced_simulations': 0,
            'simple_fallbacks': 0,
            'errors': 0,
            'average_response_time': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the typing integration system."""
        try:
            logger.info("Initializing Enhanced Typing Integration")
            
            # Initialize typing simulator
            self.typing_simulator = await get_typing_simulator()
            
            # Start background maintenance tasks
            asyncio.create_task(self._maintenance_task())
            
            logger.info("Enhanced Typing Integration initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize typing integration", error=str(e))
            raise
    
    async def calculate_typing_delay_enhanced(
        self,
        text: str,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        message: Optional[Message] = None,
        risk_level: Optional[RiskLevel] = None
    ) -> float:
        """
        Enhanced typing delay calculation that integrates with existing anti-ban system.
        
        This method provides backward compatibility while adding advanced simulation capabilities.
        """
        start_time = time.time()
        
        try:
            self.integration_metrics['total_requests'] += 1
            
            # Quick validation
            if not text or len(text) > 4000:  # Telegram message limit
                return random.uniform(0.5, 2.0)
            
            # Check if advanced simulation is enabled and conditions are met
            if (self.enable_advanced_simulation and 
                self.typing_simulator and
                len(text) > 10):  # Only for substantial messages
                
                try:
                    # Get personality profile
                    personality_profile = None
                    if self.personality_manager:
                        try:
                            user_mapping = await self.personality_manager.get_user_personality_mapping(user_id)
                            if user_mapping:
                                personality_profile = user_mapping.profile
                        except Exception:
                            pass  # Continue without personality
                    
                    # Build enhanced context
                    enhanced_context = await self._build_enhanced_context(
                        user_id, context, message, risk_level
                    )
                    
                    # Get advanced simulation
                    delay = await self.typing_simulator.get_typing_delay(
                        text=text,
                        user_id=user_id,
                        personality_profile=personality_profile,
                        context=enhanced_context,
                        conversation_state=enhanced_context.get('conversation_state')
                    )
                    
                    # Apply risk-based adjustments (integration with existing anti-ban)
                    if risk_level:
                        delay = await self._apply_risk_adjustments(delay, risk_level, user_id)
                    
                    # Ensure reasonable bounds
                    delay = max(0.3, min(delay, self.max_simulation_time))
                    
                    self.integration_metrics['advanced_simulations'] += 1
                    
                    logger.debug(
                        "Advanced typing simulation completed",
                        user_id=user_id,
                        text_length=len(text),
                        delay=delay,
                        risk_level=risk_level.value if risk_level else None
                    )
                    
                    return delay
                    
                except Exception as e:
                    logger.warning("Advanced simulation failed, using fallback", error=str(e))
                    # Fall through to simple calculation
            
            # Simple calculation (existing behavior)
            delay = await self._calculate_simple_delay(text, user_id, risk_level)
            self.integration_metrics['simple_fallbacks'] += 1
            
            return delay
            
        except Exception as e:
            logger.error("Error in typing delay calculation", error=str(e))
            self.integration_metrics['errors'] += 1
            return random.uniform(1.0, 3.0)  # Safe fallback
        finally:
            # Update metrics
            response_time = time.time() - start_time
            self._update_response_time_metric(response_time)
    
    async def start_realistic_typing_session(
        self,
        text: str,
        user_id: int,
        chat_id: int,
        bot: Bot,
        message_context: Optional[MessageContext] = None,
        send_callback: Optional[Callable] = None
    ) -> str:
        """
        Start a realistic typing session with live indicators.
        
        Returns session ID that can be used to monitor or cancel the session.
        """
        try:
            # Get personality and conversation context
            personality_context = None
            conversation_context = {}
            
            if self.personality_manager:
                try:
                    user_mapping = await self.personality_manager.get_user_personality_mapping(user_id)
                    if user_mapping:
                        personality_context = {
                            'profile_id': str(user_mapping.profile_id),
                            'profile': user_mapping.profile,
                            'adapted_traits': user_mapping.adapted_profile_traits or {}
                        }
                except Exception as e:
                    logger.debug("Could not get personality context", error=str(e))
            
            if message_context:
                conversation_context = {
                    'conversation_mode': message_context.mode.value if message_context.mode else 'standard',
                    'topic_familiarity': message_context.topics.get('familiarity', 0.7) if message_context.topics else 0.7,
                    'emotional_state': message_context.sentiment.get('compound', 0.0) if message_context.sentiment else 0.0,
                    'conversation_flow': message_context.engagement_score,
                    'time_pressure': 0.3 if message_context.response_urgency == 'high' else 0.0
                }
            
            # Start typing session with realistic indicators
            session_id = await self.session_manager_typing.start_typing_session(
                user_id=user_id,
                chat_id=chat_id,
                message_text=text,
                bot=bot,
                personality_context=personality_context,
                conversation_context=conversation_context
            )
            
            # Schedule message sending after typing simulation
            if send_callback:
                asyncio.create_task(
                    self._schedule_message_send(session_id, send_callback)
                )
            
            return session_id
            
        except Exception as e:
            logger.error("Error starting typing session", error=str(e))
            # Fallback to simple delay
            delay = await self._calculate_simple_delay(text, user_id)
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(delay)
            
            if send_callback:
                await send_callback()
            
            return f"fallback_{int(time.time())}"
    
    async def _schedule_message_send(
        self,
        session_id: str,
        send_callback: Callable
    ) -> None:
        """Schedule message sending after typing simulation completes."""
        try:
            # Wait for session to complete
            max_wait = 60  # Maximum wait time
            elapsed = 0
            
            while elapsed < max_wait:
                session_status = await self.session_manager_typing.get_session_status(session_id)
                
                if not session_status or session_status.get('progress', 0) >= 1.0:
                    break
                
                await asyncio.sleep(0.5)
                elapsed += 0.5
            
            # Send the message
            await send_callback()
            
        except Exception as e:
            logger.error("Error in scheduled message send", error=str(e))
            # Still try to send the message
            try:
                await send_callback()
            except:
                pass  # Final fallback - message might not send
    
    async def _build_enhanced_context(
        self,
        user_id: int,
        base_context: Optional[Dict[str, Any]],
        message: Optional[Message],
        risk_level: Optional[RiskLevel]
    ) -> Dict[str, Any]:
        """Build enhanced context for advanced typing simulation."""
        context = base_context.copy() if base_context else {}
        
        # Add risk-based context
        if risk_level:
            context['time_pressure'] = {
                RiskLevel.LOW: 0.0,
                RiskLevel.MEDIUM: 0.2,
                RiskLevel.HIGH: 0.5,
                RiskLevel.CRITICAL: 0.8
            }.get(risk_level, 0.0)
        
        # Add message-based context
        if message:
            # Determine device type from message patterns (heuristic)
            if hasattr(message, 'text') and message.text:
                text = message.text
                # Mobile typing patterns (shorter messages, more abbreviations)
                if len(text) < 50 and any(abbr in text.lower() for abbr in ['u', 'ur', 'omg', 'lol']):
                    context['device_type'] = 'mobile'
                elif len(text) > 200:
                    context['device_type'] = 'desktop'
                else:
                    context['device_type'] = 'desktop'  # Default
            
            # Time-based context
            hour = datetime.now().hour
            if 6 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                context['time_pressure'] = context.get('time_pressure', 0.0) + 0.2
            elif 22 <= hour or hour <= 6:  # Late night/early morning
                context['environment_noise'] = 0.1  # Quiet environment
        
        # Add conversation state
        if self.session_manager:
            try:
                session = await self.session_manager.get_session(user_id)
                if session:
                    context['conversation_state'] = {
                        'message_count': session.message_count,
                        'session_duration': time.time() - session.start_time,
                        'engagement_score': session.engagement_score,
                        'emotional_state': session.emotional_context.get('valence', 0.0) if session.emotional_context else 0.0
                    }
            except Exception:
                pass  # Continue without session context
        
        return context
    
    async def _apply_risk_adjustments(
        self,
        base_delay: float,
        risk_level: RiskLevel,
        user_id: int
    ) -> float:
        """Apply risk-based timing adjustments for integration with anti-ban."""
        try:
            # Get existing anti-ban delay calculation for comparison/integration
            if self.anti_ban_manager:
                try:
                    # This integrates with the existing anti-ban system
                    anti_ban_delay = await self.anti_ban_manager.calculate_typing_delay(
                        text="",  # We already have the calculated delay
                        user_id=user_id,
                        context={'risk_level': risk_level.value}
                    )
                    
                    # Blend the delays - use advanced simulation as base, anti-ban as modifier
                    risk_multiplier = {
                        RiskLevel.LOW: 1.0,
                        RiskLevel.MEDIUM: 1.2,
                        RiskLevel.HIGH: 1.5,
                        RiskLevel.CRITICAL: 2.0
                    }.get(risk_level, 1.0)
                    
                    # Take the higher of the two delays for safety
                    adjusted_delay = max(base_delay * risk_multiplier, anti_ban_delay)
                    
                    return adjusted_delay
                    
                except Exception as e:
                    logger.debug("Error integrating with anti-ban delay", error=str(e))
            
            # Fallback risk adjustments
            risk_multipliers = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 1.3,
                RiskLevel.HIGH: 1.7,
                RiskLevel.CRITICAL: 2.2
            }
            
            return base_delay * risk_multipliers.get(risk_level, 1.0)
            
        except Exception as e:
            logger.error("Error applying risk adjustments", error=str(e))
            return base_delay
    
    async def _calculate_simple_delay(
        self,
        text: str,
        user_id: int,
        risk_level: Optional[RiskLevel] = None
    ) -> float:
        """Simple typing delay calculation (fallback)."""
        # Use existing anti-ban calculation if available
        if self.anti_ban_manager:
            try:
                return await self.anti_ban_manager.calculate_typing_delay(
                    text=text,
                    user_id=user_id,
                    context={'risk_level': risk_level.value if risk_level else 'low'}
                )
            except Exception:
                pass
        
        # Basic calculation
        char_count = len(text)
        base_delay = char_count / 120  # ~120 chars per minute
        variation = random.uniform(0.7, 1.4)
        
        # Apply risk multiplier
        if risk_level:
            risk_multipliers = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 1.3,
                RiskLevel.HIGH: 1.6,
                RiskLevel.CRITICAL: 2.0
            }
            variation *= risk_multipliers.get(risk_level, 1.0)
        
        return max(0.5, min(base_delay * variation, 10.0))
    
    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        try:
            total_requests = self.integration_metrics['total_requests']
            current_avg = self.integration_metrics['average_response_time']
            
            # Exponential moving average
            alpha = 0.1
            new_avg = alpha * response_time + (1 - alpha) * current_avg
            self.integration_metrics['average_response_time'] = new_avg
            
        except Exception:
            pass  # Non-critical metric update
    
    async def _maintenance_task(self) -> None:
        """Background maintenance task."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up expired sessions
                expired = await self.session_manager_typing.cleanup_expired_sessions()
                if expired > 0:
                    logger.debug(f"Cleaned up {expired} expired typing sessions")
                
                # Log performance metrics periodically
                if self.performance_monitoring:
                    metrics = await self.get_performance_metrics()
                    
                    logger.info(
                        "Typing integration performance metrics",
                        **metrics
                    )
                
            except Exception as e:
                logger.error("Error in maintenance task", error=str(e))
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        session_metrics = await self.session_manager_typing.get_performance_metrics()
        
        return {
            **self.integration_metrics,
            **session_metrics,
            'advanced_simulation_rate': (
                self.integration_metrics['advanced_simulations'] / 
                max(1, self.integration_metrics['total_requests'])
            ),
            'error_rate': (
                self.integration_metrics['errors'] / 
                max(1, self.integration_metrics['total_requests'])
            )
        }
    
    async def enable_advanced_simulation(self, enabled: bool = True) -> None:
        """Enable or disable advanced simulation."""
        self.enable_advanced_simulation = enabled
        logger.info(
            "Advanced typing simulation toggled",
            enabled=enabled
        )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop all active sessions
            if self.session_manager_typing:
                for session_id in list(self.session_manager_typing.active_sessions.keys()):
                    await self.session_manager_typing.stop_session(session_id)
            
            logger.info("Typing integration cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during typing integration cleanup", error=str(e))


# Global integration instance
typing_integration: Optional[EnhancedTypingIntegration] = None


async def get_typing_integration(
    anti_ban_manager: AntiBanManager,
    session_manager: SessionManager,
    personality_manager: PersonalityManager
) -> EnhancedTypingIntegration:
    """Get or create typing integration instance."""
    global typing_integration
    
    if typing_integration is None:
        typing_integration = EnhancedTypingIntegration(
            anti_ban_manager=anti_ban_manager,
            session_manager=session_manager,
            personality_manager=personality_manager
        )
        await typing_integration.initialize()
    
    return typing_integration


async def cleanup_typing_integration() -> None:
    """Cleanup global typing integration."""
    global typing_integration
    
    if typing_integration:
        await typing_integration.cleanup()
        typing_integration = None