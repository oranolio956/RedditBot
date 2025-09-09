"""
LLM Integration Service

Integrates LLM capabilities with the Telegram bot system, personality engine, 
and conversation management. Provides a unified interface for generating 
AI-powered responses with full context awareness.

Features:
- Seamless integration with existing Telegram handlers
- Personality-aware response generation
- Conversation context management
- Response streaming with typing indicators
- Error handling and fallbacks
- Cost optimization and rate limiting
- Performance monitoring and analytics
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
import logging

import structlog
from aiogram import Bot
from aiogram.types import Message, Chat
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.models.user import User
from app.models.conversation import (
    ConversationSession, Message as ConversationMessage,
    MessageType, MessageDirection
)
from app.services.llm_service import (
    LLMService, LLMRequest, LLMResponse, LLMModel, get_llm_service
)
from app.services.conversation_manager import (
    ConversationManager, ConversationMemory, ConversationPhase
)
from app.services.personality_engine import (
    AdvancedPersonalityEngine, ConversationContext, PersonalityState
)
from app.services.typing_simulator import TypingSimulator

logger = structlog.get_logger(__name__)


@dataclass
class ResponseGenerationRequest:
    """Request for AI response generation."""
    user_message: str
    user_id: str
    chat_id: str
    conversation_id: str
    
    # Optional context
    message_history: Optional[List[Dict[str, Any]]] = None
    system_instructions: Optional[str] = None
    response_style: str = "conversational"  # conversational, formal, casual, helpful
    max_response_length: Optional[int] = None
    
    # Integration settings
    use_personality: bool = True
    use_conversation_memory: bool = True
    use_streaming: bool = False
    enable_typing_simulation: bool = True
    
    # Metadata
    message_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResponseGenerationResult:
    """Result of AI response generation."""
    response_content: str
    generation_metadata: Dict[str, Any]
    
    # Quality metrics
    response_time_ms: int
    cost_estimate: float
    model_used: str
    tokens_used: Dict[str, int]
    
    # Context information
    personality_applied: bool = False
    conversation_phase: Optional[str] = None
    context_tokens_used: int = 0
    
    # Error information
    had_errors: bool = False
    error_details: Optional[str] = None
    fallback_used: bool = False


class LLMIntegrationService:
    """
    Main integration service that coordinates LLM responses with all bot systems.
    
    This service acts as the primary interface for generating AI responses,
    handling all the complexity of integrating with:
    - LLM providers (OpenAI, Anthropic)
    - Personality engine for adaptive responses  
    - Conversation manager for context
    - Typing simulator for natural feel
    - Database for persistence
    - Error handling and fallbacks
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings()
        
        # Core services
        self.llm_service: Optional[LLMService] = None
        self.conversation_manager: Optional[ConversationManager] = None  
        self.personality_engine: Optional[AdvancedPersonalityEngine] = None
        self.typing_simulator: Optional[TypingSimulator] = None
        
        # Performance tracking
        self.response_stats = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'avg_response_time': 0.0,
            'total_cost': 0.0,
            'personality_adaptations': 0,
            'context_optimizations': 0
        }
        
        # System prompts for different response styles
        self.system_prompts = {
            'conversational': (
                "You are a friendly, helpful AI assistant engaging in natural conversation. "
                "Be personable, empathetic, and adapt your communication style to match the user's needs. "
                "Provide helpful, accurate information while maintaining a warm, conversational tone."
            ),
            'formal': (
                "You are a professional AI assistant providing precise, well-structured responses. "
                "Use formal language and maintain a respectful, business-appropriate tone. "
                "Focus on accuracy, clarity, and comprehensive information."
            ),
            'casual': (
                "You're a relaxed, friendly AI that chats naturally with users. "
                "Keep things light, use casual language, and don't be too formal. "
                "Be helpful but in a laid-back, approachable way."
            ),
            'helpful': (
                "You are a dedicated AI assistant focused on providing maximum value to users. "
                "Be thorough in your explanations, anticipate follow-up questions, "
                "and always aim to fully resolve the user's needs."
            )
        }
        
    async def initialize(self):
        """Initialize the LLM integration service."""
        try:
            logger.info("Initializing LLM integration service...")
            
            # Initialize core services
            self.llm_service = await get_llm_service()
            
            self.conversation_manager = ConversationManager(self.db)
            await self.conversation_manager.initialize()
            
            self.personality_engine = AdvancedPersonalityEngine(
                self.db, 
                await self.conversation_manager.redis  # Reuse Redis client
            )
            await self.personality_engine.initialize_models()
            
            # Initialize typing simulator
            try:
                from app.services.typing_simulator import TypingSimulator
                self.typing_simulator = TypingSimulator()
            except ImportError:
                logger.warning("Typing simulator not available, will use fallback")
                self.typing_simulator = None
            
            logger.info("LLM integration service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize LLM integration service", error=str(e))
            raise
    
    async def generate_response(
        self, 
        request: ResponseGenerationRequest
    ) -> ResponseGenerationResult:
        """
        Generate an AI response with full integration.
        
        This is the main method that orchestrates:
        1. Context retrieval and management
        2. Personality analysis and adaptation
        3. LLM response generation
        4. Response post-processing
        5. Metrics and analytics tracking
        """
        start_time = time.time()
        
        try:
            self.response_stats['total_requests'] += 1
            
            # Step 1: Get or create conversation context
            conversation_memory = None
            if request.use_conversation_memory:
                conversation_memory = await self.conversation_manager.get_conversation_context(
                    request.user_id,
                    request.conversation_id,
                    max_tokens=3500  # Leave room for response
                )
            
            # Step 2: Analyze user personality if enabled
            personality_state = None
            if request.use_personality and self.personality_engine:
                personality_state = await self._analyze_and_adapt_personality(
                    request, conversation_memory
                )
            
            # Step 3: Generate response
            if request.use_streaming:
                # For streaming, we'll handle this differently
                return await self._generate_streaming_response(
                    request, conversation_memory, personality_state
                )
            else:
                response_content, metadata = await self._generate_standard_response(
                    request, conversation_memory, personality_state
                )
            
            # Step 4: Post-process response
            if personality_state and response_content:
                response_content = await self._apply_final_personality_touches(
                    response_content, personality_state, request
                )
            
            # Step 5: Save to conversation memory
            if conversation_memory:
                await self.conversation_manager.add_message_to_context(
                    conversation_memory,
                    response_content,
                    'assistant',
                    metadata={
                        'model_used': metadata.get('model_used'),
                        'cost': metadata.get('cost_estimate', 0.0),
                        'generation_time': time.time() - start_time
                    }
                )
            
            # Step 6: Update statistics
            response_time = int((time.time() - start_time) * 1000)
            self._update_response_stats(metadata, response_time, success=True)
            
            # Step 7: Record analytics
            await self._record_response_analytics(request, metadata, response_time)
            
            return ResponseGenerationResult(
                response_content=response_content,
                generation_metadata=metadata,
                response_time_ms=response_time,
                cost_estimate=metadata.get('cost_estimate', 0.0),
                model_used=metadata.get('model_used', 'unknown'),
                tokens_used=metadata.get('tokens_used', {}),
                personality_applied=personality_state is not None,
                conversation_phase=metadata.get('conversation_phase'),
                context_tokens_used=metadata.get('context_tokens', 0),
                had_errors=False,
                fallback_used=metadata.get('fallback', False)
            )
            
        except Exception as e:
            logger.error("Error generating AI response", error=str(e))
            
            # Generate fallback response
            fallback_response = await self._generate_fallback_response(request)
            response_time = int((time.time() - start_time) * 1000)
            
            self.response_stats['failed_responses'] += 1
            
            return ResponseGenerationResult(
                response_content=fallback_response,
                generation_metadata={'error': str(e)},
                response_time_ms=response_time,
                cost_estimate=0.0,
                model_used='fallback',
                tokens_used={'total': len(fallback_response.split())},
                had_errors=True,
                error_details=str(e),
                fallback_used=True
            )
    
    async def generate_streaming_response(
        self,
        request: ResponseGenerationRequest,
        bot: Bot,
        message: Message
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response with typing indicators.
        
        This provides real-time response generation with:
        - Realistic typing simulation
        - Chunk-by-chunk response delivery
        - Personality adaptation per chunk
        - Error handling during streaming
        """
        try:
            # Get conversation context
            conversation_memory = None
            if request.use_conversation_memory:
                conversation_memory = await self.conversation_manager.get_conversation_context(
                    request.user_id,
                    request.conversation_id
                )
            
            # Get personality state
            personality_state = None
            if request.use_personality:
                personality_state = await self._analyze_and_adapt_personality(
                    request, conversation_memory
                )
            
            # Start typing simulation if enabled
            typing_task = None
            if request.enable_typing_simulation and self.typing_simulator:
                typing_task = asyncio.create_task(
                    self._simulate_typing_for_streaming(bot, message.chat.id, request.user_message)
                )
            
            try:
                # Generate streaming response
                full_response = ""
                async for chunk in self.conversation_manager.generate_streaming_response(
                    conversation_memory,
                    request.user_message,
                    personality_state
                ):
                    full_response += chunk
                    yield chunk
                
                # Save complete response to conversation memory
                if conversation_memory and full_response:
                    await self.conversation_manager.add_message_to_context(
                        conversation_memory,
                        full_response,
                        'assistant'
                    )
                
            finally:
                # Clean up typing simulation
                if typing_task:
                    typing_task.cancel()
                    try:
                        await typing_task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            logger.error("Error in streaming response generation", error=str(e))
            fallback = await self._generate_fallback_response(request)
            yield fallback
    
    async def _analyze_and_adapt_personality(
        self,
        request: ResponseGenerationRequest,
        conversation_memory: Optional[ConversationMemory]
    ) -> Optional[PersonalityState]:
        """Analyze user personality and adapt bot personality accordingly."""
        try:
            if not self.personality_engine or not conversation_memory:
                return None
            
            # Create conversation context for personality analysis
            context = ConversationContext(
                user_id=request.user_id,
                session_id=request.conversation_id,
                message_history=conversation_memory.messages[-20:],  # Last 20 messages
                current_sentiment=0.0,  # Will be calculated
                conversation_phase=conversation_memory.current_phase.value
            )
            
            # Calculate current sentiment from recent messages
            if conversation_memory.sentiment_trend:
                context.current_sentiment = sum(conversation_memory.sentiment_trend[-3:]) / len(conversation_memory.sentiment_trend[-3:])
            
            # Analyze user personality
            user_traits = await self.personality_engine.analyze_user_personality(
                request.user_id, context
            )
            
            # Get default personality profile (you might want to make this configurable)
            from app.models.personality import PersonalityProfile
            from sqlalchemy import select
            
            query = select(PersonalityProfile).where(PersonalityProfile.is_default == True)
            result = await self.db.execute(query)
            default_profile = result.scalar_one_or_none()
            
            if not default_profile:
                logger.warning("No default personality profile found")
                return None
            
            # Adapt personality to user
            personality_state = await self.personality_engine.adapt_personality(
                request.user_id,
                user_traits,
                default_profile,
                context
            )
            
            self.response_stats['personality_adaptations'] += 1
            
            logger.debug(
                "Applied personality adaptation",
                user_id=request.user_id,
                confidence=personality_state.confidence_level,
                adaptation_count=len(personality_state.adaptation_history)
            )
            
            return personality_state
            
        except Exception as e:
            logger.error("Error analyzing and adapting personality", error=str(e))
            return None
    
    async def _generate_standard_response(
        self,
        request: ResponseGenerationRequest,
        conversation_memory: Optional[ConversationMemory],
        personality_state: Optional[PersonalityState]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate standard (non-streaming) response."""
        try:
            if conversation_memory and self.conversation_manager:
                # Use conversation manager for contextual response
                response_content, metadata = await self.conversation_manager.generate_contextual_response(
                    conversation_memory,
                    request.user_message,
                    personality_state,
                    self.system_prompts.get(request.response_style, self.system_prompts['conversational'])
                )
                return response_content, metadata
            else:
                # Direct LLM call without conversation context
                llm_request = LLMRequest(
                    messages=[{'role': 'user', 'content': request.user_message}],
                    system_prompt=self.system_prompts.get(request.response_style),
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    personality_context=personality_state,
                    max_tokens=request.max_response_length,
                    use_cache=True
                )
                
                response = await self.llm_service.generate_response(llm_request)
                
                metadata = {
                    'model_used': response.model_used.value,
                    'cost_estimate': response.cost_estimate,
                    'tokens_used': response.tokens_used,
                    'response_time_ms': response.response_time_ms,
                    'quality_score': response.quality_score
                }
                
                return response.content, metadata
                
        except Exception as e:
            logger.error("Error generating standard response", error=str(e))
            raise
    
    async def _generate_streaming_response(
        self,
        request: ResponseGenerationRequest,
        conversation_memory: Optional[ConversationMemory],
        personality_state: Optional[PersonalityState]
    ) -> ResponseGenerationResult:
        """Handle streaming response generation (for non-real-time use)."""
        try:
            # Collect streaming response
            response_chunks = []
            if conversation_memory:
                async for chunk in self.conversation_manager.generate_streaming_response(
                    conversation_memory,
                    request.user_message,
                    personality_state
                ):
                    response_chunks.append(chunk)
            else:
                # Direct streaming from LLM service
                llm_request = LLMRequest(
                    messages=[{'role': 'user', 'content': request.user_message}],
                    system_prompt=self.system_prompts.get(request.response_style),
                    personality_context=personality_state,
                    stream=True
                )
                
                async for chunk in self.llm_service.generate_streaming_response(llm_request):
                    response_chunks.append(chunk)
            
            response_content = ''.join(response_chunks)
            
            # Create metadata
            metadata = {
                'model_used': 'streaming',
                'cost_estimate': len(response_content) * 0.00002,  # Rough estimate
                'tokens_used': {'total': len(response_content.split())},
                'streaming': True
            }
            
            return response_content, metadata
            
        except Exception as e:
            logger.error("Error generating streaming response", error=str(e))
            raise
    
    async def _apply_final_personality_touches(
        self,
        response: str,
        personality_state: PersonalityState,
        request: ResponseGenerationRequest
    ) -> str:
        """Apply final personality-based modifications to response."""
        try:
            if not personality_state:
                return response
            
            # Use personality engine to modify response
            modified_response = await self.personality_engine.generate_personality_response(
                personality_state,
                ConversationContext(
                    user_id=request.user_id,
                    session_id=request.conversation_id,
                    message_history=[{'content': request.user_message}],
                    current_sentiment=0.0
                ),
                response
            )
            
            return modified_response
            
        except Exception as e:
            logger.error("Error applying personality touches", error=str(e))
            return response
    
    async def _generate_fallback_response(self, request: ResponseGenerationRequest) -> str:
        """Generate fallback response when AI systems fail."""
        try:
            # Context-aware fallback responses
            user_message_lower = request.user_message.lower()
            
            if any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
                return "Hello! I'm here to help. How can I assist you today?"
            
            elif '?' in request.user_message:
                return "That's a great question! I'd like to help you find the answer. Could you provide a bit more context?"
            
            elif any(word in user_message_lower for word in ['help', 'assist', 'support']):
                return "I'm here to help! Please let me know what specific assistance you need, and I'll do my best to support you."
            
            elif any(word in user_message_lower for word in ['thank', 'thanks']):
                return "You're very welcome! Is there anything else I can help you with?"
            
            elif any(word in user_message_lower for word in ['sorry', 'apologize']):
                return "No need to apologize! I'm here to help. What can I do for you?"
            
            else:
                fallback_responses = [
                    "I understand what you're saying. Could you help me understand exactly how I can assist you?",
                    "I'm here and ready to help! What specific information or assistance are you looking for?",
                    "Thank you for reaching out. I'd like to make sure I give you the most helpful response possible. Could you share a bit more detail?",
                    "I appreciate you taking the time to message me. How can I best support you today?"
                ]
                
                # Use hash of message for consistent selection
                import hashlib
                message_hash = hashlib.md5(request.user_message.encode()).hexdigest()
                index = int(message_hash[:8], 16) % len(fallback_responses)
                return fallback_responses[index]
                
        except Exception as e:
            logger.error("Error generating fallback response", error=str(e))
            return "I'm here to help! How can I assist you?"
    
    async def _simulate_typing_for_streaming(
        self, 
        bot: Bot, 
        chat_id: str, 
        trigger_message: str
    ):
        """Simulate typing indicators during streaming response."""
        try:
            if not self.typing_simulator:
                return
            
            # Calculate realistic typing duration based on response complexity
            estimated_response_length = min(500, len(trigger_message) * 2)  # Rough estimate
            typing_duration = self.typing_simulator.calculate_typing_duration(
                estimated_response_length,
                complexity_factor=1.2  # Slightly more complex for AI responses
            )
            
            # Send typing indicators
            start_time = time.time()
            while time.time() - start_time < typing_duration:
                await bot.send_chat_action(chat_id, "typing")
                await asyncio.sleep(3)  # Typing action lasts ~3 seconds
                
        except Exception as e:
            logger.error("Error simulating typing for streaming", error=str(e))
    
    def _update_response_stats(self, metadata: Dict[str, Any], response_time: int, success: bool):
        """Update internal response statistics."""
        try:
            if success:
                self.response_stats['successful_responses'] += 1
                
                # Update average response time
                total_responses = self.response_stats['successful_responses']
                current_avg = self.response_stats['avg_response_time']
                self.response_stats['avg_response_time'] = (
                    (current_avg * (total_responses - 1) + response_time) / total_responses
                )
                
                # Update total cost
                cost = metadata.get('cost_estimate', 0.0)
                self.response_stats['total_cost'] += cost
            else:
                self.response_stats['failed_responses'] += 1
                
        except Exception as e:
            logger.error("Error updating response stats", error=str(e))
    
    async def _record_response_analytics(
        self, 
        request: ResponseGenerationRequest,
        metadata: Dict[str, Any], 
        response_time: int
    ):
        """Record detailed analytics for response generation."""
        try:
            # This could be expanded to record to analytics database
            analytics_data = {
                'timestamp': datetime.now().isoformat(),
                'user_id': request.user_id,
                'conversation_id': request.conversation_id,
                'response_style': request.response_style,
                'model_used': metadata.get('model_used'),
                'response_time_ms': response_time,
                'tokens_used': metadata.get('tokens_used', {}),
                'cost_estimate': metadata.get('cost_estimate', 0.0),
                'personality_applied': request.use_personality,
                'conversation_memory_used': request.use_conversation_memory,
                'streaming_used': request.use_streaming,
                'conversation_phase': metadata.get('conversation_phase'),
                'quality_score': metadata.get('quality_score')
            }
            
            # Could send to analytics service, save to database, etc.
            logger.debug("Response analytics recorded", **analytics_data)
            
        except Exception as e:
            logger.error("Error recording response analytics", error=str(e))
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        try:
            stats = self.response_stats.copy()
            
            # Add derived metrics
            if stats['total_requests'] > 0:
                stats['success_rate'] = stats['successful_responses'] / stats['total_requests']
                stats['failure_rate'] = stats['failed_responses'] / stats['total_requests']
            
            # Add service status
            stats['llm_service_available'] = self.llm_service is not None
            stats['conversation_manager_available'] = self.conversation_manager is not None
            stats['personality_engine_available'] = self.personality_engine is not None
            stats['typing_simulator_available'] = self.typing_simulator is not None
            
            # Add LLM service stats if available
            if self.llm_service:
                llm_stats = await self.llm_service.get_usage_stats()
                stats['llm_usage'] = llm_stats
            
            return stats
            
        except Exception as e:
            logger.error("Error getting service stats", error=str(e))
            return self.response_stats
    
    async def create_telegram_integration(
        self,
        user_id: int,
        chat_id: int,
        message_text: str,
        bot: Bot,
        message: Message,
        conversation_session: Optional[ConversationSession] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a complete integration with Telegram bot handlers.
        
        This method provides a simple interface for Telegram handlers to get
        AI-powered responses with full context and personality integration.
        """
        try:
            # Create request
            request = ResponseGenerationRequest(
                user_message=message_text,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=str(conversation_session.id) if conversation_session else str(uuid.uuid4()),
                response_style='conversational',
                use_personality=True,
                use_conversation_memory=True,
                enable_typing_simulation=True
            )
            
            # Generate response
            result = await self.generate_response(request)
            
            # Return response and metadata for handlers
            return result.response_content, {
                'model_used': result.model_used,
                'response_time_ms': result.response_time_ms,
                'cost_estimate': result.cost_estimate,
                'personality_applied': result.personality_applied,
                'conversation_phase': result.conversation_phase,
                'tokens_used': result.tokens_used,
                'had_errors': result.had_errors,
                'fallback_used': result.fallback_used
            }
            
        except Exception as e:
            logger.error("Error in Telegram integration", error=str(e))
            return "I'm here to help! How can I assist you?", {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the integration service."""
        try:
            if self.conversation_manager:
                await self.conversation_manager.shutdown()
            
            if self.llm_service:
                await self.llm_service.shutdown()
            
            logger.info("LLM integration service shutdown complete")
            
        except Exception as e:
            logger.error("Error during LLM integration shutdown", error=str(e))


# Global instance
llm_integration_service: Optional[LLMIntegrationService] = None


async def get_llm_integration_service(db_session: AsyncSession) -> LLMIntegrationService:
    """Get the global LLM integration service instance."""
    global llm_integration_service
    if llm_integration_service is None:
        llm_integration_service = LLMIntegrationService(db_session)
        await llm_integration_service.initialize()
    return llm_integration_service


# Export main classes
__all__ = [
    'LLMIntegrationService',
    'ResponseGenerationRequest',
    'ResponseGenerationResult',
    'get_llm_integration_service'
]