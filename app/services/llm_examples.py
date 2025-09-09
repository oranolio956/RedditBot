"""
LLM Integration Examples

Complete examples showing how to use the LLM integration system with
the Telegram bot, personality engine, and conversation management.

This file provides:
- Setup and configuration examples
- Basic and advanced usage patterns
- Integration with existing bot handlers
- Error handling and fallback scenarios
- Performance optimization examples
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
from aiogram import Bot
from aiogram.types import Message, User as TelegramUser
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_integration import (
    LLMIntegrationService, ResponseGenerationRequest, get_llm_integration_service
)
from app.services.llm_config import get_llm_config
from app.models.user import User
from app.models.conversation import ConversationSession

logger = structlog.get_logger(__name__)


class LLMExampleService:
    """
    Example service demonstrating LLM integration patterns.
    
    Shows how to:
    - Initialize and configure the LLM system
    - Handle different conversation scenarios
    - Integrate with existing bot infrastructure
    - Implement error handling and fallbacks
    - Optimize for performance and cost
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.llm_integration: Optional[LLMIntegrationService] = None
        
    async def initialize(self):
        """Initialize the example service."""
        try:
            # Initialize LLM integration
            self.llm_integration = await get_llm_integration_service(self.db)
            
            # Validate configuration
            config = get_llm_config()
            validation = config.validate_configuration()
            
            if not validation['valid']:
                logger.error("LLM configuration invalid", errors=validation['errors'])
                raise ValueError("Invalid LLM configuration")
            
            logger.info("LLM example service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize LLM example service", error=str(e))
            raise
    
    async def basic_conversation_example(
        self, 
        user_message: str, 
        user_id: int,
        chat_id: int
    ) -> str:
        """
        Basic example: Generate a simple AI response.
        
        This is the simplest way to get an AI response with full integration.
        """
        try:
            # Create request
            request = ResponseGenerationRequest(
                user_message=user_message,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=str(uuid.uuid4()),  # Generate unique conversation ID
                response_style='conversational',
                use_personality=True,
                use_conversation_memory=True
            )
            
            # Generate response
            result = await self.llm_integration.generate_response(request)
            
            logger.info(
                "Generated basic response",
                model=result.model_used,
                tokens=result.tokens_used.get('total', 0),
                cost=result.cost_estimate,
                response_time=result.response_time_ms
            )
            
            return result.response_content
            
        except Exception as e:
            logger.error("Error in basic conversation example", error=str(e))
            return "I'm here to help! How can I assist you today?"
    
    async def personality_aware_example(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        personality_style: str = "friendly"
    ) -> str:
        """
        Example: Generate response with specific personality adaptation.
        
        Shows how personality affects response style and content.
        """
        try:
            # Create request with personality preferences
            request = ResponseGenerationRequest(
                user_message=user_message,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=f"personality_demo_{user_id}",
                response_style=personality_style,
                use_personality=True,
                use_conversation_memory=True,
                system_instructions=f"Adapt your personality to be {personality_style} and engaging."
            )
            
            result = await self.llm_integration.generate_response(request)
            
            logger.info(
                "Generated personality-aware response",
                personality_style=personality_style,
                personality_applied=result.personality_applied,
                model=result.model_used
            )
            
            return result.response_content
            
        except Exception as e:
            logger.error("Error in personality-aware example", error=str(e))
            return "I'm here to help with a friendly attitude!"
    
    async def conversation_memory_example(
        self,
        messages: List[Dict[str, str]],  # [{'role': 'user', 'content': '...'}, ...]
        user_id: int,
        chat_id: int
    ) -> str:
        """
        Example: Multi-turn conversation with memory.
        
        Shows how the system maintains context across multiple messages.
        """
        try:
            conversation_id = f"memory_demo_{user_id}"
            
            # Simulate adding multiple messages to conversation
            for i, msg in enumerate(messages[:-1]):  # All but the last message
                request = ResponseGenerationRequest(
                    user_message=msg['content'],
                    user_id=str(user_id),
                    chat_id=str(chat_id),
                    conversation_id=conversation_id,
                    use_conversation_memory=True,
                    use_personality=True
                )
                
                # Generate intermediate responses to build context
                await self.llm_integration.generate_response(request)
                
                logger.debug(f"Processed message {i+1}/{len(messages)-1} for context building")
            
            # Generate final response with full context
            final_message = messages[-1]
            request = ResponseGenerationRequest(
                user_message=final_message['content'],
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=conversation_id,
                use_conversation_memory=True,
                use_personality=True
            )
            
            result = await self.llm_integration.generate_response(request)
            
            logger.info(
                "Generated memory-aware response",
                conversation_length=len(messages),
                context_tokens=result.context_tokens_used,
                conversation_phase=result.conversation_phase
            )
            
            return result.response_content
            
        except Exception as e:
            logger.error("Error in conversation memory example", error=str(e))
            return "I remember our conversation and I'm here to continue helping!"
    
    async def streaming_response_example(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        bot: Bot,
        telegram_message: Message
    ) -> str:
        """
        Example: Generate streaming response with typing indicators.
        
        Shows real-time response generation for better user experience.
        """
        try:
            request = ResponseGenerationRequest(
                user_message=user_message,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=f"streaming_{user_id}",
                use_streaming=True,
                enable_typing_simulation=True,
                use_personality=True
            )
            
            # Collect streaming response
            response_parts = []
            async for chunk in self.llm_integration.generate_streaming_response(
                request, bot, telegram_message
            ):
                response_parts.append(chunk)
                # In a real implementation, you might send partial updates to the user
            
            full_response = ''.join(response_parts)
            
            logger.info(
                "Generated streaming response",
                chunks=len(response_parts),
                total_length=len(full_response)
            )
            
            return full_response
            
        except Exception as e:
            logger.error("Error in streaming response example", error=str(e))
            return "I'm processing your request..."
    
    async def cost_optimized_example(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        priority: str = "low"
    ) -> str:
        """
        Example: Cost-optimized response generation.
        
        Shows how to balance cost and quality based on use case priority.
        """
        try:
            # Configure request based on priority
            if priority == "high":
                response_style = "helpful"
                max_tokens = 4000
                priority_level = 1
            elif priority == "medium":
                response_style = "conversational"
                max_tokens = 2000
                priority_level = 2
            else:  # low priority
                response_style = "casual"
                max_tokens = 1000
                priority_level = 3
            
            request = ResponseGenerationRequest(
                user_message=user_message,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=f"cost_opt_{user_id}",
                response_style=response_style,
                max_response_length=max_tokens,
                use_personality=priority != "low",  # Disable for low priority
                use_conversation_memory=priority == "high"  # Only for high priority
            )
            
            result = await self.llm_integration.generate_response(request)
            
            logger.info(
                "Generated cost-optimized response",
                priority=priority,
                model=result.model_used,
                cost=result.cost_estimate,
                tokens=result.tokens_used.get('total', 0)
            )
            
            return result.response_content
            
        except Exception as e:
            logger.error("Error in cost-optimized example", error=str(e))
            return "I'll help you efficiently!"
    
    async def error_handling_example(
        self,
        user_message: str,
        user_id: int,
        chat_id: int
    ) -> Dict[str, Any]:
        """
        Example: Comprehensive error handling and fallbacks.
        
        Shows how the system gracefully handles various error conditions.
        """
        result = {
            'response': '',
            'success': False,
            'error_type': None,
            'fallback_used': False,
            'metadata': {}
        }
        
        try:
            request = ResponseGenerationRequest(
                user_message=user_message,
                user_id=str(user_id),
                chat_id=str(chat_id),
                conversation_id=f"error_demo_{user_id}",
                use_personality=True,
                use_conversation_memory=True
            )
            
            response = await self.llm_integration.generate_response(request)
            
            result.update({
                'response': response.response_content,
                'success': not response.had_errors,
                'error_type': None if not response.had_errors else 'llm_error',
                'fallback_used': response.fallback_used,
                'metadata': {
                    'model_used': response.model_used,
                    'response_time_ms': response.response_time_ms,
                    'cost_estimate': response.cost_estimate,
                    'tokens_used': response.tokens_used,
                    'personality_applied': response.personality_applied
                }
            })
            
        except Exception as e:
            logger.error("Error in error handling example", error=str(e))
            
            # Demonstrate fallback handling
            result.update({
                'response': "I apologize, but I'm having some technical difficulties. I'm still here to help in any way I can!",
                'success': False,
                'error_type': 'system_error',
                'fallback_used': True,
                'metadata': {'error_details': str(e)}
            })
        
        return result
    
    async def telegram_handler_integration_example(
        self,
        telegram_message: Message,
        bot: Bot,
        conversation_session: Optional[ConversationSession] = None
    ) -> str:
        """
        Example: Complete integration with Telegram bot handlers.
        
        Shows how to integrate LLM responses into existing Telegram bot infrastructure.
        """
        try:
            # Extract information from Telegram message
            user_id = telegram_message.from_user.id
            chat_id = telegram_message.chat.id
            message_text = telegram_message.text or "Hello"
            
            # Use the integration service's Telegram helper
            response, metadata = await self.llm_integration.create_telegram_integration(
                user_id=user_id,
                chat_id=chat_id,
                message_text=message_text,
                bot=bot,
                message=telegram_message,
                conversation_session=conversation_session
            )
            
            # Log integration metrics
            logger.info(
                "Telegram integration complete",
                user_id=user_id,
                chat_id=chat_id,
                model=metadata.get('model_used'),
                response_time=metadata.get('response_time_ms'),
                personality_applied=metadata.get('personality_applied'),
                tokens=metadata.get('tokens_used', {}).get('total', 0)
            )
            
            return response
            
        except Exception as e:
            logger.error("Error in Telegram handler integration", error=str(e))
            return "I'm here to help! How can I assist you today?"
    
    async def performance_analysis_example(self, test_messages: List[str]) -> Dict[str, Any]:
        """
        Example: Performance analysis and optimization.
        
        Demonstrates how to analyze and optimize LLM integration performance.
        """
        results = {
            'total_messages': len(test_messages),
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0.0,
            'total_cost': 0.0,
            'model_usage': {},
            'personality_adaptations': 0,
            'cache_hits': 0
        }
        
        response_times = []
        
        try:
            for i, message in enumerate(test_messages):
                start_time = datetime.now()
                
                request = ResponseGenerationRequest(
                    user_message=message,
                    user_id="performance_test",
                    chat_id="test_chat",
                    conversation_id=f"perf_test_{i}",
                    use_personality=True,
                    use_conversation_memory=True
                )
                
                try:
                    response = await self.llm_integration.generate_response(request)
                    
                    # Collect metrics
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    response_times.append(response_time)
                    
                    results['successful_responses'] += 1
                    results['total_cost'] += response.cost_estimate
                    
                    model = response.model_used
                    if model in results['model_usage']:
                        results['model_usage'][model] += 1
                    else:
                        results['model_usage'][model] = 1
                    
                    if response.personality_applied:
                        results['personality_adaptations'] += 1
                    
                    # Note: Cache hit detection would need to be implemented
                    # in the actual LLM service
                    
                except Exception as e:
                    logger.error(f"Error processing message {i}", error=str(e))
                    results['failed_responses'] += 1
            
            # Calculate averages
            if response_times:
                results['average_response_time'] = sum(response_times) / len(response_times)
            
            # Get service statistics
            service_stats = await self.llm_integration.get_service_stats()
            results['service_stats'] = service_stats
            
            logger.info("Performance analysis complete", **results)
            
        except Exception as e:
            logger.error("Error in performance analysis", error=str(e))
            results['error'] = str(e)
        
        return results
    
    async def configuration_example(self) -> Dict[str, Any]:
        """
        Example: Configuration management and validation.
        
        Shows how to manage LLM service configuration.
        """
        try:
            config = get_llm_config()
            
            # Get configuration summary
            summary = config.get_configuration_summary()
            
            # Validate configuration
            validation = config.validate_configuration()
            
            # Example: Update provider quota
            available_providers = config.get_available_providers()
            if available_providers:
                provider = available_providers[0]
                logger.info(f"Updating quota for provider: {provider}")
                # config.update_provider_quota(provider, daily_quota=200.0, hourly_quota=50.0)
            
            # Get preferred provider
            preferred = config.get_preferred_provider()
            
            result = {
                'configuration_summary': summary,
                'validation': validation,
                'available_providers': available_providers,
                'preferred_provider': preferred
            }
            
            logger.info("Configuration example complete", 
                       providers=len(available_providers),
                       valid=validation['valid'])
            
            return result
            
        except Exception as e:
            logger.error("Error in configuration example", error=str(e))
            return {'error': str(e)}


# Utility functions for common use cases

async def simple_ai_response(
    user_message: str,
    user_id: int,
    chat_id: int,
    db_session: AsyncSession
) -> str:
    """
    Simple utility function for getting an AI response.
    
    Use this for basic AI responses without complex configuration.
    """
    try:
        service = await get_llm_integration_service(db_session)
        
        response, _ = await service.create_telegram_integration(
            user_id=user_id,
            chat_id=chat_id,
            message_text=user_message,
            bot=None,  # Bot not required for basic responses
            message=None,
            conversation_session=None
        )
        
        return response
        
    except Exception as e:
        logger.error("Error in simple AI response", error=str(e))
        return "I'm here to help! Could you tell me more about what you need?"


async def personality_response(
    user_message: str,
    user_id: int,
    chat_id: int,
    personality_style: str,
    db_session: AsyncSession
) -> str:
    """
    Utility function for personality-specific responses.
    
    Args:
        personality_style: 'friendly', 'professional', 'casual', 'helpful'
    """
    try:
        service = await get_llm_integration_service(db_session)
        
        request = ResponseGenerationRequest(
            user_message=user_message,
            user_id=str(user_id),
            chat_id=str(chat_id),
            conversation_id=f"{personality_style}_{user_id}",
            response_style=personality_style,
            use_personality=True
        )
        
        result = await service.generate_response(request)
        return result.response_content
        
    except Exception as e:
        logger.error("Error in personality response", error=str(e))
        return f"I'm here to help you in a {personality_style} way!"


async def contextual_response(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    user_id: int,
    chat_id: int,
    db_session: AsyncSession
) -> str:
    """
    Utility function for context-aware responses.
    
    Args:
        conversation_history: List of {'role': 'user'/'assistant', 'content': '...'}
    """
    try:
        service = await get_llm_integration_service(db_session)
        
        request = ResponseGenerationRequest(
            user_message=user_message,
            user_id=str(user_id),
            chat_id=str(chat_id),
            conversation_id=f"contextual_{user_id}",
            message_history=conversation_history,
            use_conversation_memory=True,
            use_personality=True
        )
        
        result = await service.generate_response(request)
        return result.response_content
        
    except Exception as e:
        logger.error("Error in contextual response", error=str(e))
        return "I understand the context of our conversation and I'm here to help!"


# Export main classes and utilities
__all__ = [
    'LLMExampleService',
    'simple_ai_response',
    'personality_response',
    'contextual_response'
]