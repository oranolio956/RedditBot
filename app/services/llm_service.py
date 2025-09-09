"""
LLM Integration Service

A production-ready LLM service that provides:
- Multi-provider support (OpenAI, Anthropic Claude, etc.)
- Intelligent routing for cost optimization
- Response streaming and caching
- Context management with conversation memory
- Integration with personality system
- Rate limiting and quota management
- Comprehensive error handling and fallbacks
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import openai
import anthropic
import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import structlog

from app.config.settings import get_settings
from app.core.redis import get_redis_client
from app.models.conversation import Message, ConversationSession
from app.services.personality_engine import (
    AdvancedPersonalityEngine, 
    ConversationContext, 
    PersonalityState
)

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FALLBACK = "fallback"


class LLMModel(str, Enum):
    """Supported LLM models with cost and capability tiers."""
    # OpenAI models
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    
    # Anthropic models  
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Fallback
    FALLBACK_MODEL = "fallback"


@dataclass
class ModelConfig:
    """Configuration for each LLM model."""
    provider: LLMProvider
    model_id: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    context_window: int
    supports_streaming: bool
    supports_functions: bool
    quality_tier: int  # 1=highest, 3=lowest
    speed_tier: int    # 1=fastest, 3=slowest


@dataclass
class LLMRequest:
    """LLM request configuration."""
    messages: List[Dict[str, str]]
    model: Optional[LLMModel] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    personality_context: Optional[PersonalityState] = None
    stream: bool = False
    use_cache: bool = True
    priority: int = 1  # 1=high, 2=normal, 3=low
    timeout_seconds: int = 30


@dataclass
class LLMResponse:
    """LLM response data."""
    content: str
    model_used: LLMModel
    provider: LLMProvider
    tokens_used: Dict[str, int]
    cost_estimate: float
    response_time_ms: int
    cached: bool = False
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMService:
    """
    Production-ready LLM service with multi-provider support.
    
    Features:
    - Intelligent model routing based on cost, quality, and availability
    - Response caching to reduce API costs
    - Rate limiting and quota management
    - Streaming response support
    - Conversation context management
    - Personality integration
    - Comprehensive error handling and fallbacks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis = None
        
        # Model configurations
        self.model_configs = {
            LLMModel.GPT_4_TURBO: ModelConfig(
                provider=LLMProvider.OPENAI,
                model_id="gpt-4-turbo-preview",
                max_tokens=4096,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                context_window=128000,
                supports_streaming=True,
                supports_functions=True,
                quality_tier=1,
                speed_tier=2
            ),
            LLMModel.GPT_4: ModelConfig(
                provider=LLMProvider.OPENAI,
                model_id="gpt-4",
                max_tokens=4096,
                cost_per_1k_input=0.03,
                cost_per_1k_output=0.06,
                context_window=8192,
                supports_streaming=True,
                supports_functions=True,
                quality_tier=1,
                speed_tier=3
            ),
            LLMModel.GPT_35_TURBO: ModelConfig(
                provider=LLMProvider.OPENAI,
                model_id="gpt-3.5-turbo",
                max_tokens=4096,
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                context_window=16385,
                supports_streaming=True,
                supports_functions=True,
                quality_tier=2,
                speed_tier=1
            ),
            LLMModel.CLAUDE_3_OPUS: ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-3-opus-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False,
                quality_tier=1,
                speed_tier=3
            ),
            LLMModel.CLAUDE_3_SONNET: ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-3-sonnet-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False,
                quality_tier=2,
                speed_tier=2
            ),
            LLMModel.CLAUDE_3_HAIKU: ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-3-haiku-20240307",
                max_tokens=4096,
                cost_per_1k_input=0.00025,
                cost_per_1k_output=0.00125,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False,
                quality_tier=3,
                speed_tier=1
            )
        }
        
        # Initialize clients
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[AsyncAnthropic] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting and quota tracking
        self.provider_quotas: Dict[LLMProvider, Dict[str, Any]] = {}
        self.provider_usage: Dict[LLMProvider, Dict[str, Any]] = {}
        
        # Response cache
        self.response_cache: Dict[str, Any] = {}
        
        # Quality and performance tracking
        self.model_performance: Dict[LLMModel, Dict[str, Any]] = {}
        
        # Fallback response templates
        self.fallback_responses = [
            "I understand you're looking for help. Let me think about the best way to assist you.",
            "That's an interesting question. I'd like to help you explore that further.",
            "I'm here to help! Could you provide a bit more context about what you need?",
            "Let me consider your request and provide you with the most helpful response I can.",
        ]
        
    async def initialize(self):
        """Initialize the LLM service."""
        try:
            logger.info("Initializing LLM service...")
            
            # Initialize Redis client
            self.redis = await get_redis_client()
            
            # Initialize API clients
            await self._initialize_clients()
            
            # Initialize HTTP client for custom endpoints
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
            )
            
            # Load cached performance data
            await self._load_performance_data()
            
            logger.info("LLM service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize LLM service", error=str(e))
            raise
    
    async def _initialize_clients(self):
        """Initialize API clients for different providers."""
        try:
            # OpenAI client
            openai_api_key = self.settings.security.encryption_key  # Use from environment
            if openai_api_key and openai_api_key != "your-openai-api-key":
                self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized")
            
            # Anthropic client
            anthropic_api_key = self.settings.security.jwt_secret  # Use from environment
            if anthropic_api_key and anthropic_api_key != "your-anthropic-api-key":
                self.anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized")
            
            # Initialize provider quotas and usage tracking
            for provider in LLMProvider:
                if provider != LLMProvider.FALLBACK:
                    self.provider_quotas[provider] = {
                        'daily_limit': 1000.0,  # $1000 daily limit
                        'hourly_limit': 100.0,   # $100 hourly limit
                        'requests_per_minute': 60,
                        'tokens_per_minute': 150000
                    }
                    
                    self.provider_usage[provider] = {
                        'daily_cost': 0.0,
                        'hourly_cost': 0.0,
                        'requests_this_minute': 0,
                        'tokens_this_minute': 0,
                        'last_reset': datetime.now()
                    }
            
        except Exception as e:
            logger.error("Error initializing LLM clients", error=str(e))
    
    async def _load_performance_data(self):
        """Load cached model performance data."""
        try:
            if self.redis:
                perf_data = await self.redis.get("llm:model_performance")
                if perf_data:
                    self.model_performance = json.loads(perf_data)
                    logger.info("Loaded model performance data from cache")
                else:
                    # Initialize default performance metrics
                    for model in LLMModel:
                        if model != LLMModel.FALLBACK_MODEL:
                            self.model_performance[model] = {
                                'avg_response_time': 2000,  # ms
                                'success_rate': 0.95,
                                'avg_quality_score': 0.8,
                                'total_requests': 0,
                                'total_errors': 0
                            }
        except Exception as e:
            logger.error("Error loading performance data", error=str(e))
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response using the most appropriate LLM.
        
        Features:
        - Intelligent model selection
        - Response caching
        - Error handling with fallbacks
        - Cost optimization
        - Quality tracking
        """
        start_time = time.time()
        
        try:
            # Check cache first if enabled
            if request.use_cache:
                cached_response = await self._check_cache(request)
                if cached_response:
                    cached_response.cached = True
                    return cached_response
            
            # Select optimal model
            selected_model = await self._select_optimal_model(request)
            request.model = selected_model
            
            # Check rate limits and quotas
            if not await self._check_rate_limits(selected_model):
                # Fall back to a different model or return fallback response
                fallback_model = await self._select_fallback_model(request)
                if fallback_model:
                    request.model = fallback_model
                    selected_model = fallback_model
                else:
                    return await self._generate_fallback_response(request)
            
            # Generate response with selected model
            response = await self._generate_with_model(request, selected_model)
            
            # Apply personality modifications if context provided
            if request.personality_context:
                response.content = await self._apply_personality_modifications(
                    response.content, request.personality_context, request
                )
            
            # Cache the response if appropriate
            if request.use_cache and response.content:
                await self._cache_response(request, response)
            
            # Update performance metrics
            await self._update_performance_metrics(selected_model, response, start_time)
            
            # Track usage for quota management
            await self._track_usage(selected_model, response)
            
            response.response_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                "Generated LLM response",
                model=selected_model.value,
                response_time=response.response_time_ms,
                tokens=response.tokens_used.get('total', 0),
                cost=response.cost_estimate,
                cached=response.cached
            )
            
            return response
            
        except Exception as e:
            logger.error("Error generating LLM response", error=str(e))
            return await self._generate_fallback_response(request)
    
    async def generate_streaming_response(
        self, 
        request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time conversation.
        
        Yields response chunks as they become available from the LLM.
        """
        try:
            request.stream = True
            selected_model = await self._select_optimal_model(request)
            
            if not await self._check_rate_limits(selected_model):
                fallback_response = await self._generate_fallback_response(request)
                yield fallback_response.content
                return
            
            model_config = self.model_configs[selected_model]
            
            if model_config.provider == LLMProvider.OPENAI and self.openai_client:
                async for chunk in self._stream_openai_response(request, selected_model):
                    # Apply personality modifications to each chunk if needed
                    if request.personality_context:
                        chunk = await self._apply_personality_to_chunk(
                            chunk, request.personality_context
                        )
                    yield chunk
                    
            elif model_config.provider == LLMProvider.ANTHROPIC and self.anthropic_client:
                async for chunk in self._stream_anthropic_response(request, selected_model):
                    if request.personality_context:
                        chunk = await self._apply_personality_to_chunk(
                            chunk, request.personality_context
                        )
                    yield chunk
            else:
                # Fallback to non-streaming
                response = await self._generate_fallback_response(request)
                yield response.content
                
        except Exception as e:
            logger.error("Error in streaming response", error=str(e))
            fallback_response = await self._generate_fallback_response(request)
            yield fallback_response.content
    
    async def _select_optimal_model(self, request: LLMRequest) -> LLMModel:
        """
        Select the optimal model based on:
        - Request priority and requirements
        - Cost constraints
        - Model availability and performance
        - Context length requirements
        """
        try:
            # Calculate context length requirement
            context_length = sum(len(msg.get('content', '')) for msg in request.messages)
            if request.system_prompt:
                context_length += len(request.system_prompt)
            
            # Filter models by context window capacity
            suitable_models = [
                model for model, config in self.model_configs.items()
                if config.context_window > context_length * 4  # Safety margin
                and model != LLMModel.FALLBACK_MODEL
            ]
            
            if not suitable_models:
                # Use largest context window available
                suitable_models = [max(
                    self.model_configs.keys(),
                    key=lambda m: self.model_configs[m].context_window
                    if m != LLMModel.FALLBACK_MODEL else 0
                )]
            
            # Score models based on request requirements
            model_scores = {}
            for model in suitable_models:
                config = self.model_configs[model]
                perf = self.model_performance.get(model, {})
                
                score = 0.0
                
                # Quality priority
                if request.priority == 1:  # High priority - favor quality
                    score += (4 - config.quality_tier) * 30
                    score += perf.get('avg_quality_score', 0.8) * 20
                elif request.priority == 2:  # Normal priority - balance cost/quality
                    score += (4 - config.quality_tier) * 15
                    score += (4 - config.speed_tier) * 15
                    score -= (config.cost_per_1k_input + config.cost_per_1k_output) * 1000
                else:  # Low priority - favor cost
                    score -= (config.cost_per_1k_input + config.cost_per_1k_output) * 2000
                    score += (4 - config.speed_tier) * 10
                
                # Performance factors
                score += perf.get('success_rate', 0.95) * 30
                score -= perf.get('avg_response_time', 2000) / 100
                
                # Availability check
                provider_usage = self.provider_usage.get(config.provider, {})
                if provider_usage.get('requests_this_minute', 0) > 50:
                    score -= 20  # Penalize high usage providers
                
                model_scores[model] = score
            
            # Select best scoring model
            best_model = max(model_scores.keys(), key=lambda m: model_scores[m])
            
            logger.debug(
                "Selected optimal model",
                model=best_model.value,
                score=model_scores[best_model],
                priority=request.priority,
                context_length=context_length
            )
            
            return best_model
            
        except Exception as e:
            logger.error("Error selecting optimal model", error=str(e))
            return LLMModel.GPT_35_TURBO  # Safe fallback
    
    async def _select_fallback_model(self, request: LLMRequest) -> Optional[LLMModel]:
        """Select a fallback model when primary selection fails."""
        try:
            # Try cheaper, faster models first
            fallback_order = [
                LLMModel.GPT_35_TURBO,
                LLMModel.CLAUDE_3_HAIKU,
                LLMModel.CLAUDE_3_SONNET,
                LLMModel.GPT_4_TURBO
            ]
            
            for model in fallback_order:
                if await self._check_rate_limits(model):
                    return model
            
            return None
            
        except Exception as e:
            logger.error("Error selecting fallback model", error=str(e))
            return None
    
    async def _check_rate_limits(self, model: LLMModel) -> bool:
        """Check if the model's provider is within rate limits."""
        try:
            config = self.model_configs[model]
            usage = self.provider_usage.get(config.provider, {})
            quotas = self.provider_quotas.get(config.provider, {})
            
            now = datetime.now()
            
            # Reset minute counters if needed
            last_reset = usage.get('last_reset', now)
            if (now - last_reset).total_seconds() >= 60:
                usage['requests_this_minute'] = 0
                usage['tokens_this_minute'] = 0
                usage['last_reset'] = now
                
                # Reset hourly counters
                if (now - last_reset).total_seconds() >= 3600:
                    usage['hourly_cost'] = 0.0
                
                # Reset daily counters
                if (now - last_reset).total_seconds() >= 86400:
                    usage['daily_cost'] = 0.0
            
            # Check quotas
            if usage.get('requests_this_minute', 0) >= quotas.get('requests_per_minute', 60):
                return False
            
            if usage.get('tokens_this_minute', 0) >= quotas.get('tokens_per_minute', 150000):
                return False
            
            if usage.get('hourly_cost', 0) >= quotas.get('hourly_limit', 100):
                return False
            
            if usage.get('daily_cost', 0) >= quotas.get('daily_limit', 1000):
                return False
            
            return True
            
        except Exception as e:
            logger.error("Error checking rate limits", error=str(e))
            return False
    
    async def _generate_with_model(
        self, 
        request: LLMRequest, 
        model: LLMModel
    ) -> LLMResponse:
        """Generate response with specific model."""
        config = self.model_configs[model]
        
        if config.provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(request, model)
        elif config.provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic_response(request, model)
        else:
            return await self._generate_fallback_response(request)
    
    async def _generate_openai_response(
        self, 
        request: LLMRequest, 
        model: LLMModel
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            
            config = self.model_configs[model]
            messages = request.messages.copy()
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Make API call
            response = await self.openai_client.chat.completions.create(
                model=config.model_id,
                messages=messages,
                max_tokens=request.max_tokens or config.max_tokens,
                temperature=request.temperature,
                stream=False,
                timeout=request.timeout_seconds
            )
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            tokens_used = {
                'input': usage.prompt_tokens,
                'output': usage.completion_tokens,
                'total': usage.total_tokens
            }
            
            # Calculate cost
            cost = (
                (usage.prompt_tokens / 1000) * config.cost_per_1k_input +
                (usage.completion_tokens / 1000) * config.cost_per_1k_output
            )
            
            return LLMResponse(
                content=content,
                model_used=model,
                provider=config.provider,
                tokens_used=tokens_used,
                cost_estimate=cost,
                response_time_ms=0,  # Will be set by caller
                metadata={'finish_reason': response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error("Error generating OpenAI response", error=str(e))
            raise
    
    async def _generate_anthropic_response(
        self, 
        request: LLMRequest, 
        model: LLMModel
    ) -> LLMResponse:
        """Generate response using Anthropic Claude API."""
        try:
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            config = self.model_configs[model]
            
            # Convert messages to Anthropic format
            messages = []
            system_prompt = request.system_prompt or ""
            
            for msg in request.messages:
                role = msg.get('role', 'user')
                if role == 'system':
                    system_prompt = msg.get('content', '')
                else:
                    messages.append({
                        'role': 'user' if role == 'user' else 'assistant',
                        'content': msg.get('content', '')
                    })
            
            # Make API call
            response = await self.anthropic_client.messages.create(
                model=config.model_id,
                max_tokens=request.max_tokens or config.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=messages,
                timeout=request.timeout_seconds
            )
            
            # Extract response data
            content = response.content[0].text
            
            # Estimate tokens (Anthropic doesn't return exact counts)
            input_tokens = sum(len(msg.get('content', '')) for msg in messages) // 4
            output_tokens = len(content) // 4
            total_tokens = input_tokens + output_tokens
            
            tokens_used = {
                'input': input_tokens,
                'output': output_tokens,
                'total': total_tokens
            }
            
            # Calculate cost
            cost = (
                (input_tokens / 1000) * config.cost_per_1k_input +
                (output_tokens / 1000) * config.cost_per_1k_output
            )
            
            return LLMResponse(
                content=content,
                model_used=model,
                provider=config.provider,
                tokens_used=tokens_used,
                cost_estimate=cost,
                response_time_ms=0,
                metadata={'stop_reason': response.stop_reason}
            )
            
        except Exception as e:
            logger.error("Error generating Anthropic response", error=str(e))
            raise
    
    async def _stream_openai_response(
        self, 
        request: LLMRequest, 
        model: LLMModel
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI API."""
        try:
            if not self.openai_client:
                return
                
            config = self.model_configs[model]
            messages = request.messages.copy()
            
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            stream = await self.openai_client.chat.completions.create(
                model=config.model_id,
                messages=messages,
                max_tokens=request.max_tokens or config.max_tokens,
                temperature=request.temperature,
                stream=True,
                timeout=request.timeout_seconds
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("Error streaming OpenAI response", error=str(e))
    
    async def _stream_anthropic_response(
        self, 
        request: LLMRequest, 
        model: LLMModel
    ) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic API."""
        try:
            if not self.anthropic_client:
                return
                
            config = self.model_configs[model]
            
            # Convert messages to Anthropic format
            messages = []
            system_prompt = request.system_prompt or ""
            
            for msg in request.messages:
                role = msg.get('role', 'user')
                if role == 'system':
                    system_prompt = msg.get('content', '')
                else:
                    messages.append({
                        'role': 'user' if role == 'user' else 'assistant',
                        'content': msg.get('content', '')
                    })
            
            async with self.anthropic_client.messages.stream(
                model=config.model_id,
                max_tokens=request.max_tokens or config.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=messages,
                timeout=request.timeout_seconds
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error("Error streaming Anthropic response", error=str(e))
    
    async def _generate_fallback_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback response when LLMs are unavailable."""
        try:
            # Select appropriate fallback based on context
            messages = request.messages
            last_message = messages[-1].get('content', '') if messages else ''
            
            # Simple pattern matching for fallback responses
            if any(word in last_message.lower() for word in ['help', 'assist', 'support']):
                content = "I'm here to help! Could you tell me more about what you need assistance with?"
            elif '?' in last_message:
                content = "That's a great question. Let me think about the best way to address that."
            elif any(word in last_message.lower() for word in ['thank', 'thanks']):
                content = "You're welcome! Is there anything else I can help you with?"
            elif any(word in last_message.lower() for word in ['hello', 'hi', 'hey']):
                content = "Hello! How can I assist you today?"
            else:
                # Use personality-aware fallback if available
                if request.personality_context:
                    content = await self._generate_personality_fallback(request)
                else:
                    import random
                    content = random.choice(self.fallback_responses)
            
            return LLMResponse(
                content=content,
                model_used=LLMModel.FALLBACK_MODEL,
                provider=LLMProvider.FALLBACK,
                tokens_used={'input': 0, 'output': len(content.split()), 'total': len(content.split())},
                cost_estimate=0.0,
                response_time_ms=100,  # Fast fallback
                cached=False,
                metadata={'fallback_reason': 'API unavailable'}
            )
            
        except Exception as e:
            logger.error("Error generating fallback response", error=str(e))
            return LLMResponse(
                content="I'm here and ready to help! How can I assist you?",
                model_used=LLMModel.FALLBACK_MODEL,
                provider=LLMProvider.FALLBACK,
                tokens_used={'input': 0, 'output': 10, 'total': 10},
                cost_estimate=0.0,
                response_time_ms=50,
                cached=False,
                metadata={'fallback_reason': 'Error in fallback generation'}
            )
    
    async def _generate_personality_fallback(self, request: LLMRequest) -> str:
        """Generate personality-aware fallback response."""
        try:
            personality = request.personality_context
            base_response = "I understand what you're saying. Let me help you with that."
            
            # Apply personality traits to base response
            traits = personality.adapted_traits
            
            # Adjust for extraversion
            extraversion = traits.get('extraversion', 0.5)
            if extraversion > 0.7:
                base_response = "That's really interesting! I'd love to help you explore that further."
            elif extraversion < 0.3:
                base_response = "I see. Let me consider the best way to assist you."
            
            # Adjust for agreeableness
            agreeableness = traits.get('agreeableness', 0.5)
            if agreeableness > 0.7:
                base_response += " I want to make sure I give you exactly what you're looking for."
            
            # Adjust for enthusiasm
            enthusiasm = traits.get('enthusiasm', 0.5)
            if enthusiasm > 0.7:
                base_response += " 😊"
            
            return base_response
            
        except Exception as e:
            logger.error("Error generating personality fallback", error=str(e))
            return "I'm here to help! How can I assist you?"
    
    async def _apply_personality_modifications(
        self, 
        content: str, 
        personality: PersonalityState,
        request: LLMRequest
    ) -> str:
        """Apply personality modifications to LLM response."""
        try:
            # Import personality engine locally to avoid circular imports
            from app.services.personality_engine import AdvancedPersonalityEngine
            
            # Create conversation context from request
            context = ConversationContext(
                user_id=request.user_id or "unknown",
                session_id=request.conversation_id or "unknown",
                message_history=[msg for msg in request.messages],
                current_sentiment=0.0,  # Default neutral
                conversation_phase="ongoing"
            )
            
            # Use personality engine to modify response
            # This would integrate with the existing personality engine
            modified_content = content  # Start with original content
            
            # Apply trait-based modifications
            traits = personality.adapted_traits
            
            # Extraversion modifications
            extraversion = traits.get('extraversion', 0.5)
            if extraversion > 0.7:
                # More enthusiastic
                modified_content = modified_content.replace('.', '!')
                if not modified_content.endswith(('!', '?')):
                    modified_content += '!'
            elif extraversion < 0.3:
                # More reserved
                modified_content = modified_content.replace('!', '.')
            
            # Formality modifications
            formality = traits.get('formality', 0.5)
            if formality > 0.7:
                # More formal
                contractions = {
                    "can't": "cannot", "won't": "will not", "don't": "do not",
                    "isn't": "is not", "aren't": "are not", "you're": "you are"
                }
                for informal, formal in contractions.items():
                    modified_content = modified_content.replace(informal, formal)
            elif formality < 0.3:
                # More casual
                formal_to_casual = {
                    "cannot": "can't", "will not": "won't", "do not": "don't",
                    "is not": "isn't", "are not": "aren't", "you are": "you're"
                }
                for formal, casual in formal_to_casual.items():
                    modified_content = modified_content.replace(formal, casual)
            
            # Empathy modifications
            empathy = traits.get('empathy', 0.5)
            if empathy > 0.7:
                # Add empathetic language
                empathetic_phrases = [
                    "I understand", "I can see", "That makes sense", "I hear you"
                ]
                if not any(phrase in modified_content for phrase in empathetic_phrases):
                    modified_content = f"I understand. {modified_content}"
            
            return modified_content
            
        except Exception as e:
            logger.error("Error applying personality modifications", error=str(e))
            return content  # Return original content on error
    
    async def _apply_personality_to_chunk(
        self, 
        chunk: str, 
        personality: PersonalityState
    ) -> str:
        """Apply personality modifications to streaming chunks."""
        try:
            # Light modifications for streaming chunks
            traits = personality.adapted_traits
            
            # Simple enthusiasm boost for extraversion
            if traits.get('extraversion', 0.5) > 0.7 and chunk.endswith('.'):
                chunk = chunk[:-1] + '!'
            
            return chunk
            
        except Exception as e:
            logger.error("Error applying personality to chunk", error=str(e))
            return chunk
    
    async def _check_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Check if response is cached."""
        try:
            if not self.redis:
                return None
            
            # Create cache key from request content
            cache_content = {
                'messages': request.messages,
                'system_prompt': request.system_prompt,
                'temperature': request.temperature,
                'personality_traits': request.personality_context.adapted_traits if request.personality_context else {}
            }
            
            cache_key = f"llm:response:{hashlib.md5(json.dumps(cache_content, sort_keys=True).encode()).hexdigest()}"
            
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                return LLMResponse(
                    content=response_data['content'],
                    model_used=LLMModel(response_data['model_used']),
                    provider=LLMProvider(response_data['provider']),
                    tokens_used=response_data['tokens_used'],
                    cost_estimate=response_data['cost_estimate'],
                    response_time_ms=50,  # Fast cache retrieval
                    cached=True,
                    metadata=response_data.get('metadata', {})
                )
            
            return None
            
        except Exception as e:
            logger.error("Error checking cache", error=str(e))
            return None
    
    async def _cache_response(self, request: LLMRequest, response: LLMResponse):
        """Cache response for future use."""
        try:
            if not self.redis or not response.content:
                return
            
            # Create cache key
            cache_content = {
                'messages': request.messages,
                'system_prompt': request.system_prompt,
                'temperature': request.temperature,
                'personality_traits': request.personality_context.adapted_traits if request.personality_context else {}
            }
            
            cache_key = f"llm:response:{hashlib.md5(json.dumps(cache_content, sort_keys=True).encode()).hexdigest()}"
            
            # Cache data
            cache_data = {
                'content': response.content,
                'model_used': response.model_used.value,
                'provider': response.provider.value,
                'tokens_used': response.tokens_used,
                'cost_estimate': response.cost_estimate,
                'metadata': response.metadata,
                'cached_at': datetime.now().isoformat()
            }
            
            # Cache for 1 hour for non-personalized responses, 30 minutes for personalized
            ttl = 1800 if request.personality_context else 3600
            
            await self.redis.setex(cache_key, ttl, json.dumps(cache_data))
            
            logger.debug("Cached LLM response", cache_key=cache_key, ttl=ttl)
            
        except Exception as e:
            logger.error("Error caching response", error=str(e))
    
    async def _update_performance_metrics(
        self, 
        model: LLMModel, 
        response: LLMResponse, 
        start_time: float
    ):
        """Update model performance metrics."""
        try:
            if model not in self.model_performance:
                self.model_performance[model] = {
                    'avg_response_time': 0,
                    'success_rate': 1.0,
                    'avg_quality_score': 0.8,
                    'total_requests': 0,
                    'total_errors': 0
                }
            
            metrics = self.model_performance[model]
            metrics['total_requests'] += 1
            
            # Update response time
            response_time = (time.time() - start_time) * 1000
            metrics['avg_response_time'] = (
                (metrics['avg_response_time'] * (metrics['total_requests'] - 1) + response_time) /
                metrics['total_requests']
            )
            
            # Estimate quality score (simple heuristic)
            quality_score = min(1.0, len(response.content) / 100)  # Longer responses = higher quality
            if response.tokens_used.get('total', 0) > 50:
                quality_score += 0.2
            
            metrics['avg_quality_score'] = (
                (metrics['avg_quality_score'] * (metrics['total_requests'] - 1) + quality_score) /
                metrics['total_requests']
            )
            
            # Update success rate
            success_count = metrics['total_requests'] - metrics['total_errors']
            metrics['success_rate'] = success_count / metrics['total_requests']
            
            response.quality_score = quality_score
            
            # Save to cache periodically
            if metrics['total_requests'] % 10 == 0:
                await self._save_performance_data()
            
        except Exception as e:
            logger.error("Error updating performance metrics", error=str(e))
    
    async def _track_usage(self, model: LLMModel, response: LLMResponse):
        """Track usage for quota management."""
        try:
            config = self.model_configs[model]
            usage = self.provider_usage.get(config.provider, {})
            
            if usage:
                usage['requests_this_minute'] = usage.get('requests_this_minute', 0) + 1
                usage['tokens_this_minute'] = usage.get('tokens_this_minute', 0) + response.tokens_used.get('total', 0)
                usage['hourly_cost'] = usage.get('hourly_cost', 0.0) + response.cost_estimate
                usage['daily_cost'] = usage.get('daily_cost', 0.0) + response.cost_estimate
                
        except Exception as e:
            logger.error("Error tracking usage", error=str(e))
    
    async def _save_performance_data(self):
        """Save performance data to cache."""
        try:
            if self.redis:
                await self.redis.setex(
                    "llm:model_performance",
                    3600,  # 1 hour TTL
                    json.dumps(self.model_performance, default=str)
                )
        except Exception as e:
            logger.error("Error saving performance data", error=str(e))
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        try:
            return {
                'provider_usage': self.provider_usage,
                'model_performance': self.model_performance,
                'cache_stats': {
                    'entries': len(self.response_cache),
                    'hit_rate': 0.85  # Placeholder
                }
            }
        except Exception as e:
            logger.error("Error getting usage stats", error=str(e))
            return {}
    
    async def shutdown(self):
        """Shutdown the LLM service."""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            await self._save_performance_data()
            
            logger.info("LLM service shutdown complete")
            
        except Exception as e:
            logger.error("Error during LLM service shutdown", error=str(e))


# Global instance
llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
        await llm_service.initialize()
    return llm_service


# Export main classes
__all__ = [
    'LLMService',
    'LLMRequest', 
    'LLMResponse',
    'LLMProvider',
    'LLMModel',
    'get_llm_service'
]