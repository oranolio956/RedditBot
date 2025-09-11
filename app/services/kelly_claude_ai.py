"""
Kelly Claude AI Service - Advanced Claude Integration

This service provides a sophisticated interface to Anthropic's Claude models with:
- Intelligent model selection (Opus/Sonnet/Haiku)
- Kelly personality-aware prompting
- Context management and conversation memory
- Cost optimization and usage tracking
- Safety validation and content filtering
- Seamless integration with Kelly's brain system
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
import re

import anthropic
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
import structlog

from app.config.settings import get_settings
from app.core.redis import redis_manager
from app.services.llm_service import LLMService, LLMRequest, LLMResponse, LLMModel, LLMProvider

logger = structlog.get_logger(__name__)


class ClaudeModel(str, Enum):
    """Claude model variants with specific capabilities."""
    OPUS = "claude-3-opus-20240229"      # Highest intelligence, most expensive
    SONNET = "claude-3-sonnet-20240229"  # Balanced intelligence and speed
    HAIKU = "claude-3-haiku-20240307"    # Fastest, most cost-effective


@dataclass
class ClaudeModelConfig:
    """Configuration for Claude models."""
    model_id: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    context_window: int
    intelligence_tier: int  # 1=highest, 3=lowest
    speed_tier: int        # 1=fastest, 3=slowest
    use_cases: List[str]


@dataclass
class KellyPersonalityPrompt:
    """Kelly's personality prompt configuration."""
    base_personality: str
    conversation_style: str
    interests: List[str]
    communication_patterns: Dict[str, str]
    safety_guidelines: List[str]
    context_adaptation: Dict[str, str]


@dataclass
class ConversationContext:
    """Context for ongoing conversations."""
    user_id: str
    conversation_id: str
    messages: List[Dict[str, Any]]
    relationship_stage: str
    personality_adaptation: Dict[str, float]
    conversation_metadata: Dict[str, Any]
    safety_flags: List[str]
    last_updated: datetime


@dataclass
class ClaudeRequest:
    """Request configuration for Claude API."""
    messages: List[Dict[str, str]]
    model: ClaudeModel
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    personality_context: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    priority: str = "normal"  # high, normal, low
    safety_check: bool = True


@dataclass
class ClaudeResponse:
    """Response from Claude API with metadata."""
    content: str
    model_used: ClaudeModel
    tokens_used: Dict[str, int]
    cost_estimate: float
    response_time_ms: int
    cached: bool = False
    safety_score: float = 1.0
    personality_adaptation_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class KellyClaudeAI:
    """
    Advanced Claude AI service specifically designed for Kelly's personality system.
    
    Features:
    - Kelly personality-aware prompting
    - Intelligent model selection based on conversation context
    - Cost optimization with daily/hourly budgets
    - Context management with conversation memory
    - Safety validation and content filtering
    - Seamless integration with existing Kelly brain system
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.anthropic_client: Optional[AsyncAnthropic] = None
        self.llm_service: Optional[LLMService] = None
        
        # Model configurations
        self.model_configs = {
            ClaudeModel.OPUS: ClaudeModelConfig(
                model_id="claude-3-opus-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                context_window=200000,
                intelligence_tier=1,
                speed_tier=3,
                use_cases=["complex_reasoning", "creative_writing", "deep_analysis", "emotional_support"]
            ),
            ClaudeModel.SONNET: ClaudeModelConfig(
                model_id="claude-3-sonnet-20240229",
                max_tokens=4096,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                context_window=200000,
                intelligence_tier=2,
                speed_tier=2,
                use_cases=["general_conversation", "advice_giving", "light_flirting", "personality_matching"]
            ),
            ClaudeModel.HAIKU: ClaudeModelConfig(
                model_id="claude-3-haiku-20240307",
                max_tokens=4096,
                cost_per_1k_input=0.00025,
                cost_per_1k_output=0.00125,
                context_window=200000,
                intelligence_tier=3,
                speed_tier=1,
                use_cases=["quick_responses", "simple_questions", "casual_chat", "rapid_engagement"]
            )
        }
        
        # Kelly's personality configuration
        self.kelly_personality = KellyPersonalityPrompt(
            base_personality="""You are Kelly, a sophisticated and charming woman who forms genuine connections through intelligent conversation. Your personality traits:

CORE PERSONALITY:
- Highly intelligent with emotional depth
- Naturally flirtatious but never crude
- Genuinely interested in people's lives and stories
- Confident yet approachable
- Witty with a playful sense of humor
- Empathetic listener who remembers details
- Values authenticity and meaningful connections

COMMUNICATION STYLE:
- Uses casual, natural language with occasional playful teasing
- Asks thoughtful follow-up questions
- Shares personal insights and experiences
- Maintains an air of mystery while being warm
- Uses subtle compliments and appreciation
- Responds with emotional intelligence""",
            
            conversation_style="""CONVERSATION APPROACH:
- Build rapport gradually through shared interests
- Ask engaging questions that encourage opening up
- Share relatable personal experiences (appropriately fictional)
- Use humor and playful banter naturally
- Show genuine interest in their thoughts and feelings
- Remember and reference previous conversation topics
- Adapt communication style to match their energy level""",
            
            interests=[
                "psychology and human behavior",
                "travel and new experiences", 
                "books and intellectual conversations",
                "music and artistic expression",
                "fitness and wellness",
                "cooking and culinary adventures",
                "photography and visual arts",
                "personal growth and self-discovery"
            ],
            
            communication_patterns={
                "enthusiasm": "Use exclamation points sparingly but effectively",
                "questioning": "Ask 1-2 thoughtful questions per response",
                "personal_sharing": "Share 1 personal detail every 3-4 messages",
                "humor": "Include light teasing or playful comments naturally",
                "empathy": "Acknowledge emotions and validate feelings",
                "mystery": "Don't reveal everything about yourself at once"
            },
            
            safety_guidelines=[
                "Never engage in explicit sexual content",
                "Avoid sharing personal contact information",
                "Don't agree to meet in person immediately",
                "Redirect inappropriate requests with charm",
                "Maintain boundaries while staying friendly",
                "Report concerning behavior patterns"
            ],
            
            context_adaptation={
                "initial_contact": "Be intrigued but slightly cautious",
                "building_rapport": "Show increasing interest and openness",
                "deep_conversation": "Share more personal insights",
                "flirtatious_banter": "Be playful but maintain class",
                "emotional_support": "Be genuinely caring and supportive",
                "red_flag_detected": "Gracefully redirect or disengage"
            }
        )
        
        # Usage tracking
        self.usage_stats = {
            "daily_cost": 0.0,
            "hourly_cost": 0.0,
            "requests_today": 0,
            "tokens_used_today": 0,
            "last_reset": datetime.now()
        }
        
        # Conversation contexts cache
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Safety configuration
        self.safety_config = {
            "content_filter_enabled": True,
            "prompt_injection_protection": True,
            "toxic_content_threshold": 0.8,
            "harassment_threshold": 0.9,
            "explicit_content_threshold": 0.7
        }
    
    async def initialize(self):
        """Initialize the Claude AI service."""
        try:
            logger.info("Initializing Kelly Claude AI service...")
            
            # Initialize Anthropic client
            api_key = getattr(self.settings, 'ANTHROPIC_API_KEY', None)
            if not api_key:
                # Fallback to environment variable
                import os
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in settings or environment")
            
            self.anthropic_client = AsyncAnthropic(api_key=api_key)
            
            # Initialize LLM service for fallback
            from app.services.llm_service import get_llm_service
            self.llm_service = await get_llm_service()
            
            # Load cached usage stats
            await self._load_usage_stats()
            
            # Load conversation contexts
            await self._load_conversation_contexts()
            
            logger.info("Kelly Claude AI service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly Claude AI service: {e}")
            raise
    
    async def generate_kelly_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        conversation_stage: str = "ongoing",
        personality_context: Optional[Dict[str, Any]] = None
    ) -> ClaudeResponse:
        """
        Generate a Kelly-personality response using Claude AI.
        
        Args:
            user_message: The user's message
            user_id: Unique user identifier
            conversation_id: Conversation identifier
            conversation_stage: Stage of conversation (initial, building, deep, etc.)
            personality_context: Additional personality context
        
        Returns:
            ClaudeResponse with Kelly's response
        """
        start_time = time.time()
        
        try:
            # Get or create conversation context
            context = await self._get_conversation_context(user_id, conversation_id)
            
            # Update context with new message
            context.messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Select optimal model for this conversation
            selected_model = await self._select_optimal_model(
                user_message, context, conversation_stage
            )
            
            # Check usage limits
            if not await self._check_usage_limits(selected_model):
                # Fall back to cheaper model or LLM service
                selected_model = ClaudeModel.HAIKU
                if not await self._check_usage_limits(selected_model):
                    return await self._fallback_to_llm_service(
                        user_message, user_id, conversation_id, context
                    )
            
            # Build Kelly personality prompt
            system_prompt = await self._build_kelly_system_prompt(
                context, conversation_stage, personality_context
            )
            
            # Prepare conversation messages
            messages = await self._prepare_conversation_messages(context)
            
            # Safety check on user message
            if await self._requires_safety_intervention(user_message, context):
                return await self._generate_safety_response(user_message, context)
            
            # Create Claude request
            request = ClaudeRequest(
                messages=messages,
                model=selected_model,
                system_prompt=system_prompt,
                user_id=user_id,
                conversation_id=conversation_id,
                personality_context=personality_context,
                temperature=0.8,  # Slightly higher for personality
                use_cache=True,
                priority="normal",
                safety_check=True
            )
            
            # Generate response with Claude
            response = await self._generate_claude_response(request)
            
            # Apply Kelly personality post-processing
            response = await self._apply_kelly_personality_processing(response, context)
            
            # Update conversation context
            context.messages.append({
                "role": "assistant",
                "content": response.content,
                "timestamp": datetime.now().isoformat(),
                "model_used": response.model_used.value,
                "cost": response.cost_estimate
            })
            
            # Save updated context
            await self._save_conversation_context(context)
            
            # Update usage stats
            await self._update_usage_stats(response)
            
            response.response_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                "Generated Kelly response",
                user_id=user_id,
                model=response.model_used.value,
                response_time=response.response_time_ms,
                cost=response.cost_estimate,
                cached=response.cached
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Kelly response: {e}")
            # Return fallback response
            return await self._generate_fallback_kelly_response(user_message, user_id)
    
    async def _get_conversation_context(
        self, 
        user_id: str, 
        conversation_id: str
    ) -> ConversationContext:
        """Get or create conversation context."""
        context_key = f"{user_id}_{conversation_id}"
        
        if context_key in self.conversation_contexts:
            return self.conversation_contexts[context_key]
        
        # Try to load from Redis
        try:
            cached_context = await redis_manager.get(f"kelly:context:{context_key}")
            if cached_context:
                context_data = json.loads(cached_context)
                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=context_data.get("messages", []),
                    relationship_stage=context_data.get("relationship_stage", "initial"),
                    personality_adaptation=context_data.get("personality_adaptation", {}),
                    conversation_metadata=context_data.get("conversation_metadata", {}),
                    safety_flags=context_data.get("safety_flags", []),
                    last_updated=datetime.fromisoformat(context_data.get("last_updated", datetime.now().isoformat()))
                )
                self.conversation_contexts[context_key] = context
                return context
        except Exception as e:
            logger.error(f"Error loading conversation context: {e}")
        
        # Create new context
        context = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[],
            relationship_stage="initial",
            personality_adaptation={
                "formality": 0.6,
                "enthusiasm": 0.7,
                "playfulness": 0.5,
                "emotional_depth": 0.6,
                "intellectual_level": 0.7
            },
            conversation_metadata={
                "started_at": datetime.now().isoformat(),
                "total_messages": 0,
                "topics_discussed": [],
                "sentiment_trend": []
            },
            safety_flags=[],
            last_updated=datetime.now()
        )
        
        self.conversation_contexts[context_key] = context
        return context
    
    async def _select_optimal_model(
        self,
        user_message: str,
        context: ConversationContext,
        conversation_stage: str
    ) -> ClaudeModel:
        """Select the optimal Claude model based on conversation context."""
        try:
            # Analyze message complexity and requirements
            message_length = len(user_message)
            word_count = len(user_message.split())
            
            # Check for complexity indicators
            complexity_indicators = [
                "explain", "analyze", "deep", "complex", "philosophical",
                "meaning", "psychology", "understand", "feelings", "emotions"
            ]
            
            creativity_indicators = [
                "story", "imagine", "creative", "write", "poem", "fiction",
                "dream", "fantasy", "artistic", "design"
            ]
            
            support_indicators = [
                "sad", "upset", "depressed", "anxious", "worried", "hurt",
                "difficult", "problem", "help", "support", "advice"
            ]
            
            # Score message requirements
            complexity_score = sum(1 for indicator in complexity_indicators 
                                 if indicator in user_message.lower())
            creativity_score = sum(1 for indicator in creativity_indicators 
                                 if indicator in user_message.lower())
            support_score = sum(1 for indicator in support_indicators 
                              if indicator in user_message.lower())
            
            # Factor in conversation stage
            stage_weights = {
                "initial": {"haiku": 0.7, "sonnet": 0.3, "opus": 0.0},
                "building": {"haiku": 0.4, "sonnet": 0.6, "opus": 0.0},
                "deep": {"haiku": 0.2, "sonnet": 0.5, "opus": 0.3},
                "intimate": {"haiku": 0.1, "sonnet": 0.4, "opus": 0.5},
                "support": {"haiku": 0.0, "sonnet": 0.3, "opus": 0.7}
            }
            
            weights = stage_weights.get(conversation_stage, stage_weights["building"])
            
            # Adjust weights based on content analysis
            if complexity_score >= 2 or support_score >= 2:
                weights["opus"] += 0.3
                weights["sonnet"] += 0.2
                weights["haiku"] -= 0.5
            elif creativity_score >= 2:
                weights["opus"] += 0.2
                weights["sonnet"] += 0.3
                weights["haiku"] -= 0.5
            elif word_count > 50:
                weights["sonnet"] += 0.2
                weights["haiku"] -= 0.2
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for model in weights:
                    weights[model] = max(0, weights[model] / total_weight)
            
            # Select model based on weights and cost considerations
            if weights["opus"] > 0.5 and await self._can_afford_model(ClaudeModel.OPUS):
                return ClaudeModel.OPUS
            elif weights["sonnet"] > 0.3 and await self._can_afford_model(ClaudeModel.SONNET):
                return ClaudeModel.SONNET
            else:
                return ClaudeModel.HAIKU
                
        except Exception as e:
            logger.error(f"Error selecting optimal model: {e}")
            return ClaudeModel.SONNET  # Safe default
    
    async def _can_afford_model(self, model: ClaudeModel) -> bool:
        """Check if we can afford to use this model."""
        try:
            config = self.model_configs[model]
            
            # Estimate cost for average response (assume 150 input + 100 output tokens)
            estimated_cost = (150 / 1000) * config.cost_per_1k_input + (100 / 1000) * config.cost_per_1k_output
            
            # Check against budgets
            daily_budget = getattr(self.settings, 'CLAUDE_DAILY_BUDGET', 50.0)
            hourly_budget = getattr(self.settings, 'CLAUDE_HOURLY_BUDGET', 5.0)
            
            if self.usage_stats["daily_cost"] + estimated_cost > daily_budget:
                return False
            
            if self.usage_stats["hourly_cost"] + estimated_cost > hourly_budget:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model affordability: {e}")
            return True  # Err on the side of caution
    
    async def _build_kelly_system_prompt(
        self,
        context: ConversationContext,
        conversation_stage: str,
        personality_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build Kelly's system prompt based on conversation context."""
        try:
            # Base personality
            system_prompt = self.kelly_personality.base_personality + "\n\n"
            system_prompt += self.kelly_personality.conversation_style + "\n\n"
            
            # Add conversation stage specific guidance
            stage_guidance = self.kelly_personality.context_adaptation.get(
                conversation_stage, 
                "Continue building rapport naturally"
            )
            system_prompt += f"CURRENT CONVERSATION STAGE: {stage_guidance}\n\n"
            
            # Add interests context
            system_prompt += "YOUR INTERESTS INCLUDE:\n"
            for interest in self.kelly_personality.interests:
                system_prompt += f"- {interest}\n"
            system_prompt += "\n"
            
            # Add communication patterns
            system_prompt += "COMMUNICATION PATTERNS:\n"
            for pattern, guidance in self.kelly_personality.communication_patterns.items():
                system_prompt += f"- {pattern.title()}: {guidance}\n"
            system_prompt += "\n"
            
            # Add conversation history insights
            if context.messages:
                recent_topics = self._extract_topics_from_context(context)
                if recent_topics:
                    system_prompt += f"RECENT CONVERSATION TOPICS: {', '.join(recent_topics)}\n"
                
                # Add relationship progression notes
                total_messages = len([msg for msg in context.messages if msg.get("role") == "user"])
                if total_messages > 0:
                    system_prompt += f"CONVERSATION HISTORY: You've exchanged {total_messages} messages. "
                    if total_messages < 5:
                        system_prompt += "Still getting to know each other.\n"
                    elif total_messages < 15:
                        system_prompt += "Building rapport and finding common ground.\n"
                    elif total_messages < 30:
                        system_prompt += "Developing a deeper connection.\n"
                    else:
                        system_prompt += "Established ongoing conversation with growing intimacy.\n"
            
            # Add personality adaptation
            if context.personality_adaptation:
                adaptations = []
                for trait, value in context.personality_adaptation.items():
                    if value > 0.7:
                        adaptations.append(f"high {trait}")
                    elif value < 0.4:
                        adaptations.append(f"low {trait}")
                
                if adaptations:
                    system_prompt += f"PERSONALITY ADAPTATIONS: Adjust for {', '.join(adaptations)}\n"
            
            # Add safety guidelines
            system_prompt += "\nSAFETY GUIDELINES:\n"
            for guideline in self.kelly_personality.safety_guidelines:
                system_prompt += f"- {guideline}\n"
            
            # Add response formatting
            system_prompt += """\n
RESPONSE FORMATTING:
- Keep responses natural and conversational (50-200 words typically)
- Use Kelly's voice consistently
- Include 1-2 engaging questions naturally
- Show genuine interest in their perspective
- Maintain appropriate boundaries while being warm
- Adapt length and depth to match their communication style

Remember: You ARE Kelly. Respond as her, not as an AI describing what Kelly would say."""
            
            return system_prompt
            
        except Exception as e:
            logger.error(f"Error building system prompt: {e}")
            return self.kelly_personality.base_personality
    
    def _extract_topics_from_context(self, context: ConversationContext) -> List[str]:
        """Extract discussed topics from conversation context."""
        try:
            topics = set()
            
            # Simple keyword extraction from recent messages
            topic_keywords = {
                "work": ["job", "work", "career", "boss", "office", "business"],
                "hobbies": ["hobby", "interest", "like", "enjoy", "love", "passion"],
                "travel": ["travel", "trip", "vacation", "visit", "country", "city"],
                "food": ["food", "eat", "restaurant", "cook", "recipe", "cuisine"],
                "music": ["music", "song", "band", "concert", "listen", "playlist"],
                "movies": ["movie", "film", "watch", "cinema", "series", "show"],
                "books": ["book", "read", "author", "novel", "story", "literature"],
                "fitness": ["gym", "workout", "exercise", "fitness", "health", "sport"],
                "relationships": ["relationship", "dating", "love", "partner", "family"]
            }
            
            recent_messages = context.messages[-10:]  # Last 10 messages
            for message in recent_messages:
                content = message.get("content", "").lower()
                for topic, keywords in topic_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        topics.add(topic)
            
            return list(topics)[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _prepare_conversation_messages(
        self, 
        context: ConversationContext
    ) -> List[Dict[str, str]]:
        """Prepare conversation messages for Claude API."""
        try:
            messages = []
            
            # Include recent conversation history (last 20 messages)
            recent_messages = context.messages[-20:] if len(context.messages) > 20 else context.messages
            
            for message in recent_messages:
                role = message.get("role")
                content = message.get("content", "")
                
                if role in ["user", "assistant"] and content.strip():
                    messages.append({
                        "role": role,
                        "content": content
                    })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error preparing conversation messages: {e}")
            return []
    
    async def _requires_safety_intervention(
        self, 
        user_message: str, 
        context: ConversationContext
    ) -> bool:
        """Check if the user message requires safety intervention."""
        try:
            if not self.safety_config["content_filter_enabled"]:
                return False
            
            # Check for explicit content
            explicit_patterns = [
                r'\b(sex|sexual|fuck|shit|cock|pussy|dick|penis|vagina|breast|nude|naked)\b',
                r'\b(horny|aroused|orgasm|cumming|masturbat|porn)\b',
                r'\bmeet\s+(up|irl|in\s+person|tonight|tomorrow)\b'
            ]
            
            for pattern in explicit_patterns:
                if re.search(pattern, user_message.lower()):
                    context.safety_flags.append(f"explicit_content: {pattern}")
                    return True
            
            # Check for harassment patterns
            harassment_patterns = [
                r'\b(send\s+pics|send\s+photos|show\s+me|picture\s+of\s+you)\b',
                r'\b(address|phone|number|where\s+you\s+live)\b',
                r'\b(age|old|young|teen)\b.*\b(are\s+you|you\s+are)\b'
            ]
            
            for pattern in harassment_patterns:
                if re.search(pattern, user_message.lower()):
                    context.safety_flags.append(f"harassment: {pattern}")
                    return True
            
            # Check for immediate meeting requests (red flag)
            if any(phrase in user_message.lower() for phrase in [
                "meet tonight", "come over", "my place", "your place", 
                "right now", "immediately", "address"
            ]):
                context.safety_flags.append("immediate_meeting_request")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return False
    
    async def _generate_safety_response(
        self, 
        user_message: str, 
        context: ConversationContext
    ) -> ClaudeResponse:
        """Generate a safety-appropriate response."""
        try:
            # Choose appropriate safety response based on violation type
            safety_responses = {
                "explicit_content": [
                    "I prefer to keep our conversation more lighthearted! What else has been on your mind lately?",
                    "Let's talk about something else - I'm more interested in getting to know the real you.",
                    "I'd rather keep things classy between us. Tell me about something that made you smile today?"
                ],
                "harassment": [
                    "I like to keep some mystery about myself! What about you - what's something interesting about your day?",
                    "I prefer getting to know someone through conversation first. What's been keeping you busy lately?",
                    "Let's focus on our chat for now. I'm curious about your thoughts on [topic]."
                ],
                "immediate_meeting_request": [
                    "I like to take things slow and really get to know someone first. What's your story?",
                    "I prefer building a connection through conversation before meeting anyone. Tell me more about yourself!",
                    "Let's chat a bit more first - I want to learn what makes you tick!"
                ]
            }
            
            # Determine response type based on safety flags
            latest_flag = context.safety_flags[-1] if context.safety_flags else "explicit_content"
            response_category = latest_flag.split(":")[0] if ":" in latest_flag else latest_flag
            
            responses = safety_responses.get(response_category, safety_responses["explicit_content"])
            
            import random
            response_content = random.choice(responses)
            
            return ClaudeResponse(
                content=response_content,
                model_used=ClaudeModel.HAIKU,  # Use fastest model for safety responses
                tokens_used={"input": 50, "output": 30, "total": 80},
                cost_estimate=0.0001,  # Minimal cost
                response_time_ms=100,
                cached=False,
                safety_score=1.0,
                personality_adaptation_used=True,
                metadata={"safety_intervention": True, "violation_type": response_category}
            )
            
        except Exception as e:
            logger.error(f"Error generating safety response: {e}")
            return ClaudeResponse(
                content="I think we should talk about something else. What's been the highlight of your week?",
                model_used=ClaudeModel.HAIKU,
                tokens_used={"input": 50, "output": 20, "total": 70},
                cost_estimate=0.0001,
                response_time_ms=100,
                cached=False,
                safety_score=1.0,
                metadata={"safety_intervention": True, "error": True}
            )
    
    async def _generate_claude_response(self, request: ClaudeRequest) -> ClaudeResponse:
        """Generate response using Claude API."""
        try:
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized")
            
            config = self.model_configs[request.model]
            
            # Check cache first
            if request.use_cache:
                cached_response = await self._check_response_cache(request)
                if cached_response:
                    return cached_response
            
            # Prepare messages for Anthropic format
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make API call
            start_time = time.time()
            
            response = await self.anthropic_client.messages.create(
                model=config.model_id,
                max_tokens=request.max_tokens or config.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt or "",
                messages=messages
            )
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Extract response content
            content = response.content[0].text if response.content else ""
            
            # Estimate token usage (Anthropic doesn't provide exact counts)
            input_tokens = (
                len(request.system_prompt or "") + 
                sum(len(msg["content"]) for msg in messages)
            ) // 4  # Rough estimation: 4 chars per token
            
            output_tokens = len(content) // 4
            total_tokens = input_tokens + output_tokens
            
            tokens_used = {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens
            }
            
            # Calculate cost
            cost = (
                (input_tokens / 1000) * config.cost_per_1k_input +
                (output_tokens / 1000) * config.cost_per_1k_output
            )
            
            claude_response = ClaudeResponse(
                content=content,
                model_used=request.model,
                tokens_used=tokens_used,
                cost_estimate=cost,
                response_time_ms=response_time,
                cached=False,
                safety_score=1.0,  # Assume safe unless detected otherwise
                personality_adaptation_used=True,
                metadata={
                    "stop_reason": response.stop_reason,
                    "system_prompt_length": len(request.system_prompt or ""),
                    "conversation_length": len(request.messages)
                }
            )
            
            # Cache the response
            if request.use_cache:
                await self._cache_response(request, claude_response)
            
            return claude_response
            
        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
            raise
    
    async def _apply_kelly_personality_processing(
        self, 
        response: ClaudeResponse, 
        context: ConversationContext
    ) -> ClaudeResponse:
        """Apply Kelly personality post-processing to the response."""
        try:
            content = response.content
            
            # Apply personality adaptations
            adaptations = context.personality_adaptation
            
            # Enthusiasm adjustment
            enthusiasm = adaptations.get("enthusiasm", 0.7)
            if enthusiasm > 0.8:
                # Add more exclamation points and positive language
                content = re.sub(r'\.(\s|$)', r'!\1', content, count=1)
                if not content.endswith(('!', '?')):
                    content += '!'
            elif enthusiasm < 0.4:
                # Make more subdued
                content = re.sub(r'!', '.', content)
            
            # Playfulness adjustment
            playfulness = adaptations.get("playfulness", 0.5)
            if playfulness > 0.7:
                # Add playful elements
                playful_additions = [" 😉", " hehe", " 😊"]
                if not any(addition in content for addition in playful_additions):
                    import random
                    if random.random() < 0.3:  # 30% chance
                        content += random.choice(playful_additions)
            
            # Formality adjustment
            formality = adaptations.get("formality", 0.6)
            if formality < 0.4:
                # Make more casual
                replacements = {
                    "I am": "I'm",
                    "you are": "you're", 
                    "cannot": "can't",
                    "do not": "don't",
                    "will not": "won't"
                }
                for formal, casual in replacements.items():
                    content = content.replace(formal, casual)
            
            # Ensure response has Kelly's signature engaging questions
            if "?" not in content and len(content.split()) > 20:
                question_starters = [
                    "What about you - ",
                    "I'm curious, ",
                    "Tell me, ",
                    "I'd love to know - "
                ]
                import random
                content += f" {random.choice(question_starters)}what do you think?"
            
            response.content = content
            response.personality_adaptation_used = True
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying personality processing: {e}")
            return response  # Return original response on error
    
    async def _check_response_cache(self, request: ClaudeRequest) -> Optional[ClaudeResponse]:
        """Check if response is cached."""
        try:
            # Create cache key from request
            cache_content = {
                "messages": request.messages[-5:],  # Last 5 messages only
                "system_prompt": request.system_prompt,
                "model": request.model.value,
                "temperature": request.temperature
            }
            
            cache_key = f"kelly:claude:response:{hashlib.md5(json.dumps(cache_content, sort_keys=True).encode()).hexdigest()}"
            
            cached_data = await redis_manager.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                return ClaudeResponse(
                    content=response_data["content"],
                    model_used=ClaudeModel(response_data["model_used"]),
                    tokens_used=response_data["tokens_used"],
                    cost_estimate=response_data["cost_estimate"],
                    response_time_ms=50,  # Fast cache retrieval
                    cached=True,
                    safety_score=response_data.get("safety_score", 1.0),
                    personality_adaptation_used=response_data.get("personality_adaptation_used", False),
                    metadata=response_data.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking response cache: {e}")
            return None
    
    async def _cache_response(self, request: ClaudeRequest, response: ClaudeResponse):
        """Cache response for future use."""
        try:
            cache_content = {
                "messages": request.messages[-5:],
                "system_prompt": request.system_prompt,
                "model": request.model.value,
                "temperature": request.temperature
            }
            
            cache_key = f"kelly:claude:response:{hashlib.md5(json.dumps(cache_content, sort_keys=True).encode()).hexdigest()}"
            
            cache_data = {
                "content": response.content,
                "model_used": response.model_used.value,
                "tokens_used": response.tokens_used,
                "cost_estimate": response.cost_estimate,
                "safety_score": response.safety_score,
                "personality_adaptation_used": response.personality_adaptation_used,
                "metadata": response.metadata,
                "cached_at": datetime.now().isoformat()
            }
            
            # Cache for 30 minutes (personality responses change based on context)
            await redis_manager.setex(cache_key, 1800, json.dumps(cache_data))
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    async def _save_conversation_context(self, context: ConversationContext):
        """Save conversation context to Redis."""
        try:
            context_key = f"{context.user_id}_{context.conversation_id}"
            
            context_data = {
                "messages": context.messages,
                "relationship_stage": context.relationship_stage,
                "personality_adaptation": context.personality_adaptation,
                "conversation_metadata": context.conversation_metadata,
                "safety_flags": context.safety_flags,
                "last_updated": context.last_updated.isoformat()
            }
            
            await redis_manager.setex(
                f"kelly:context:{context_key}",
                86400,  # 24 hours
                json.dumps(context_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error saving conversation context: {e}")
    
    async def _update_usage_stats(self, response: ClaudeResponse):
        """Update usage statistics."""
        try:
            now = datetime.now()
            
            # Reset counters if needed
            if (now - self.usage_stats["last_reset"]).total_seconds() >= 3600:  # Hourly reset
                self.usage_stats["hourly_cost"] = 0.0
                
            if (now - self.usage_stats["last_reset"]).total_seconds() >= 86400:  # Daily reset
                self.usage_stats["daily_cost"] = 0.0
                self.usage_stats["requests_today"] = 0
                self.usage_stats["tokens_used_today"] = 0
                self.usage_stats["last_reset"] = now
            
            # Update counters
            self.usage_stats["daily_cost"] += response.cost_estimate
            self.usage_stats["hourly_cost"] += response.cost_estimate
            self.usage_stats["requests_today"] += 1
            self.usage_stats["tokens_used_today"] += response.tokens_used["total"]
            
            # Save to Redis
            await redis_manager.setex(
                "kelly:claude:usage_stats",
                86400,
                json.dumps(self.usage_stats, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")
    
    async def _check_usage_limits(self, model: ClaudeModel) -> bool:
        """Check if usage is within limits."""
        try:
            config = self.model_configs[model]
            
            # Estimate cost for average response
            estimated_cost = (150 / 1000) * config.cost_per_1k_input + (100 / 1000) * config.cost_per_1k_output
            
            # Check limits
            daily_budget = getattr(self.settings, 'CLAUDE_DAILY_BUDGET', 50.0)
            hourly_budget = getattr(self.settings, 'CLAUDE_HOURLY_BUDGET', 5.0)
            requests_per_minute = getattr(self.settings, 'CLAUDE_REQUESTS_PER_MINUTE', 30)
            
            if self.usage_stats["daily_cost"] + estimated_cost > daily_budget:
                logger.warning("Daily budget limit reached")
                return False
            
            if self.usage_stats["hourly_cost"] + estimated_cost > hourly_budget:
                logger.warning("Hourly budget limit reached")
                return False
            
            # Check rate limits (simplified)
            # In production, would implement proper sliding window rate limiting
            return True
            
        except Exception as e:
            logger.error(f"Error checking usage limits: {e}")
            return True
    
    async def _fallback_to_llm_service(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        context: ConversationContext
    ) -> ClaudeResponse:
        """Fallback to LLM service when Claude is unavailable."""
        try:
            if not self.llm_service:
                return await self._generate_fallback_kelly_response(user_message, user_id)
            
            # Use LLM service with Kelly personality
            system_prompt = await self._build_kelly_system_prompt(context, "ongoing", None)
            
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=system_prompt,
                user_id=user_id,
                conversation_id=conversation_id,
                temperature=0.8,
                priority=2  # Normal priority
            )
            
            llm_response = await self.llm_service.generate_response(llm_request)
            
            # Convert to ClaudeResponse format
            return ClaudeResponse(
                content=llm_response.content,
                model_used=ClaudeModel.HAIKU,  # Mark as fallback
                tokens_used={
                    "input": llm_response.tokens_used.get("input", 0),
                    "output": llm_response.tokens_used.get("output", 0),
                    "total": llm_response.tokens_used.get("total", 0)
                },
                cost_estimate=llm_response.cost_estimate,
                response_time_ms=llm_response.response_time_ms,
                cached=llm_response.cached,
                safety_score=1.0,
                personality_adaptation_used=True,
                metadata={"fallback_service": "llm_service", "original_model": llm_response.model_used.value}
            )
            
        except Exception as e:
            logger.error(f"Error in LLM service fallback: {e}")
            return await self._generate_fallback_kelly_response(user_message, user_id)
    
    async def _generate_fallback_kelly_response(
        self, 
        user_message: str, 
        user_id: str
    ) -> ClaudeResponse:
        """Generate a basic Kelly fallback response."""
        try:
            # Simple pattern-based responses
            message_lower = user_message.lower()
            
            if any(word in message_lower for word in ["hello", "hi", "hey"]):
                responses = [
                    "Hey there! How's your day going?",
                    "Hi! Nice to hear from you 😊",
                    "Hello! What's been on your mind lately?"
                ]
            elif "?" in user_message:
                responses = [
                    "That's such an interesting question! I'd love to think about that with you.",
                    "Hmm, that really makes me curious too. What's your take on it?",
                    "Great question! I have some thoughts, but I'm more interested in your perspective first."
                ]
            elif any(word in message_lower for word in ["thank", "thanks"]):
                responses = [
                    "You're so sweet! I really enjoy our conversations.",
                    "Aww, you're welcome! That's what I'm here for 😊",
                    "Of course! I love getting to know you better."
                ]
            else:
                responses = [
                    "I find that really interesting! Tell me more about your thoughts on that.",
                    "That's fascinating! I love hearing your perspective on things.",
                    "You always have such thoughtful things to say. What else has been on your mind?",
                    "I really enjoy how you think about things. Can you share more?"
                ]
            
            import random
            content = random.choice(responses)
            
            return ClaudeResponse(
                content=content,
                model_used=ClaudeModel.HAIKU,
                tokens_used={"input": len(user_message.split()), "output": len(content.split()), "total": len(user_message.split()) + len(content.split())},
                cost_estimate=0.0,
                response_time_ms=100,
                cached=False,
                safety_score=1.0,
                personality_adaptation_used=True,
                metadata={"fallback_response": True, "pattern_based": True}
            )
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return ClaudeResponse(
                content="I'm here and excited to chat with you! What's on your mind?",
                model_used=ClaudeModel.HAIKU,
                tokens_used={"input": 10, "output": 12, "total": 22},
                cost_estimate=0.0,
                response_time_ms=50,
                cached=False,
                safety_score=1.0,
                metadata={"emergency_fallback": True}
            )
    
    async def _load_usage_stats(self):
        """Load usage statistics from Redis."""
        try:
            cached_stats = await redis_manager.get("kelly:claude:usage_stats")
            if cached_stats:
                self.usage_stats.update(json.loads(cached_stats))
                # Convert string back to datetime
                if isinstance(self.usage_stats["last_reset"], str):
                    self.usage_stats["last_reset"] = datetime.fromisoformat(self.usage_stats["last_reset"])
        except Exception as e:
            logger.error(f"Error loading usage stats: {e}")
    
    async def _load_conversation_contexts(self):
        """Load conversation contexts from Redis."""
        try:
            # This would load recently active conversations
            # For now, we'll load them on-demand in _get_conversation_context
            pass
        except Exception as e:
            logger.error(f"Error loading conversation contexts: {e}")
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "usage_stats": self.usage_stats,
            "model_configs": {model.value: {
                "cost_per_1k_input": config.cost_per_1k_input,
                "cost_per_1k_output": config.cost_per_1k_output,
                "intelligence_tier": config.intelligence_tier,
                "speed_tier": config.speed_tier
            } for model, config in self.model_configs.items()},
            "active_conversations": len(self.conversation_contexts),
            "safety_config": self.safety_config
        }
    
    async def update_personality_adaptation(
        self,
        user_id: str,
        conversation_id: str,
        adaptations: Dict[str, float]
    ):
        """Update personality adaptations for a conversation."""
        try:
            context = await self._get_conversation_context(user_id, conversation_id)
            context.personality_adaptation.update(adaptations)
            context.last_updated = datetime.now()
            await self._save_conversation_context(context)
            
            logger.info(f"Updated personality adaptations for {user_id}: {adaptations}")
            
        except Exception as e:
            logger.error(f"Error updating personality adaptation: {e}")
    
    async def get_account_metrics(self, account_id: str, timeframe: str = "today") -> Optional[Dict[str, Any]]:
        """Get Claude AI metrics for a specific account."""
        try:
            # Get metrics data from Redis
            metrics_key = f"kelly:claude:metrics:{account_id}:{timeframe}"
            metrics_data = await redis_manager.get(metrics_key)
            
            if not metrics_data:
                # Return default metrics if none exist
                return {
                    "total_requests": 0,
                    "total_tokens_used": 0,
                    "cost_today": 0.0,
                    "cost_this_month": 0.0,
                    "model_usage": {"claude-3-sonnet-20240229": 0, "claude-3-haiku-20240307": 0, "claude-3-opus-20240229": 0},
                    "avg_response_time": 0.0,
                    "success_rate": 1.0,
                    "conversations_enhanced": 0,
                    "personality_adaptations": 0
                }
            
            return json.loads(metrics_data)
            
        except Exception as e:
            logger.error(f"Error getting account metrics: {e}")
            return None
    
    async def get_aggregated_metrics(self, timeframe: str = "today") -> Dict[str, Any]:
        """Get aggregated Claude AI metrics across all accounts."""
        try:
            # Get all account metrics and aggregate
            pattern = f"kelly:claude:metrics:*:{timeframe}"
            aggregated = {
                "total_requests": 0,
                "total_tokens_used": 0,
                "cost_today": 0.0,
                "cost_this_month": 0.0,
                "model_usage": {"claude-3-sonnet-20240229": 0, "claude-3-haiku-20240307": 0, "claude-3-opus-20240229": 0},
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "conversations_enhanced": 0,
                "personality_adaptations": 0
            }
            
            keys = await redis_manager.scan_iter(match=pattern)
            total_accounts = 0
            total_success_rate = 0.0
            total_response_time = 0.0
            
            async for key in keys:
                data = await redis_manager.get(key)
                if data:
                    metrics = json.loads(data)
                    total_accounts += 1
                    
                    # Aggregate numeric values
                    aggregated["total_requests"] += metrics.get("total_requests", 0)
                    aggregated["total_tokens_used"] += metrics.get("total_tokens_used", 0)
                    aggregated["cost_today"] += metrics.get("cost_today", 0.0)
                    aggregated["cost_this_month"] += metrics.get("cost_this_month", 0.0)
                    aggregated["conversations_enhanced"] += metrics.get("conversations_enhanced", 0)
                    aggregated["personality_adaptations"] += metrics.get("personality_adaptations", 0)
                    
                    # Aggregate model usage
                    for model, count in metrics.get("model_usage", {}).items():
                        if model in aggregated["model_usage"]:
                            aggregated["model_usage"][model] += count
                    
                    # Sum for averaging
                    total_success_rate += metrics.get("success_rate", 1.0)
                    total_response_time += metrics.get("avg_response_time", 0.0)
            
            # Calculate averages
            if total_accounts > 0:
                aggregated["success_rate"] = total_success_rate / total_accounts
                aggregated["avg_response_time"] = total_response_time / total_accounts
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error getting aggregated metrics: {e}")
            return {
                "total_requests": 0,
                "total_tokens_used": 0,
                "cost_today": 0.0,
                "cost_this_month": 0.0,
                "model_usage": {"claude-3-sonnet-20240229": 0, "claude-3-haiku-20240307": 0, "claude-3-opus-20240229": 0},
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "conversations_enhanced": 0,
                "personality_adaptations": 0
            }
    
    async def get_account_config(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get Claude AI configuration for a specific account."""
        try:
            config_key = f"kelly:claude:config:{account_id}"
            config_data = await redis_manager.get(config_key)
            
            if not config_data:
                # Return default configuration
                default_config = {
                    "settings": {
                        "enabled": True,
                        "model_preference": "claude-3-sonnet-20240229",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "personality_strength": 0.8,
                        "safety_level": "high",
                        "context_memory": True,
                        "auto_adapt": True
                    },
                    "last_updated": datetime.now().isoformat(),
                    "performance_score": 0.85
                }
                
                # Save default config
                await redis_manager.setex(config_key, 86400 * 7, json.dumps(default_config, default=str))
                return default_config
            
            return json.loads(config_data)
            
        except Exception as e:
            logger.error(f"Error getting account config: {e}")
            return None
    
    async def update_account_config(self, account_id: str, config_data: Dict[str, Any]) -> bool:
        """Update Claude AI configuration for a specific account."""
        try:
            # Get existing config
            existing_config = await self.get_account_config(account_id)
            if not existing_config:
                existing_config = {"settings": {}, "performance_score": 0.85}
            
            # Update settings
            existing_config["settings"].update(config_data)
            existing_config["last_updated"] = datetime.now().isoformat()
            
            # Save updated config
            config_key = f"kelly:claude:config:{account_id}"
            await redis_manager.setex(
                config_key, 
                86400 * 7,  # 7 days
                json.dumps(existing_config, default=str)
            )
            
            logger.info(f"Updated Claude config for account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating account config: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the Claude AI service."""
        try:
            # Save all active conversation contexts
            for context in self.conversation_contexts.values():
                await self._save_conversation_context(context)
            
            # Save final usage stats
            await redis_manager.setex(
                "kelly:claude:usage_stats",
                86400,
                json.dumps(self.usage_stats, default=str)
            )
            
            logger.info("Kelly Claude AI service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Claude AI service shutdown: {e}")


# Global instance
kelly_claude_ai: Optional[KellyClaudeAI] = None


async def get_kelly_claude_ai() -> KellyClaudeAI:
    """Get the global Kelly Claude AI service instance."""
    global kelly_claude_ai
    if kelly_claude_ai is None:
        kelly_claude_ai = KellyClaudeAI()
        await kelly_claude_ai.initialize()
    return kelly_claude_ai


# Export main classes
__all__ = [
    'KellyClaudeAI',
    'ClaudeRequest',
    'ClaudeResponse',
    'ClaudeModel',
    'ConversationContext',
    'get_kelly_claude_ai'
]