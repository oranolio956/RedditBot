"""
Conversation Context Manager

Manages conversation context, memory, and flow for LLM interactions.
Integrates with the personality system and provides context-aware responses.

Features:
- Conversation memory with intelligent summarization
- Context window management for different LLM models
- Conversation flow tracking and state management
- Integration with personality adaptation
- Multi-turn conversation support
- Context relevance scoring and pruning
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.orm import selectinload

from app.config.settings import get_settings
from app.core.redis import get_redis_client
from app.models.conversation import (
    Message, ConversationSession, Conversation, 
    MessageType, MessageDirection, ConversationStatus
)
from app.models.user import User
from app.services.llm_service import LLMService, LLMRequest, LLMResponse, get_llm_service
from app.services.personality_engine import (
    AdvancedPersonalityEngine, ConversationContext, PersonalityState
)

logger = structlog.get_logger(__name__)


class ConversationPhase(str, Enum):
    """Conversation phase tracking."""
    GREETING = "greeting"
    CONTEXT_GATHERING = "context_gathering"
    PROBLEM_SOLVING = "problem_solving"
    INFORMATION_SHARING = "information_sharing"
    CLOSING = "closing"
    ONGOING = "ongoing"


class ContextPriority(str, Enum):
    """Context message priority levels."""
    CRITICAL = "critical"      # Always keep (user preferences, key facts)
    HIGH = "high"             # Keep unless space is tight
    MEDIUM = "medium"         # Keep if space allows
    LOW = "low"               # First to be pruned


@dataclass
class ConversationMemory:
    """Structured conversation memory."""
    user_id: str
    session_id: str
    conversation_id: str
    
    # Message history with priorities
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summarized context for long conversations
    context_summary: str = ""
    
    # Key facts and preferences extracted from conversation
    user_facts: Dict[str, Any] = field(default_factory=dict)
    conversation_facts: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation metadata
    current_phase: ConversationPhase = ConversationPhase.ONGOING
    current_topic: Optional[str] = None
    sentiment_trend: List[float] = field(default_factory=list)
    
    # Context management
    total_tokens_used: int = 0
    max_context_tokens: int = 4000
    last_pruned_at: Optional[datetime] = None


@dataclass
class ContextMessage:
    """Message with context metadata."""
    content: str
    role: str  # user, assistant, system
    timestamp: datetime
    message_id: Optional[str] = None
    priority: ContextPriority = ContextPriority.MEDIUM
    relevance_score: float = 1.0
    token_count: int = 0
    
    # Message metadata
    sentiment_score: Optional[float] = None
    topic: Optional[str] = None
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)


class ConversationManager:
    """
    Manages conversation context and memory for LLM interactions.
    
    Handles:
    - Context window management for different models
    - Intelligent message prioritization and pruning
    - Conversation flow tracking
    - Memory persistence and retrieval
    - Integration with personality system
    - Multi-turn conversation coherence
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings()
        self.redis = None
        self.llm_service: Optional[LLMService] = None
        
        # Memory storage
        self.active_conversations: Dict[str, ConversationMemory] = {}
        
        # Context management settings
        self.max_messages_per_conversation = 100
        self.context_window_buffer = 500  # Reserve tokens for response
        
        # Topic and intent tracking
        self.topic_keywords = {
            'technical': ['code', 'programming', 'software', 'bug', 'error', 'api'],
            'personal': ['feel', 'think', 'believe', 'opinion', 'experience'],
            'business': ['company', 'work', 'project', 'meeting', 'deadline'],
            'general': ['help', 'question', 'how', 'what', 'when', 'where', 'why']
        }
        
    async def initialize(self):
        """Initialize the conversation manager."""
        try:
            logger.info("Initializing conversation manager...")
            
            # Initialize Redis client
            self.redis = await get_redis_client()
            
            # Initialize LLM service
            self.llm_service = await get_llm_service()
            
            # Load active conversations from cache
            await self._load_active_conversations()
            
            logger.info("Conversation manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize conversation manager", error=str(e))
            raise
    
    async def get_conversation_context(
        self, 
        user_id: str, 
        conversation_id: str,
        max_tokens: int = 4000
    ) -> ConversationMemory:
        """
        Get or create conversation context for a user.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            max_tokens: Maximum tokens for context window
            
        Returns:
            ConversationMemory with optimized context
        """
        try:
            cache_key = f"conversation:{user_id}:{conversation_id}"
            
            # Check active memory first
            if cache_key in self.active_conversations:
                memory = self.active_conversations[cache_key]
                memory.max_context_tokens = max_tokens
                await self._optimize_context(memory)
                return memory
            
            # Try to load from database
            memory = await self._load_conversation_from_db(user_id, conversation_id)
            if memory:
                memory.max_context_tokens = max_tokens
                await self._optimize_context(memory)
                self.active_conversations[cache_key] = memory
                return memory
            
            # Create new conversation memory
            memory = ConversationMemory(
                user_id=user_id,
                session_id=conversation_id,  # Using conversation_id as session_id
                conversation_id=conversation_id,
                max_context_tokens=max_tokens
            )
            
            self.active_conversations[cache_key] = memory
            await self._save_conversation_to_cache(memory)
            
            logger.info(f"Created new conversation context for user {user_id}")
            return memory
            
        except Exception as e:
            logger.error("Error getting conversation context", error=str(e))
            # Return minimal context as fallback
            return ConversationMemory(
                user_id=user_id,
                session_id=conversation_id,
                conversation_id=conversation_id,
                max_context_tokens=max_tokens
            )
    
    async def add_message_to_context(
        self, 
        memory: ConversationMemory,
        content: str,
        role: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new message to conversation context.
        
        Args:
            memory: Conversation memory to update
            content: Message content
            role: Message role (user/assistant/system)
            message_id: Optional message ID for tracking
            metadata: Optional message metadata
        """
        try:
            # Create context message
            context_msg = ContextMessage(
                content=content,
                role=role,
                timestamp=datetime.now(),
                message_id=message_id,
                token_count=len(content.split()) * 1.3  # Rough token estimation
            )
            
            # Analyze message for metadata
            if metadata:
                context_msg.sentiment_score = metadata.get('sentiment_score')
                context_msg.topic = metadata.get('topic')
                context_msg.intent = metadata.get('intent')
                context_msg.entities = metadata.get('entities', [])
            else:
                # Quick analysis for priority and relevance
                await self._analyze_message(context_msg, memory)
            
            # Determine message priority
            context_msg.priority = await self._calculate_message_priority(context_msg, memory)
            
            # Add to memory
            message_dict = {
                'role': context_msg.role,
                'content': context_msg.content,
                'timestamp': context_msg.timestamp.isoformat(),
                'message_id': context_msg.message_id,
                'priority': context_msg.priority.value,
                'relevance_score': context_msg.relevance_score,
                'token_count': context_msg.token_count,
                'sentiment_score': context_msg.sentiment_score,
                'topic': context_msg.topic,
                'intent': context_msg.intent,
                'entities': context_msg.entities
            }
            
            memory.messages.append(message_dict)
            memory.total_tokens_used += context_msg.token_count
            
            # Update conversation metadata
            if context_msg.sentiment_score:
                memory.sentiment_trend.append(context_msg.sentiment_score)
                if len(memory.sentiment_trend) > 20:  # Keep last 20 sentiment scores
                    memory.sentiment_trend = memory.sentiment_trend[-20:]
            
            if context_msg.topic:
                memory.current_topic = context_msg.topic
            
            # Update conversation phase
            memory.current_phase = await self._detect_conversation_phase(memory)
            
            # Optimize context if needed
            if memory.total_tokens_used > memory.max_context_tokens:
                await self._optimize_context(memory)
            
            # Save to cache
            await self._save_conversation_to_cache(memory)
            
            logger.debug(
                "Added message to conversation context",
                role=role,
                tokens=context_msg.token_count,
                priority=context_msg.priority.value,
                total_tokens=memory.total_tokens_used
            )
            
        except Exception as e:
            logger.error("Error adding message to context", error=str(e))
    
    async def generate_contextual_response(
        self, 
        memory: ConversationMemory,
        user_message: str,
        personality_state: Optional[PersonalityState] = None,
        system_instructions: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a contextual response using LLM with conversation memory.
        
        Args:
            memory: Conversation memory
            user_message: Latest user message
            personality_state: Optional personality context
            system_instructions: Optional system-level instructions
            
        Returns:
            Tuple of (response_content, response_metadata)
        """
        try:
            # Add user message to context
            await self.add_message_to_context(memory, user_message, 'user')
            
            # Prepare messages for LLM
            llm_messages = await self._prepare_llm_messages(memory)
            
            # Create system prompt with context
            system_prompt = await self._create_system_prompt(memory, personality_state, system_instructions)
            
            # Create LLM request
            llm_request = LLMRequest(
                messages=llm_messages,
                system_prompt=system_prompt,
                user_id=memory.user_id,
                conversation_id=memory.conversation_id,
                personality_context=personality_state,
                temperature=self._calculate_temperature(memory),
                use_cache=True,
                priority=self._calculate_request_priority(memory)
            )
            
            # Generate response
            response = await self.llm_service.generate_response(llm_request)
            
            # Add assistant response to context
            await self.add_message_to_context(
                memory, 
                response.content, 
                'assistant',
                metadata={
                    'model_used': response.model_used.value,
                    'tokens_used': response.tokens_used,
                    'cost': response.cost_estimate,
                    'response_time_ms': response.response_time_ms,
                    'quality_score': response.quality_score
                }
            )
            
            # Update conversation facts
            await self._extract_conversation_facts(memory, user_message, response.content)
            
            # Prepare response metadata
            response_metadata = {
                'model_used': response.model_used.value,
                'tokens_used': response.tokens_used,
                'cost_estimate': response.cost_estimate,
                'response_time_ms': response.response_time_ms,
                'quality_score': response.quality_score,
                'conversation_phase': memory.current_phase.value,
                'current_topic': memory.current_topic,
                'context_tokens': memory.total_tokens_used,
                'sentiment_trend': memory.sentiment_trend[-5:] if memory.sentiment_trend else []
            }
            
            logger.info(
                "Generated contextual response",
                user_id=memory.user_id,
                model=response.model_used.value,
                response_time=response.response_time_ms,
                tokens=response.tokens_used.get('total', 0),
                phase=memory.current_phase.value
            )
            
            return response.content, response_metadata
            
        except Exception as e:
            logger.error("Error generating contextual response", error=str(e))
            # Return fallback response
            return (
                "I understand what you're saying. Let me help you with that.",
                {'error': str(e), 'fallback': True}
            )
    
    async def generate_streaming_response(
        self,
        memory: ConversationMemory,
        user_message: str,
        personality_state: Optional[PersonalityState] = None
    ):
        """Generate streaming contextual response."""
        try:
            # Add user message to context
            await self.add_message_to_context(memory, user_message, 'user')
            
            # Prepare messages for LLM
            llm_messages = await self._prepare_llm_messages(memory)
            
            # Create system prompt
            system_prompt = await self._create_system_prompt(memory, personality_state)
            
            # Create streaming LLM request
            llm_request = LLMRequest(
                messages=llm_messages,
                system_prompt=system_prompt,
                user_id=memory.user_id,
                conversation_id=memory.conversation_id,
                personality_context=personality_state,
                stream=True,
                temperature=self._calculate_temperature(memory)
            )
            
            # Collect response chunks
            response_content = ""
            async for chunk in self.llm_service.generate_streaming_response(llm_request):
                response_content += chunk
                yield chunk
            
            # Add complete response to context
            if response_content:
                await self.add_message_to_context(memory, response_content, 'assistant')
                
        except Exception as e:
            logger.error("Error generating streaming response", error=str(e))
            yield "I'm here to help! How can I assist you?"
    
    async def _load_conversation_from_db(
        self, 
        user_id: str, 
        conversation_id: str
    ) -> Optional[ConversationMemory]:
        """Load conversation memory from database."""
        try:
            # Get conversation with messages
            query = select(Conversation).where(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.id == conversation_id
                )
            ).options(selectinload(Conversation.messages))
            
            result = await self.db.execute(query)
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return None
            
            # Create memory from database data
            memory = ConversationMemory(
                user_id=user_id,
                session_id=conversation.session_id,
                conversation_id=conversation_id,
                current_topic=conversation.topic,
                context_summary=conversation.context_data.get('summary', '') if conversation.context_data else ''
            )
            
            # Load messages
            for msg in conversation.messages[-50:]:  # Load last 50 messages
                message_dict = {
                    'role': 'user' if msg.direction == MessageDirection.INCOMING else 'assistant',
                    'content': msg.content or '',
                    'timestamp': msg.created_at.isoformat() if msg.created_at else datetime.now().isoformat(),
                    'message_id': str(msg.id),
                    'priority': ContextPriority.MEDIUM.value,
                    'relevance_score': 1.0,
                    'token_count': len(msg.content.split()) * 1.3 if msg.content else 0,
                    'sentiment_score': msg.sentiment_score,
                    'topic': msg.intent_classification,
                    'intent': msg.intent_classification,
                    'entities': msg.keywords or []
                }
                memory.messages.append(message_dict)
                memory.total_tokens_used += message_dict['token_count']
            
            logger.info(f"Loaded conversation from database with {len(memory.messages)} messages")
            return memory
            
        except Exception as e:
            logger.error("Error loading conversation from database", error=str(e))
            return None
    
    async def _load_active_conversations(self):
        """Load active conversations from cache."""
        try:
            if not self.redis:
                return
                
            # Get list of active conversation keys
            pattern = "conversation:*"
            keys = await self.redis.keys(pattern)
            
            for key in keys:
                try:
                    data = await self.redis.get(key)
                    if data:
                        memory_data = json.loads(data)
                        memory = ConversationMemory(**memory_data)
                        self.active_conversations[key] = memory
                except Exception as e:
                    logger.error(f"Error loading conversation {key}", error=str(e))
                    
            logger.info(f"Loaded {len(self.active_conversations)} active conversations from cache")
            
        except Exception as e:
            logger.error("Error loading active conversations", error=str(e))
    
    async def _save_conversation_to_cache(self, memory: ConversationMemory):
        """Save conversation memory to cache."""
        try:
            if not self.redis:
                return
                
            cache_key = f"conversation:{memory.user_id}:{memory.conversation_id}"
            
            # Convert to JSON-serializable format
            memory_dict = {
                'user_id': memory.user_id,
                'session_id': memory.session_id,
                'conversation_id': memory.conversation_id,
                'messages': memory.messages,
                'context_summary': memory.context_summary,
                'user_facts': memory.user_facts,
                'conversation_facts': memory.conversation_facts,
                'current_phase': memory.current_phase.value,
                'current_topic': memory.current_topic,
                'sentiment_trend': memory.sentiment_trend,
                'total_tokens_used': memory.total_tokens_used,
                'max_context_tokens': memory.max_context_tokens,
                'last_pruned_at': memory.last_pruned_at.isoformat() if memory.last_pruned_at else None
            }
            
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(memory_dict)
            )
            
        except Exception as e:
            logger.error("Error saving conversation to cache", error=str(e))
    
    async def _analyze_message(self, message: ContextMessage, memory: ConversationMemory):
        """Perform quick analysis on message for priority and relevance."""
        try:
            content_lower = message.content.lower()
            
            # Sentiment analysis (simple keyword-based)
            positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'wonderful']
            negative_words = ['bad', 'terrible', 'hate', 'angry', 'frustrated', 'problem', 'error']
            
            pos_count = sum(1 for word in positive_words if word in content_lower)
            neg_count = sum(1 for word in negative_words if word in content_lower)
            
            if pos_count > neg_count:
                message.sentiment_score = min(1.0, 0.5 + (pos_count - neg_count) * 0.2)
            elif neg_count > pos_count:
                message.sentiment_score = max(-1.0, -0.5 - (neg_count - pos_count) * 0.2)
            else:
                message.sentiment_score = 0.0
            
            # Topic detection
            for topic, keywords in self.topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    message.topic = topic
                    break
            
            # Intent detection (simple patterns)
            if '?' in message.content:
                message.intent = 'question'
            elif any(word in content_lower for word in ['help', 'assist', 'support']):
                message.intent = 'help_request'
            elif any(word in content_lower for word in ['thank', 'thanks']):
                message.intent = 'gratitude'
            elif message.role == 'user':
                message.intent = 'information_sharing'
            
            # Calculate relevance score
            message.relevance_score = self._calculate_relevance_score(message, memory)
            
        except Exception as e:
            logger.error("Error analyzing message", error=str(e))
    
    def _calculate_relevance_score(self, message: ContextMessage, memory: ConversationMemory) -> float:
        """Calculate relevance score for message prioritization."""
        try:
            score = 1.0
            
            # Recent messages are more relevant
            if memory.messages:
                hours_old = (datetime.now() - message.timestamp).total_seconds() / 3600
                recency_factor = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
                score *= recency_factor
            
            # Topic continuity
            if message.topic and message.topic == memory.current_topic:
                score *= 1.5
            
            # Important intents get higher relevance
            if message.intent in ['help_request', 'question']:
                score *= 1.3
            elif message.intent == 'gratitude':
                score *= 0.8
            
            # System messages are generally less relevant for context
            if message.role == 'system':
                score *= 0.5
            
            return min(2.0, score)
            
        except Exception as e:
            logger.error("Error calculating relevance score", error=str(e))
            return 1.0
    
    async def _calculate_message_priority(
        self, 
        message: ContextMessage, 
        memory: ConversationMemory
    ) -> ContextPriority:
        """Calculate priority level for message."""
        try:
            # Critical messages (user preferences, key facts)
            if message.role == 'system':
                return ContextPriority.CRITICAL
            
            if any(keyword in message.content.lower() for keyword in 
                   ['preference', 'setting', 'remember', 'important', 'key']):
                return ContextPriority.CRITICAL
            
            # High priority messages
            if message.intent in ['help_request', 'question']:
                return ContextPriority.HIGH
            
            if message.relevance_score > 1.5:
                return ContextPriority.HIGH
            
            # Recent messages within current conversation flow
            if len(memory.messages) > 0:
                recent_threshold = datetime.now() - timedelta(minutes=30)
                if message.timestamp > recent_threshold:
                    return ContextPriority.MEDIUM
            
            # Low priority (older, less relevant messages)
            return ContextPriority.LOW
            
        except Exception as e:
            logger.error("Error calculating message priority", error=str(e))
            return ContextPriority.MEDIUM
    
    async def _optimize_context(self, memory: ConversationMemory):
        """Optimize conversation context to fit within token limits."""
        try:
            if memory.total_tokens_used <= memory.max_context_tokens:
                return  # No optimization needed
            
            # Sort messages by priority and relevance
            messages_with_scores = []
            for i, msg in enumerate(memory.messages):
                priority_score = {
                    'critical': 4,
                    'high': 3,
                    'medium': 2,
                    'low': 1
                }.get(msg.get('priority', 'medium'), 2)
                
                relevance = msg.get('relevance_score', 1.0)
                recency_bonus = max(0, len(memory.messages) - i) * 0.1  # Recent messages get bonus
                
                total_score = priority_score * 2 + relevance + recency_bonus
                messages_with_scores.append((i, msg, total_score))
            
            # Sort by score (highest first)
            messages_with_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Keep critical messages and highest scoring messages that fit
            optimized_messages = []
            token_count = 0
            target_tokens = memory.max_context_tokens - self.context_window_buffer
            
            # First pass: keep all critical messages
            for i, msg, score in messages_with_scores:
                if msg.get('priority') == 'critical':
                    optimized_messages.append(msg)
                    token_count += msg.get('token_count', 0)
            
            # Second pass: add other messages by score until we reach limit
            for i, msg, score in messages_with_scores:
                if msg.get('priority') != 'critical':
                    msg_tokens = msg.get('token_count', 0)
                    if token_count + msg_tokens <= target_tokens:
                        optimized_messages.append(msg)
                        token_count += msg_tokens
                    else:
                        break
            
            # Sort optimized messages by timestamp to maintain conversation flow
            optimized_messages.sort(key=lambda x: x.get('timestamp', ''))
            
            # Update memory
            pruned_count = len(memory.messages) - len(optimized_messages)
            memory.messages = optimized_messages
            memory.total_tokens_used = token_count
            memory.last_pruned_at = datetime.now()
            
            logger.info(
                "Optimized conversation context",
                original_messages=len(messages_with_scores),
                optimized_messages=len(optimized_messages),
                pruned_count=pruned_count,
                token_count=token_count,
                target_tokens=target_tokens
            )
            
            # Generate summary of pruned content if significant
            if pruned_count > 5:
                await self._generate_context_summary(memory, messages_with_scores)
                
        except Exception as e:
            logger.error("Error optimizing context", error=str(e))
    
    async def _generate_context_summary(self, memory: ConversationMemory, all_messages):
        """Generate summary of conversation context for long conversations."""
        try:
            # Extract key information from pruned messages
            pruned_messages = [msg for i, msg, score in all_messages if msg not in memory.messages]
            
            if not pruned_messages:
                return
            
            # Simple summarization (in production, could use LLM)
            topics = set()
            key_intents = []
            important_entities = []
            
            for msg in pruned_messages:
                if msg.get('topic'):
                    topics.add(msg['topic'])
                if msg.get('intent') in ['help_request', 'question']:
                    key_intents.append(msg.get('intent'))
                if msg.get('entities'):
                    important_entities.extend(msg['entities'])
            
            summary_parts = []
            if topics:
                summary_parts.append(f"Topics discussed: {', '.join(topics)}")
            if key_intents:
                summary_parts.append(f"User made {len(key_intents)} help requests/questions")
            if important_entities:
                unique_entities = list(set(important_entities))[:5]
                summary_parts.append(f"Key entities: {', '.join(unique_entities)}")
            
            if summary_parts:
                new_summary = ". ".join(summary_parts) + "."
                
                # Combine with existing summary
                if memory.context_summary:
                    memory.context_summary = f"{memory.context_summary} {new_summary}"
                else:
                    memory.context_summary = new_summary
                
                # Keep summary concise (max 200 words)
                summary_words = memory.context_summary.split()
                if len(summary_words) > 200:
                    memory.context_summary = " ".join(summary_words[-200:])
                
        except Exception as e:
            logger.error("Error generating context summary", error=str(e))
    
    async def _prepare_llm_messages(self, memory: ConversationMemory) -> List[Dict[str, str]]:
        """Prepare messages for LLM API call."""
        try:
            llm_messages = []
            
            # Add context summary if available
            if memory.context_summary:
                llm_messages.append({
                    'role': 'system',
                    'content': f"Previous conversation context: {memory.context_summary}"
                })
            
            # Add conversation messages
            for msg in memory.messages:
                llm_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            return llm_messages
            
        except Exception as e:
            logger.error("Error preparing LLM messages", error=str(e))
            return []
    
    async def _create_system_prompt(
        self, 
        memory: ConversationMemory,
        personality_state: Optional[PersonalityState] = None,
        additional_instructions: Optional[str] = None
    ) -> str:
        """Create system prompt with conversation context."""
        try:
            prompt_parts = []
            
            # Base instructions
            prompt_parts.append(
                "You are a helpful, intelligent assistant having a conversation with a user. "
                "Use the conversation history to provide contextual, coherent responses."
            )
            
            # Conversation phase context
            if memory.current_phase != ConversationPhase.ONGOING:
                phase_instructions = {
                    ConversationPhase.GREETING: "The user is just starting the conversation. Be welcoming and engaging.",
                    ConversationPhase.CONTEXT_GATHERING: "Focus on understanding the user's needs and gathering relevant information.",
                    ConversationPhase.PROBLEM_SOLVING: "Help the user solve their problem with clear, actionable advice.",
                    ConversationPhase.INFORMATION_SHARING: "Provide informative, accurate responses to the user's questions.",
                    ConversationPhase.CLOSING: "The conversation seems to be wrapping up. Be helpful but concise."
                }
                if memory.current_phase in phase_instructions:
                    prompt_parts.append(phase_instructions[memory.current_phase])
            
            # Topic context
            if memory.current_topic:
                prompt_parts.append(f"The current conversation topic is: {memory.current_topic}")
            
            # Sentiment context
            if memory.sentiment_trend:
                recent_sentiment = sum(memory.sentiment_trend[-3:]) / len(memory.sentiment_trend[-3:])
                if recent_sentiment > 0.2:
                    prompt_parts.append("The user seems to be in a positive mood.")
                elif recent_sentiment < -0.2:
                    prompt_parts.append("The user seems frustrated or negative. Be extra helpful and empathetic.")
            
            # User facts
            if memory.user_facts:
                facts_summary = []
                for key, value in list(memory.user_facts.items())[:3]:  # Limit to 3 key facts
                    facts_summary.append(f"{key}: {value}")
                if facts_summary:
                    prompt_parts.append(f"Key user information: {'; '.join(facts_summary)}")
            
            # Personality context
            if personality_state:
                traits = personality_state.adapted_traits
                personality_notes = []
                
                if traits.get('extraversion', 0.5) > 0.7:
                    personality_notes.append("be enthusiastic and engaging")
                elif traits.get('extraversion', 0.5) < 0.3:
                    personality_notes.append("be more reserved and thoughtful")
                
                if traits.get('agreeableness', 0.5) > 0.7:
                    personality_notes.append("be supportive and understanding")
                
                if traits.get('formality', 0.5) > 0.7:
                    personality_notes.append("use formal language")
                elif traits.get('formality', 0.5) < 0.3:
                    personality_notes.append("use casual, friendly language")
                
                if personality_notes:
                    prompt_parts.append(f"Personality guidance: {', '.join(personality_notes)}")
            
            # Additional instructions
            if additional_instructions:
                prompt_parts.append(additional_instructions)
            
            return " ".join(prompt_parts)
            
        except Exception as e:
            logger.error("Error creating system prompt", error=str(e))
            return "You are a helpful assistant. Provide thoughtful, contextual responses."
    
    async def _detect_conversation_phase(self, memory: ConversationMemory) -> ConversationPhase:
        """Detect current conversation phase."""
        try:
            if len(memory.messages) == 0:
                return ConversationPhase.GREETING
            
            if len(memory.messages) <= 2:
                return ConversationPhase.GREETING
            
            # Look at recent messages
            recent_messages = memory.messages[-5:]
            recent_content = " ".join([msg['content'].lower() for msg in recent_messages])
            
            # Check for closing indicators
            closing_words = ['bye', 'goodbye', 'thanks', 'thank you', 'that\'s all', 'done', 'finished']
            if any(word in recent_content for word in closing_words):
                return ConversationPhase.CLOSING
            
            # Check for problem-solving indicators
            problem_words = ['how', 'fix', 'solve', 'problem', 'error', 'issue', 'help me']
            if any(word in recent_content for word in problem_words):
                return ConversationPhase.PROBLEM_SOLVING
            
            # Check for information gathering
            question_count = sum(1 for msg in recent_messages if '?' in msg['content'])
            if question_count >= 2:
                return ConversationPhase.CONTEXT_GATHERING
            
            # Check for information sharing
            info_words = ['explain', 'tell me', 'what is', 'how does', 'information']
            if any(word in recent_content for word in info_words):
                return ConversationPhase.INFORMATION_SHARING
            
            return ConversationPhase.ONGOING
            
        except Exception as e:
            logger.error("Error detecting conversation phase", error=str(e))
            return ConversationPhase.ONGOING
    
    def _calculate_temperature(self, memory: ConversationMemory) -> float:
        """Calculate temperature based on conversation context."""
        try:
            base_temp = 0.7
            
            # Lower temperature for problem-solving (more focused)
            if memory.current_phase == ConversationPhase.PROBLEM_SOLVING:
                base_temp = 0.5
            
            # Higher temperature for creative/information sharing
            elif memory.current_phase == ConversationPhase.INFORMATION_SHARING:
                base_temp = 0.8
            
            # Adjust based on conversation length (more consistency for longer convos)
            if len(memory.messages) > 20:
                base_temp *= 0.9
            
            return max(0.1, min(1.0, base_temp))
            
        except Exception as e:
            logger.error("Error calculating temperature", error=str(e))
            return 0.7
    
    def _calculate_request_priority(self, memory: ConversationMemory) -> int:
        """Calculate request priority based on conversation context."""
        try:
            # High priority for problem-solving
            if memory.current_phase == ConversationPhase.PROBLEM_SOLVING:
                return 1
            
            # Medium priority for most interactions
            if memory.current_phase in [ConversationPhase.CONTEXT_GATHERING, ConversationPhase.INFORMATION_SHARING]:
                return 2
            
            # Low priority for casual/closing
            return 3
            
        except Exception as e:
            logger.error("Error calculating request priority", error=str(e))
            return 2
    
    async def _extract_conversation_facts(
        self, 
        memory: ConversationMemory,
        user_message: str,
        assistant_response: str
    ):
        """Extract and store key facts from conversation."""
        try:
            # Simple fact extraction (in production, could use NLP)
            user_lower = user_message.lower()
            
            # Extract user preferences
            if 'i like' in user_lower or 'i prefer' in user_lower:
                # Simple pattern matching
                preference_start = user_lower.find('i like')
                if preference_start == -1:
                    preference_start = user_lower.find('i prefer')
                
                if preference_start != -1:
                    preference_text = user_message[preference_start:preference_start+100]
                    memory.user_facts[f"preference_{len(memory.user_facts)}"] = preference_text
            
            # Extract user information
            if 'my name is' in user_lower:
                name_start = user_lower.find('my name is') + len('my name is')
                name_end = user_lower.find(' ', name_start + 1)
                if name_end == -1:
                    name_end = len(user_message)
                name = user_message[name_start:name_end].strip()
                memory.user_facts['name'] = name
            
            # Extract conversation facts from assistant response
            if 'you mentioned' in assistant_response.lower():
                memory.conversation_facts[f"mention_{len(memory.conversation_facts)}"] = assistant_response[:100]
            
        except Exception as e:
            logger.error("Error extracting conversation facts", error=str(e))
    
    async def get_conversation_summary(self, memory: ConversationMemory) -> Dict[str, Any]:
        """Get comprehensive conversation summary."""
        try:
            return {
                'conversation_id': memory.conversation_id,
                'user_id': memory.user_id,
                'message_count': len(memory.messages),
                'total_tokens': memory.total_tokens_used,
                'current_phase': memory.current_phase.value,
                'current_topic': memory.current_topic,
                'sentiment_trend': memory.sentiment_trend[-10:] if memory.sentiment_trend else [],
                'user_facts_count': len(memory.user_facts),
                'conversation_facts_count': len(memory.conversation_facts),
                'context_summary': memory.context_summary,
                'last_pruned': memory.last_pruned_at.isoformat() if memory.last_pruned_at else None
            }
        except Exception as e:
            logger.error("Error getting conversation summary", error=str(e))
            return {}
    
    async def cleanup_inactive_conversations(self, hours_threshold: int = 24):
        """Clean up inactive conversations from memory."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_threshold)
            inactive_keys = []
            
            for key, memory in self.active_conversations.items():
                if memory.messages:
                    last_message_time = datetime.fromisoformat(memory.messages[-1]['timestamp'])
                    if last_message_time < cutoff_time:
                        inactive_keys.append(key)
                else:
                    # No messages, consider inactive
                    inactive_keys.append(key)
            
            # Remove inactive conversations
            for key in inactive_keys:
                del self.active_conversations[key]
                
                # Remove from cache
                if self.redis:
                    await self.redis.delete(key)
            
            logger.info(f"Cleaned up {len(inactive_keys)} inactive conversations")
            
        except Exception as e:
            logger.error("Error cleaning up inactive conversations", error=str(e))
    
    async def shutdown(self):
        """Shutdown conversation manager and save state."""
        try:
            # Save all active conversations to cache
            for memory in self.active_conversations.values():
                await self._save_conversation_to_cache(memory)
            
            logger.info("Conversation manager shutdown complete")
            
        except Exception as e:
            logger.error("Error during conversation manager shutdown", error=str(e))


# Export main classes
__all__ = [
    'ConversationManager',
    'ConversationMemory',
    'ContextMessage',
    'ConversationPhase',
    'ContextPriority'
]