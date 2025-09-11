"""
Kelly Database Service - Advanced Database Integration

Provides sophisticated database operations for Kelly's conversation system:
- PostgreSQL integration with conversation history
- Redis caching for real-time performance
- User profile and conversation stage management
- AI insights storage and retrieval
- Conversation analytics and pattern analysis
- Scalable data architecture for millions of conversations
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, and_, or_, func, text
import structlog

from app.config.settings import get_settings
from app.core.redis import redis_manager
from app.models.conversation import ConversationSession, Message, UserProfile
from app.services.kelly_claude_ai import ConversationContext, ClaudeResponse

logger = structlog.get_logger(__name__)


class ConversationStage(str, Enum):
    """Conversation relationship stages."""
    INITIAL = "initial"
    BUILDING_RAPPORT = "building_rapport"
    GETTING_ACQUAINTED = "getting_acquainted"
    DEEPENING_CONNECTION = "deepening_connection"
    INTIMATE_CONVERSATION = "intimate_conversation"
    LONG_TERM_ENGAGEMENT = "long_term_engagement"
    COOLING_OFF = "cooling_off"
    INACTIVE = "inactive"


class MessageType(str, Enum):
    """Types of messages in conversations."""
    USER_MESSAGE = "user_message"
    KELLY_RESPONSE = "kelly_response"
    SYSTEM_EVENT = "system_event"
    SAFETY_INTERVENTION = "safety_intervention"
    ESCALATION = "escalation"


@dataclass
class ConversationMetrics:
    """Metrics for conversation analysis."""
    total_messages: int
    user_messages: int
    kelly_responses: int
    avg_response_time: float
    sentiment_trend: List[float]
    engagement_score: float
    topics_discussed: List[str]
    conversation_duration_hours: float
    last_activity: datetime


@dataclass
class UserInsights:
    """AI-generated insights about user behavior."""
    personality_traits: Dict[str, float]
    communication_style: Dict[str, float]
    interests: List[str]
    emotional_patterns: Dict[str, Any]
    conversation_preferences: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    engagement_likelihood: float
    optimal_response_style: Dict[str, Any]


@dataclass
class ConversationAnalytics:
    """Advanced conversation analytics."""
    conversation_id: str
    user_id: str
    stage: ConversationStage
    metrics: ConversationMetrics
    user_insights: UserInsights
    ai_recommendations: Dict[str, Any]
    last_updated: datetime


class KellyDatabase:
    """
    Advanced database service for Kelly's conversation system.
    
    Features:
    - High-performance PostgreSQL operations
    - Redis caching for real-time data access
    - Conversation history management
    - User profile and insights storage
    - AI analytics and recommendations
    - Scalable architecture for millions of users
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.postgres_engine = None
        self.async_session_factory = None
        self.redis_client = None
        
        # Cache configuration
        self.cache_config = {
            "conversation_ttl": 3600,      # 1 hour
            "user_profile_ttl": 7200,     # 2 hours
            "analytics_ttl": 1800,        # 30 minutes
            "insights_ttl": 86400,        # 24 hours
            "message_history_ttl": 3600   # 1 hour
        }
        
        # Database connection pools
        self.postgres_pool = None
        self.connection_config = {
            "min_connections": 5,
            "max_connections": 20,
            "command_timeout": 60,
            "server_settings": {
                "jit": "off",  # Optimize for OLTP workloads
                "application_name": "kelly_ai_system"
            }
        }
    
    async def initialize(self):
        """Initialize database connections and setup."""
        try:
            logger.info("Initializing Kelly Database service...")
            
            # Initialize PostgreSQL connection
            await self._initialize_postgres()
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Setup database schema if needed
            await self._ensure_database_schema()
            
            # Initialize performance monitoring
            await self._setup_performance_monitoring()
            
            logger.info("Kelly Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kelly Database service: {e}")
            raise
    
    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection."""
        try:
            # Get database URL from settings
            database_url = getattr(self.settings, 'DATABASE_URL', None)
            if not database_url:
                # Construct from individual components
                db_config = {
                    'host': getattr(self.settings, 'DB_HOST', 'localhost'),
                    'port': getattr(self.settings, 'DB_PORT', 5432),
                    'user': getattr(self.settings, 'DB_USER', 'postgres'),
                    'password': getattr(self.settings, 'DB_PASSWORD', ''),
                    'database': getattr(self.settings, 'DB_NAME', 'telegram_bot')
                }
                database_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
            # Create async engine
            self.postgres_engine = create_async_engine(
                database_url,
                pool_size=self.connection_config["max_connections"],
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create connection pool for direct queries
            self.postgres_pool = await asyncpg.create_pool(
                database_url,
                min_size=self.connection_config["min_connections"],
                max_size=self.connection_config["max_connections"],
                command_timeout=self.connection_config["command_timeout"],
                server_settings=self.connection_config["server_settings"]
            )
            
            logger.info("PostgreSQL connection initialized")
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis_manager
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
            raise
    
    async def _ensure_database_schema(self):
        """Ensure database schema exists."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Create tables if they don't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS kelly_conversations (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        account_id TEXT NOT NULL,
                        stage TEXT DEFAULT 'initial',
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        last_activity TIMESTAMPTZ DEFAULT NOW(),
                        total_messages INTEGER DEFAULT 0,
                        user_messages INTEGER DEFAULT 0,
                        kelly_responses INTEGER DEFAULT 0,
                        metadata JSONB DEFAULT '{}',
                        personality_adaptation JSONB DEFAULT '{}',
                        safety_flags TEXT[] DEFAULT '{}',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS kelly_messages (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        conversation_uuid UUID REFERENCES kelly_conversations(id),
                        user_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        sender_role TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        model_used TEXT,
                        tokens_used JSONB,
                        cost_estimate DECIMAL(10,6),
                        response_time_ms INTEGER,
                        safety_score DECIMAL(3,2),
                        sentiment_score DECIMAL(3,2),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS kelly_user_profiles (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id TEXT UNIQUE NOT NULL,
                        username TEXT,
                        first_name TEXT,
                        last_name TEXT,
                        personality_traits JSONB DEFAULT '{}',
                        communication_style JSONB DEFAULT '{}',
                        interests TEXT[] DEFAULT '{}',
                        conversation_preferences JSONB DEFAULT '{}',
                        risk_assessment JSONB DEFAULT '{}',
                        engagement_score DECIMAL(3,2) DEFAULT 0.5,
                        total_conversations INTEGER DEFAULT 0,
                        total_messages INTEGER DEFAULT 0,
                        avg_session_duration DECIMAL(10,2) DEFAULT 0,
                        last_seen TIMESTAMPTZ,
                        is_blocked BOOLEAN DEFAULT FALSE,
                        block_reason TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS kelly_conversation_analytics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        conversation_uuid UUID REFERENCES kelly_conversations(id),
                        user_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        metrics JSONB NOT NULL,
                        user_insights JSONB NOT NULL,
                        ai_recommendations JSONB NOT NULL,
                        analysis_timestamp TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS kelly_system_events (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        event_type TEXT NOT NULL,
                        user_id TEXT,
                        conversation_id TEXT,
                        account_id TEXT,
                        event_data JSONB NOT NULL,
                        severity TEXT DEFAULT 'info',
                        requires_attention BOOLEAN DEFAULT FALSE,
                        handled BOOLEAN DEFAULT FALSE,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_kelly_conversations_user_id ON kelly_conversations(user_id);
                    CREATE INDEX IF NOT EXISTS idx_kelly_conversations_account_id ON kelly_conversations(account_id);
                    CREATE INDEX IF NOT EXISTS idx_kelly_conversations_stage ON kelly_conversations(stage);
                    CREATE INDEX IF NOT EXISTS idx_kelly_conversations_last_activity ON kelly_conversations(last_activity);
                    
                    CREATE INDEX IF NOT EXISTS idx_kelly_messages_conversation_uuid ON kelly_messages(conversation_uuid);
                    CREATE INDEX IF NOT EXISTS idx_kelly_messages_user_id ON kelly_messages(user_id);
                    CREATE INDEX IF NOT EXISTS idx_kelly_messages_timestamp ON kelly_messages(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_kelly_messages_message_type ON kelly_messages(message_type);
                    
                    CREATE INDEX IF NOT EXISTS idx_kelly_user_profiles_user_id ON kelly_user_profiles(user_id);
                    CREATE INDEX IF NOT EXISTS idx_kelly_user_profiles_engagement_score ON kelly_user_profiles(engagement_score);
                    CREATE INDEX IF NOT EXISTS idx_kelly_user_profiles_last_seen ON kelly_user_profiles(last_seen);
                    
                    CREATE INDEX IF NOT EXISTS idx_kelly_analytics_conversation_uuid ON kelly_conversation_analytics(conversation_uuid);
                    CREATE INDEX IF NOT EXISTS idx_kelly_analytics_user_id ON kelly_conversation_analytics(user_id);
                    CREATE INDEX IF NOT EXISTS idx_kelly_analytics_timestamp ON kelly_conversation_analytics(analysis_timestamp);
                    
                    CREATE INDEX IF NOT EXISTS idx_kelly_events_timestamp ON kelly_system_events(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_kelly_events_requires_attention ON kelly_system_events(requires_attention);
                    CREATE INDEX IF NOT EXISTS idx_kelly_events_event_type ON kelly_system_events(event_type);
                """)
                
            logger.info("Database schema ensured")
            
        except Exception as e:
            logger.error(f"Error ensuring database schema: {e}")
            raise
    
    async def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        try:
            # Initialize performance counters in Redis
            await self.redis_client.hset("kelly:db:performance", mapping={
                "total_queries": 0,
                "avg_query_time": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "active_connections": 0
            })
            
            logger.info("Performance monitoring setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up performance monitoring: {e}")
    
    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str,
        account_id: str,
        initial_message: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation session."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Create conversation record
                conversation_uuid = await conn.fetchval("""
                    INSERT INTO kelly_conversations (
                        user_id, conversation_id, account_id, stage,
                        started_at, last_activity, total_messages,
                        user_messages, kelly_responses, metadata
                    ) VALUES ($1, $2, $3, $4, NOW(), NOW(), 1, 1, 0, $5)
                    RETURNING id
                """, user_id, conversation_id, account_id, ConversationStage.INITIAL.value, json.dumps({}))
                
                # Store initial message
                await conn.execute("""
                    INSERT INTO kelly_messages (
                        conversation_uuid, user_id, message_type, content,
                        sender_role, timestamp, metadata
                    ) VALUES ($1, $2, $3, $4, $5, NOW(), $6)
                """, conversation_uuid, user_id, MessageType.USER_MESSAGE.value, 
                    initial_message, "user", json.dumps({}))
                
                # Create or update user profile
                await self._upsert_user_profile(user_id, user_profile or {})
                
                # Cache conversation context
                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=[{
                        "role": "user",
                        "content": initial_message,
                        "timestamp": datetime.now().isoformat()
                    }],
                    relationship_stage=ConversationStage.INITIAL.value,
                    personality_adaptation={
                        "formality": 0.6,
                        "enthusiasm": 0.7,
                        "playfulness": 0.5,
                        "emotional_depth": 0.6,
                        "intellectual_level": 0.7
                    },
                    conversation_metadata={
                        "conversation_uuid": str(conversation_uuid),
                        "account_id": account_id,
                        "started_at": datetime.now().isoformat()
                    },
                    safety_flags=[],
                    last_updated=datetime.now()
                )
                
                await self._cache_conversation_context(context)
                
                logger.info(f"Created conversation {conversation_uuid} for user {user_id}")
                return str(conversation_uuid)
                
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def add_message(
        self,
        conversation_uuid: str,
        user_id: str,
        content: str,
        sender_role: str,
        message_type: MessageType = MessageType.USER_MESSAGE,
        claude_response: Optional[ClaudeResponse] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to a conversation."""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Insert message
                message_id = await conn.fetchval("""
                    INSERT INTO kelly_messages (
                        conversation_uuid, user_id, message_type, content, sender_role,
                        timestamp, model_used, tokens_used, cost_estimate,
                        response_time_ms, safety_score, metadata
                    ) VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, 
                    uuid.UUID(conversation_uuid), user_id, message_type.value, content, sender_role,
                    claude_response.model_used.value if claude_response else None,
                    json.dumps(claude_response.tokens_used) if claude_response else None,
                    claude_response.cost_estimate if claude_response else None,
                    claude_response.response_time_ms if claude_response else None,
                    claude_response.safety_score if claude_response else None,
                    json.dumps(metadata or {})
                )
                
                # Update conversation statistics
                if sender_role == "user":
                    await conn.execute("""
                        UPDATE kelly_conversations 
                        SET user_messages = user_messages + 1,
                            total_messages = total_messages + 1,
                            last_activity = NOW(),
                            updated_at = NOW()
                        WHERE id = $1
                    """, uuid.UUID(conversation_uuid))
                elif sender_role == "assistant":
                    await conn.execute("""
                        UPDATE kelly_conversations 
                        SET kelly_responses = kelly_responses + 1,
                            total_messages = total_messages + 1,
                            last_activity = NOW(),
                            updated_at = NOW()
                        WHERE id = $1
                    """, uuid.UUID(conversation_uuid))
                
                # Update conversation context cache
                await self._update_conversation_context_cache(
                    conversation_uuid, content, sender_role, claude_response
                )
                
                logger.debug(f"Added message {message_id} to conversation {conversation_uuid}")
                return str(message_id)
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise
    
    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 50,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        try:
            # Check cache first
            cache_key = f"kelly:history:{user_id}_{conversation_id}:{limit}"
            cached_history = await self.redis_client.get(cache_key)
            
            if cached_history:
                await self._track_cache_hit("conversation_history")
                return json.loads(cached_history)
            
            await self._track_cache_miss("conversation_history")
            
            # Query database
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT m.content, m.sender_role, m.timestamp, m.message_type,
                           m.model_used, m.tokens_used, m.cost_estimate,
                           m.response_time_ms, m.safety_score
                """
                if include_metadata:
                    query += ", m.metadata"
                
                query += """
                    FROM kelly_messages m
                    JOIN kelly_conversations c ON m.conversation_uuid = c.id
                    WHERE c.user_id = $1 AND c.conversation_id = $2
                    ORDER BY m.timestamp ASC
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, user_id, conversation_id, limit)
                
                messages = []
                for row in rows:
                    message = {
                        "content": row["content"],
                        "role": row["sender_role"],
                        "timestamp": row["timestamp"].isoformat(),
                        "message_type": row["message_type"],
                        "model_used": row["model_used"],
                        "tokens_used": json.loads(row["tokens_used"]) if row["tokens_used"] else None,
                        "cost_estimate": float(row["cost_estimate"]) if row["cost_estimate"] else None,
                        "response_time_ms": row["response_time_ms"],
                        "safety_score": float(row["safety_score"]) if row["safety_score"] else None
                    }
                    
                    if include_metadata and row.get("metadata"):
                        message["metadata"] = json.loads(row["metadata"])
                    
                    messages.append(message)
                
                # Cache the result
                await self.redis_client.setex(
                    cache_key,
                    self.cache_config["message_history_ttl"],
                    json.dumps(messages, default=str)
                )
                
                return messages
                
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with insights."""
        try:
            # Check cache first
            cache_key = f"kelly:profile:{user_id}"
            cached_profile = await self.redis_client.get(cache_key)
            
            if cached_profile:
                await self._track_cache_hit("user_profile")
                return json.loads(cached_profile)
            
            await self._track_cache_miss("user_profile")
            
            # Query database
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT user_id, username, first_name, last_name,
                           personality_traits, communication_style, interests,
                           conversation_preferences, risk_assessment,
                           engagement_score, total_conversations, total_messages,
                           avg_session_duration, last_seen, is_blocked, block_reason,
                           created_at, updated_at
                    FROM kelly_user_profiles
                    WHERE user_id = $1
                """, user_id)
                
                if not row:
                    return None
                
                profile = {
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "personality_traits": json.loads(row["personality_traits"]) if row["personality_traits"] else {},
                    "communication_style": json.loads(row["communication_style"]) if row["communication_style"] else {},
                    "interests": row["interests"] or [],
                    "conversation_preferences": json.loads(row["conversation_preferences"]) if row["conversation_preferences"] else {},
                    "risk_assessment": json.loads(row["risk_assessment"]) if row["risk_assessment"] else {},
                    "engagement_score": float(row["engagement_score"]) if row["engagement_score"] else 0.5,
                    "total_conversations": row["total_conversations"] or 0,
                    "total_messages": row["total_messages"] or 0,
                    "avg_session_duration": float(row["avg_session_duration"]) if row["avg_session_duration"] else 0,
                    "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
                    "is_blocked": row["is_blocked"] or False,
                    "block_reason": row["block_reason"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
                
                # Cache the result
                await self.redis_client.setex(
                    cache_key,
                    self.cache_config["user_profile_ttl"],
                    json.dumps(profile, default=str)
                )
                
                return profile
                
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    async def update_conversation_stage(
        self,
        conversation_uuid: str,
        new_stage: ConversationStage,
        personality_adaptations: Optional[Dict[str, float]] = None
    ):
        """Update conversation stage and personality adaptations."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE kelly_conversations 
                    SET stage = $1,
                        personality_adaptation = $2,
                        updated_at = NOW()
                    WHERE id = $3
                """, new_stage.value, 
                    json.dumps(personality_adaptations or {}),
                    uuid.UUID(conversation_uuid))
                
                # Update cache
                await self._invalidate_conversation_cache(conversation_uuid)
                
                logger.info(f"Updated conversation {conversation_uuid} stage to {new_stage.value}")
                
        except Exception as e:
            logger.error(f"Error updating conversation stage: {e}")
            raise
    
    async def store_conversation_analytics(
        self,
        conversation_uuid: str,
        user_id: str,
        analytics: ConversationAnalytics
    ):
        """Store conversation analytics and insights."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kelly_conversation_analytics (
                        conversation_uuid, user_id, stage, metrics,
                        user_insights, ai_recommendations, analysis_timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """, 
                    uuid.UUID(conversation_uuid), user_id, analytics.stage.value,
                    json.dumps(analytics.metrics.__dict__, default=str),
                    json.dumps(analytics.user_insights.__dict__, default=str),
                    json.dumps(analytics.ai_recommendations, default=str)
                )
                
                # Cache analytics
                cache_key = f"kelly:analytics:{conversation_uuid}"
                await self.redis_client.setex(
                    cache_key,
                    self.cache_config["analytics_ttl"],
                    json.dumps(analytics.__dict__, default=str)
                )
                
                logger.debug(f"Stored analytics for conversation {conversation_uuid}")
                
        except Exception as e:
            logger.error(f"Error storing conversation analytics: {e}")
            raise
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        try:
            cache_key = f"kelly:user_conversations:{user_id}:{limit}:{include_inactive}"
            cached_conversations = await self.redis_client.get(cache_key)
            
            if cached_conversations:
                await self._track_cache_hit("user_conversations")
                return json.loads(cached_conversations)
            
            await self._track_cache_miss("user_conversations")
            
            async with self.postgres_pool.acquire() as conn:
                where_clause = "WHERE user_id = $1"
                params = [user_id]
                
                if not include_inactive:
                    where_clause += " AND is_active = TRUE"
                
                query = f"""
                    SELECT id, conversation_id, account_id, stage, started_at,
                           last_activity, total_messages, user_messages,
                           kelly_responses, metadata, personality_adaptation,
                           safety_flags, is_active
                    FROM kelly_conversations
                    {where_clause}
                    ORDER BY last_activity DESC
                    LIMIT ${len(params) + 1}
                """
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                conversations = []
                for row in rows:
                    conversation = {
                        "id": str(row["id"]),
                        "conversation_id": row["conversation_id"],
                        "account_id": row["account_id"],
                        "stage": row["stage"],
                        "started_at": row["started_at"].isoformat(),
                        "last_activity": row["last_activity"].isoformat(),
                        "total_messages": row["total_messages"],
                        "user_messages": row["user_messages"],
                        "kelly_responses": row["kelly_responses"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "personality_adaptation": json.loads(row["personality_adaptation"]) if row["personality_adaptation"] else {},
                        "safety_flags": row["safety_flags"] or [],
                        "is_active": row["is_active"]
                    }
                    conversations.append(conversation)
                
                # Cache result
                await self.redis_client.setex(
                    cache_key,
                    self.cache_config["conversation_ttl"],
                    json.dumps(conversations, default=str)
                )
                
                return conversations
                
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    async def log_system_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        event_data: Dict[str, Any] = None,
        severity: str = "info",
        requires_attention: bool = False
    ):
        """Log system events for monitoring and analysis."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kelly_system_events (
                        event_type, user_id, conversation_id, account_id,
                        event_data, severity, requires_attention, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """, event_type, user_id, conversation_id, account_id,
                    json.dumps(event_data or {}), severity, requires_attention)
                
                # If requires attention, also cache for quick access
                if requires_attention:
                    alert_key = f"kelly:alerts:{datetime.now().strftime('%Y%m%d')}"
                    alert_data = {
                        "event_type": event_type,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "account_id": account_id,
                        "event_data": event_data or {},
                        "severity": severity,
                        "timestamp": datetime.now().isoformat()
                    }
                    await self.redis_client.lpush(alert_key, json.dumps(alert_data))
                    await self.redis_client.expire(alert_key, 86400)  # 24 hours
                
                logger.info(f"Logged system event: {event_type}", severity=severity)
                
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    async def _upsert_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Create or update user profile."""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kelly_user_profiles (
                        user_id, username, first_name, last_name,
                        personality_traits, communication_style, interests,
                        conversation_preferences, risk_assessment,
                        engagement_score, total_conversations, total_messages,
                        last_seen, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 1, 1, NOW(), NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        username = COALESCE(EXCLUDED.username, kelly_user_profiles.username),
                        first_name = COALESCE(EXCLUDED.first_name, kelly_user_profiles.first_name),
                        last_name = COALESCE(EXCLUDED.last_name, kelly_user_profiles.last_name),
                        total_conversations = kelly_user_profiles.total_conversations + 1,
                        total_messages = kelly_user_profiles.total_messages + 1,
                        last_seen = NOW(),
                        updated_at = NOW()
                """, 
                    user_id,
                    profile_data.get("username"),
                    profile_data.get("first_name"),
                    profile_data.get("last_name"),
                    json.dumps(profile_data.get("personality_traits", {})),
                    json.dumps(profile_data.get("communication_style", {})),
                    profile_data.get("interests", []),
                    json.dumps(profile_data.get("conversation_preferences", {})),
                    json.dumps(profile_data.get("risk_assessment", {})),
                    profile_data.get("engagement_score", 0.5)
                )
                
                # Invalidate cache
                await self.redis_client.delete(f"kelly:profile:{user_id}")
                
        except Exception as e:
            logger.error(f"Error upserting user profile: {e}")
            raise
    
    async def _cache_conversation_context(self, context: ConversationContext):
        """Cache conversation context in Redis."""
        try:
            cache_key = f"kelly:context:{context.user_id}_{context.conversation_id}"
            context_data = {
                "messages": context.messages,
                "relationship_stage": context.relationship_stage,
                "personality_adaptation": context.personality_adaptation,
                "conversation_metadata": context.conversation_metadata,
                "safety_flags": context.safety_flags,
                "last_updated": context.last_updated.isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                self.cache_config["conversation_ttl"],
                json.dumps(context_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error caching conversation context: {e}")
    
    async def _update_conversation_context_cache(
        self,
        conversation_uuid: str,
        content: str,
        sender_role: str,
        claude_response: Optional[ClaudeResponse] = None
    ):
        """Update conversation context cache with new message."""
        try:
            # This would update the cached context with the new message
            # Implementation depends on how context keys are structured
            pass
            
        except Exception as e:
            logger.error(f"Error updating conversation context cache: {e}")
    
    async def _invalidate_conversation_cache(self, conversation_uuid: str):
        """Invalidate conversation-related caches."""
        try:
            # Get conversation details to find cache keys to invalidate
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT user_id, conversation_id FROM kelly_conversations WHERE id = $1
                """, uuid.UUID(conversation_uuid))
                
                if row:
                    user_id = row["user_id"]
                    conversation_id = row["conversation_id"]
                    
                    # Invalidate relevant caches
                    cache_keys = [
                        f"kelly:context:{user_id}_{conversation_id}",
                        f"kelly:analytics:{conversation_uuid}",
                        f"kelly:user_conversations:{user_id}:*"
                    ]
                    
                    for key in cache_keys:
                        if "*" in key:
                            # Pattern-based deletion (would need to scan in real implementation)
                            pass
                        else:
                            await self.redis_client.delete(key)
                            
        except Exception as e:
            logger.error(f"Error invalidating conversation cache: {e}")
    
    async def _track_cache_hit(self, operation: str):
        """Track cache hit for performance monitoring."""
        try:
            await self.redis_client.hincrby("kelly:db:performance", "cache_hits", 1)
            await self.redis_client.hincrby(f"kelly:db:cache_hits:{operation}", "count", 1)
        except Exception as e:
            logger.error(f"Error tracking cache hit: {e}")
    
    async def _track_cache_miss(self, operation: str):
        """Track cache miss for performance monitoring."""
        try:
            await self.redis_client.hincrby("kelly:db:performance", "cache_misses", 1)
            await self.redis_client.hincrby(f"kelly:db:cache_misses:{operation}", "count", 1)
        except Exception as e:
            logger.error(f"Error tracking cache miss: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        try:
            performance_data = await self.redis_client.hgetall("kelly:db:performance")
            
            # Get connection pool stats
            pool_stats = {
                "postgres_pool_size": self.postgres_pool.get_size() if self.postgres_pool else 0,
                "postgres_pool_free": self.postgres_pool.get_idle_size() if self.postgres_pool else 0,
            }
            
            return {
                "cache_stats": {k.decode(): int(v) for k, v in performance_data.items()},
                "connection_pools": pool_stats,
                "cache_config": self.cache_config
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old conversation data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with self.postgres_pool.acquire() as conn:
                # Delete old messages
                deleted_messages = await conn.fetchval("""
                    DELETE FROM kelly_messages 
                    WHERE timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                # Delete old analytics
                deleted_analytics = await conn.fetchval("""
                    DELETE FROM kelly_conversation_analytics 
                    WHERE analysis_timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                # Mark old conversations as inactive
                updated_conversations = await conn.fetchval("""
                    UPDATE kelly_conversations 
                    SET is_active = FALSE 
                    WHERE last_activity < $1 AND is_active = TRUE
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                logger.info(f"Cleanup complete: {deleted_messages} messages, {deleted_analytics} analytics, {updated_conversations} conversations")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def shutdown(self):
        """Shutdown database connections."""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.postgres_engine:
                await self.postgres_engine.dispose()
            
            logger.info("Kelly Database service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")


# Global instance
kelly_database: Optional[KellyDatabase] = None


async def get_kelly_database() -> KellyDatabase:
    """Get the global Kelly Database service instance."""
    global kelly_database
    if kelly_database is None:
        kelly_database = KellyDatabase()
        await kelly_database.initialize()
    return kelly_database


# Export main classes
__all__ = [
    'KellyDatabase',
    'ConversationStage',
    'MessageType',
    'ConversationMetrics',
    'UserInsights',
    'ConversationAnalytics',
    'get_kelly_database'
]