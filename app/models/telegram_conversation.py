"""
Telegram Conversation Management Models
Advanced models for tracking conversations and context across communities.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, JSON, Float, Text,
    ForeignKey, Enum as SQLEnum, BigInteger, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

from app.database.base import BaseModel


class MessageType(str, Enum):
    """Type of message"""
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACT = "contact"
    POLL = "poll"
    ANIMATION = "animation"


class ConversationStatus(str, Enum):
    """Status of conversation"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


class MessageDirection(str, Enum):
    """Direction of message flow"""
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class TelegramConversation(BaseModel):
    """
    Conversation tracking for context-aware responses
    """
    __tablename__ = "telegram_conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("telegram_accounts.id"), nullable=False)
    community_id = Column(UUID(as_uuid=True), ForeignKey("telegram_communities.id"), nullable=True)
    
    # Conversation Identity
    chat_id = Column(BigInteger, nullable=False)
    thread_id = Column(BigInteger, nullable=True)  # For topic groups
    conversation_hash = Column(String(64), nullable=False)  # Unique conversation identifier
    
    # Conversation Metadata
    title = Column(String(255), nullable=True)
    participants = Column(JSON, default=list)  # List of user IDs in conversation
    is_group = Column(Boolean, default=False)
    is_private = Column(Boolean, default=False)
    
    # Conversation State
    status = Column(SQLEnum(ConversationStatus), default=ConversationStatus.ACTIVE)
    last_message_date = Column(DateTime, nullable=True)
    message_count = Column(Integer, default=0)
    our_message_count = Column(Integer, default=0)
    
    # Context and Memory
    conversation_context = Column(JSON, default=dict)  # Context for AI responses
    topics_discussed = Column(JSON, default=list)  # Main topics in conversation
    sentiment_history = Column(JSON, default=list)  # Sentiment progression
    
    # AI Configuration
    personality_snapshot = Column(JSON, nullable=True)  # Personality used in this conversation
    response_style = Column(String(50), default="adaptive")
    memory_weight = Column(Float, default=1.0)  # Importance for memory retention
    
    # Performance Metrics
    engagement_score = Column(Float, default=0.0)
    response_rate = Column(Float, default=0.0)  # How often others respond to us
    avg_response_time = Column(Integer, nullable=True)  # Average response time in seconds
    
    # Safety and Compliance
    contains_sensitive_data = Column(Boolean, default=False)
    gdpr_retention_date = Column(DateTime, nullable=True)
    compliance_tags = Column(JSON, default=list)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TelegramAccount", back_populates="conversations")
    community = relationship("TelegramCommunity", back_populates="conversations")
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_conversations_account', 'account_id'),
        Index('ix_conversations_community', 'community_id'),
        Index('ix_conversations_chat', 'chat_id'),
        Index('ix_conversations_hash', 'conversation_hash'),
        Index('ix_conversations_status', 'status'),
        Index('ix_conversations_last_message', 'last_message_date'),
    )
    
    def __repr__(self):
        return f"<TelegramConversation(chat_id={self.chat_id}, messages={self.message_count})>"
    
    @property
    def is_recent(self) -> bool:
        """Check if conversation has recent activity"""
        if not self.last_message_date:
            return False
        return datetime.utcnow() - self.last_message_date < timedelta(hours=24)
    
    @property
    def should_respond(self) -> bool:
        """Determine if we should respond to this conversation"""
        return (
            self.status == ConversationStatus.ACTIVE and
            self.is_recent and
            self.response_rate > 0.1  # Others respond to us at least 10% of the time
        )
    
    def add_context(self, key: str, value: Any, expire_hours: int = 24):
        """Add context with expiration"""
        if self.conversation_context is None:
            self.conversation_context = {}
        
        expires_at = datetime.utcnow() + timedelta(hours=expire_hours)
        self.conversation_context[key] = {
            "value": value,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
    
    def get_context(self, key: str) -> Optional[Any]:
        """Get unexpired context"""
        if not self.conversation_context or key not in self.conversation_context:
            return None
        
        context_item = self.conversation_context[key]
        expires_at = datetime.fromisoformat(context_item["expires_at"])
        
        if datetime.utcnow() > expires_at:
            # Context expired, remove it
            del self.conversation_context[key]
            return None
        
        return context_item["value"]
    
    def update_sentiment_history(self, sentiment: str, confidence: float):
        """Update sentiment history with new analysis"""
        if self.sentiment_history is None:
            self.sentiment_history = []
        
        self.sentiment_history.append({
            "sentiment": sentiment,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 50 sentiment entries
        if len(self.sentiment_history) > 50:
            self.sentiment_history = self.sentiment_history[-50:]
    
    def get_recent_topics(self, hours: int = 24) -> List[str]:
        """Get topics discussed in recent hours"""
        if not self.topics_discussed:
            return []
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_topics = []
        
        for topic_data in self.topics_discussed:
            if isinstance(topic_data, dict) and "timestamp" in topic_data:
                topic_time = datetime.fromisoformat(topic_data["timestamp"])
                if topic_time > cutoff:
                    recent_topics.append(topic_data["topic"])
            elif isinstance(topic_data, str):
                # Legacy format - include all
                recent_topics.append(topic_data)
        
        return recent_topics


class ConversationMessage(BaseModel):
    """
    Individual messages within conversations
    """
    __tablename__ = "conversation_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("telegram_conversations.id"), nullable=False)
    
    # Message Identity
    message_id = Column(BigInteger, nullable=False)  # Telegram message ID
    from_user_id = Column(BigInteger, nullable=False)
    reply_to_message_id = Column(BigInteger, nullable=True)
    
    # Message Content
    message_type = Column(SQLEnum(MessageType), default=MessageType.TEXT)
    direction = Column(SQLEnum(MessageDirection), nullable=False)
    content = Column(Text, nullable=True)
    raw_content = Column(JSON, nullable=True)  # Full message data from Telegram
    
    # Message Analysis
    word_count = Column(Integer, default=0)
    character_count = Column(Integer, default=0)
    language = Column(String(10), nullable=True)
    sentiment = Column(String(20), nullable=True)
    sentiment_confidence = Column(Float, nullable=True)
    
    # AI Processing
    topics_extracted = Column(JSON, default=list)
    entities_extracted = Column(JSON, default=list)
    intent_detected = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Response Generation (for outgoing messages)
    generated_by_ai = Column(Boolean, default=False)
    prompt_used = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=True)
    generation_time = Column(Float, nullable=True)  # Time taken to generate
    
    # Timing and Behavior
    sent_at = Column(DateTime, nullable=False)
    typing_duration = Column(Integer, nullable=True)  # Simulated typing time
    response_time = Column(Integer, nullable=True)  # Time to respond to previous message
    
    # Engagement Metrics
    reactions_received = Column(JSON, default=list)
    replies_count = Column(Integer, default=0)
    forwards_count = Column(Integer, default=0)
    
    # Safety and Moderation
    flagged_content = Column(Boolean, default=False)
    moderation_score = Column(Float, nullable=True)
    contains_links = Column(Boolean, default=False)
    contains_mentions = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("TelegramConversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('ix_messages_conversation', 'conversation_id'),
        Index('ix_messages_telegram_id', 'message_id'),
        Index('ix_messages_direction', 'direction'),
        Index('ix_messages_sent_at', 'sent_at'),
        Index('ix_messages_from_user', 'from_user_id'),
    )
    
    def __repr__(self):
        return f"<ConversationMessage(id={self.message_id}, direction={self.direction})>"
    
    @property
    def is_our_message(self) -> bool:
        """Check if this is our message"""
        return self.direction == MessageDirection.OUTGOING
    
    @property
    def is_recent(self) -> bool:
        """Check if message is recent (last hour)"""
        return datetime.utcnow() - self.sent_at < timedelta(hours=1)
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract useful metadata from message"""
        metadata = {
            "word_count": self.word_count,
            "character_count": self.character_count,
            "contains_links": self.contains_links,
            "contains_mentions": self.contains_mentions,
            "sentiment": self.sentiment,
            "topics": self.topics_extracted,
            "entities": self.entities_extracted
        }
        
        if self.is_our_message:
            metadata.update({
                "generated_by_ai": self.generated_by_ai,
                "model_used": self.model_used,
                "generation_time": self.generation_time,
                "typing_duration": self.typing_duration
            })
        
        return metadata


class ConversationContext(BaseModel):
    """
    Extended context storage for conversations
    """
    __tablename__ = "conversation_contexts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("telegram_conversations.id"), nullable=False)
    
    # Context Details
    context_type = Column(String(50), nullable=False)  # topic, relationship, preference, etc.
    key = Column(String(100), nullable=False)
    value = Column(JSON, nullable=False)
    
    # Context Metadata
    confidence = Column(Float, default=1.0)
    importance = Column(Float, default=0.5)  # 0-1 importance for memory retention
    source = Column(String(50), nullable=True)  # How context was acquired
    
    # Expiration and Updates
    expires_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conversation = relationship("TelegramConversation", backref="context_items")
    
    # Indexes
    __table_args__ = (
        Index('ix_context_conversation', 'conversation_id'),
        Index('ix_context_type', 'context_type'),
        Index('ix_context_key', 'key'),
        Index('ix_context_expires', 'expires_at'),
        Index('ix_context_importance', 'importance'),
    )
    
    def __repr__(self):
        return f"<ConversationContext(type={self.context_type}, key={self.key})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if context has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def access(self):
        """Record access to this context"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def extend_expiration(self, hours: int):
        """Extend expiration by specified hours"""
        if self.expires_at is None:
            self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        else:
            self.expires_at += timedelta(hours=hours)