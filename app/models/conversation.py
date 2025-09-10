"""
Conversation Models

Defines models for storing conversation data, messages, and session management.
Supports advanced conversation tracking with ML features and analytics.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, Text, JSON, ForeignKey, 
    Index, CheckConstraint, Float, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class ConversationStatus(str, Enum):
    """Conversation status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


class MessageType(str, Enum):
    """Message type enumeration."""
    TEXT = "text"
    COMMAND = "command"
    CALLBACK = "callback"
    INLINE = "inline"
    STICKER = "sticker"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"
    POLL = "poll"
    VENUE = "venue"
    ANIMATION = "animation"
    VIDEO_NOTE = "video_note"
    GAME = "game"
    INVOICE = "invoice"
    SUCCESSFUL_PAYMENT = "successful_payment"


class MessageDirection(str, Enum):
    """Message direction enumeration."""
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class ConversationSession(FullAuditModel):
    """
    Conversation session model for tracking individual conversation sessions.
    
    A session represents a continuous interaction period between user and bot,
    typically bounded by periods of inactivity.
    """
    
    __tablename__ = "conversation_sessions"
    
    # Basic session information
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this session"
    )
    
    session_token = Column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique session identifier"
    )
    
    status = Column(
        SQLEnum(ConversationStatus, name="session_status"),
        default=ConversationStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Current session status"
    )
    
    # Session timing
    started_at = Column(
        "started_at",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Session start timestamp"
    )
    
    last_activity_at = Column(
        "last_activity_at",
        nullable=True,
        index=True,
        comment="Last activity in this session"
    )
    
    ended_at = Column(
        "ended_at",
        nullable=True,
        index=True,
        comment="Session end timestamp"
    )
    
    # Session metrics
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total messages in this session"
    )
    
    duration_seconds = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Session duration in seconds"
    )
    
    # Context and state
    context_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Session context and state data"
    )
    
    conversation_topic = Column(
        String(255),
        nullable=True,
        comment="Main topic of conversation (ML-determined)"
    )
    
    # ML and analytics data
    sentiment_scores = Column(
        JSONB,
        nullable=True,
        comment="Sentiment analysis scores throughout session"
    )
    
    personality_adaptations = Column(
        JSONB,
        nullable=True,
        comment="Personality adaptations made during session"
    )
    
    engagement_metrics = Column(
        JSONB,
        nullable=True,
        comment="User engagement metrics for this session"
    )
    
    # Relationships
    user = relationship("User", back_populates="conversation_sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_session_user_status', 'user_id', 'status'),
        Index('idx_session_activity', 'last_activity_at'),
        Index('idx_session_duration', 'started_at', 'ended_at'),
        CheckConstraint('duration_seconds >= 0', name='ck_session_duration_positive'),
        CheckConstraint('message_count >= 0', name='ck_session_message_count_positive'),
    )
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
    
    def end_session(self) -> None:
        """End the session and calculate final metrics."""
        now = datetime.utcnow()
        self.ended_at = now
        self.status = ConversationStatus.ENDED
        
        if self.started_at:
            self.duration_seconds = int((now - self.started_at).total_seconds())
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this session."""
        return {
            "session_id": str(self.id),
            "duration_minutes": self.duration_seconds // 60 if self.duration_seconds else 0,
            "message_count": self.message_count,
            "topic": self.conversation_topic,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class Conversation(FullAuditModel):
    """
    Conversation model for tracking individual conversation threads.
    
    Represents a thematic conversation unit within a session,
    focused on a specific topic or interaction flow.
    """
    
    __tablename__ = "conversations"
    
    # Relationship to session and user
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session this conversation belongs to"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User participating in conversation"
    )
    
    # Conversation metadata
    title = Column(
        String(255),
        nullable=True,
        comment="Conversation title (auto-generated or user-defined)"
    )
    
    topic = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Main conversation topic"
    )
    
    status = Column(
        SQLEnum(ConversationStatus, name="conversation_status"),
        default=ConversationStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Current conversation status"
    )
    
    # Conversation flow and context
    context_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Conversation context and variables"
    )
    
    conversation_flow = Column(
        String(50),
        nullable=True,
        comment="Current conversation flow/state"
    )
    
    # ML and analytics
    sentiment_summary = Column(
        JSONB,
        nullable=True,
        comment="Overall sentiment analysis for conversation"
    )
    
    complexity_score = Column(
        Float,
        nullable=True,
        comment="Conversation complexity score (0-1)"
    )
    
    engagement_score = Column(
        Float,
        nullable=True,
        comment="User engagement score (0-1)"
    )
    
    # Timing and metrics
    started_at = Column(
        "started_at",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Conversation start time"
    )
    
    last_message_at = Column(
        "last_message_at",
        nullable=True,
        index=True,
        comment="Timestamp of last message"
    )
    
    ended_at = Column(
        "ended_at",
        nullable=True,
        comment="Conversation end time"
    )
    
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total messages in conversation"
    )
    
    # Relationships
    session = relationship("ConversationSession", back_populates="conversations")
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    risk_assessments = relationship("ConversationRisk", back_populates="conversation")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_conversation_user_topic', 'user_id', 'topic'),
        Index('idx_conversation_session_status', 'session_id', 'status'),
        Index('idx_conversation_timing', 'started_at', 'last_message_at'),
        CheckConstraint('message_count >= 0', name='ck_conversation_message_count_positive'),
        CheckConstraint('complexity_score >= 0 AND complexity_score <= 1', name='ck_complexity_range'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_engagement_range'),
    )
    
    def update_last_message(self) -> None:
        """Update last message timestamp."""
        self.last_message_at = datetime.utcnow()
        self.message_count += 1
    
    def end_conversation(self) -> None:
        """End the conversation."""
        self.ended_at = datetime.utcnow()
        self.status = ConversationStatus.ENDED
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary for analytics."""
        duration = None
        if self.started_at and self.ended_at:
            duration = int((self.ended_at - self.started_at).total_seconds())
        elif self.started_at:
            duration = int((datetime.utcnow() - self.started_at).total_seconds())
        
        return {
            "conversation_id": str(self.id),
            "topic": self.topic,
            "message_count": self.message_count,
            "duration_seconds": duration,
            "complexity_score": self.complexity_score,
            "engagement_score": self.engagement_score,
            "status": self.status,
        }


class Message(FullAuditModel):
    """
    Message model for storing individual messages in conversations.
    
    Captures all message details including content, metadata, and ML analysis.
    """
    
    __tablename__ = "messages"
    
    # Relationships
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Conversation this message belongs to"
    )
    
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session this message belongs to"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who sent/received this message"
    )
    
    # Telegram-specific fields
    telegram_message_id = Column(
        BigInteger,
        nullable=True,
        index=True,
        comment="Telegram message ID"
    )
    
    telegram_chat_id = Column(
        BigInteger,
        nullable=True,
        index=True,
        comment="Telegram chat ID"
    )
    
    # Message content and metadata
    message_type = Column(
        SQLEnum(MessageType, name="message_type"),
        nullable=False,
        index=True,
        comment="Type of message content"
    )
    
    direction = Column(
        SQLEnum(MessageDirection, name="message_direction"),
        nullable=False,
        index=True,
        comment="Message direction (incoming/outgoing)"
    )
    
    content = Column(
        Text,
        nullable=True,
        comment="Message text content"
    )
    
    content_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="Hash of content for deduplication"
    )
    
    # Message metadata
    message_metadata = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional message metadata (entities, formatting, etc.)"
    )
    
    attachments = Column(
        JSONB,
        nullable=True,
        comment="File attachments and media information"
    )
    
    reply_to_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="Message this is replying to"
    )
    
    # Processing and analysis
    processed_at = Column(
        "processed_at",
        nullable=True,
        comment="When message was processed by ML pipeline"
    )
    
    sentiment_score = Column(
        Float,
        nullable=True,
        comment="Sentiment analysis score (-1 to 1)"
    )
    
    sentiment_label = Column(
        String(20),
        nullable=True,
        comment="Sentiment classification label"
    )
    
    emotion_scores = Column(
        JSONB,
        nullable=True,
        comment="Emotion classification scores"
    )
    
    intent_classification = Column(
        String(50),
        nullable=True,
        comment="Classified user intent"
    )
    
    entities = Column(
        JSONB,
        nullable=True,
        comment="Named entities extracted from message"
    )
    
    keywords = Column(
        ARRAY(String),
        nullable=True,
        comment="Key terms extracted from message"
    )
    
    # Response generation data
    response_generated = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether a response was generated for this message"
    )
    
    response_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken to generate response in milliseconds"
    )
    
    response_model_used = Column(
        String(100),
        nullable=True,
        comment="AI model used for response generation"
    )
    
    # Quality and feedback
    user_feedback = Column(
        String(20),
        nullable=True,
        comment="User feedback on bot response (positive/negative/neutral)"
    )
    
    quality_score = Column(
        Float,
        nullable=True,
        comment="Internal quality assessment score (0-1)"
    )
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    session = relationship("ConversationSession", back_populates="messages")
    user = relationship("User", back_populates="messages")
    reply_to = relationship("Message", remote_side="Message.id", backref="replies")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_message_conversation_created', 'conversation_id', 'created_at'),
        Index('idx_message_user_type', 'user_id', 'message_type'),
        Index('idx_message_direction_created', 'direction', 'created_at'),
        Index('idx_message_telegram', 'telegram_chat_id', 'telegram_message_id'),
        Index('idx_message_content_hash', 'content_hash'),
        Index('idx_message_sentiment', 'sentiment_label'),
        Index('idx_message_intent', 'intent_classification'),
        CheckConstraint('sentiment_score >= -1 AND sentiment_score <= 1', name='ck_sentiment_range'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='ck_quality_range'),
        CheckConstraint('response_time_ms >= 0', name='ck_response_time_positive'),
    )
    
    def calculate_content_hash(self) -> None:
        """Calculate and set content hash for deduplication."""
        if self.content:
            import hashlib
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
    
    def set_processed(self) -> None:
        """Mark message as processed."""
        self.processed_at = datetime.utcnow()
    
    def get_message_summary(self) -> Dict[str, Any]:
        """Get message summary for analytics."""
        return {
            "message_id": str(self.id),
            "type": self.message_type,
            "direction": self.direction,
            "length": len(self.content) if self.content else 0,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "intent": self.intent_classification,
            "response_time_ms": self.response_time_ms,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def is_duplicate(self, other_message: 'Message') -> bool:
        """Check if this message is a duplicate of another."""
        if not self.content_hash or not other_message.content_hash:
            return False
        return (
            self.content_hash == other_message.content_hash and
            self.user_id == other_message.user_id and
            self.message_type == other_message.message_type
        )


# User model relationships are defined in the User model itself
# to avoid circular import issues