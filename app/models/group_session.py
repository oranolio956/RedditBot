"""
Group Session Models

Defines models for managing group chat sessions, member tracking,
and group-wide conversation management with advanced analytics.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import json
import uuid

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, Text, JSON, ForeignKey, 
    Index, CheckConstraint, Float, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class GroupType(str, Enum):
    """Group type enumeration."""
    PRIVATE_GROUP = "private_group"
    PUBLIC_GROUP = "public_group"
    SUPERGROUP = "supergroup"
    CHANNEL = "channel"


class MemberRole(str, Enum):
    """Member role enumeration."""
    MEMBER = "member"
    ADMIN = "admin"
    CREATOR = "creator"
    RESTRICTED = "restricted"
    LEFT = "left"
    BANNED = "banned"


class GroupStatus(str, Enum):
    """Group status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    RESTRICTED = "restricted"
    ARCHIVED = "archived"


class MessageFrequency(str, Enum):
    """Message frequency levels for rate limiting."""
    LOW = "low"          # < 5 messages/hour
    MODERATE = "moderate" # 5-20 messages/hour
    HIGH = "high"        # 20-50 messages/hour
    VERY_HIGH = "very_high" # > 50 messages/hour


class GroupSession(FullAuditModel):
    """
    Group session model for managing group chat interactions.
    
    Tracks group metadata, settings, and overall conversation flow
    across multiple members and conversation threads.
    """
    
    __tablename__ = "group_sessions"
    
    # Telegram group information
    telegram_chat_id = Column(
        BigInteger,
        unique=True,
        nullable=False,
        index=True,
        comment="Telegram chat ID for the group"
    )
    
    group_type = Column(
        SQLEnum(GroupType, name="group_type"),
        nullable=False,
        comment="Type of Telegram group"
    )
    
    title = Column(
        String(255),
        nullable=False,
        comment="Group title/name"
    )
    
    username = Column(
        String(32),
        nullable=True,
        index=True,
        comment="Group username (if public)"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Group description"
    )
    
    # Group status and settings
    status = Column(
        SQLEnum(GroupStatus, name="group_status"),
        default=GroupStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Current group status"
    )
    
    is_bot_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether the bot has admin privileges"
    )
    
    bot_permissions = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Bot permissions in the group"
    )
    
    # Group activity tracking
    member_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of group members"
    )
    
    active_member_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of recently active members"
    )
    
    total_messages = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Total messages processed in group"
    )
    
    bot_mentions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total bot mentions/interactions"
    )
    
    # Conversation and engagement metrics
    conversation_topics = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Active conversation topics (ML-determined)"
    )
    
    engagement_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Overall group engagement score (0-1)"
    )
    
    sentiment_summary = Column(
        JSONB,
        nullable=True,
        comment="Group sentiment analysis summary"
    )
    
    message_frequency = Column(
        SQLEnum(MessageFrequency, name="message_frequency"),
        default=MessageFrequency.LOW,
        nullable=False,
        comment="Group message frequency level"
    )
    
    # Group settings and preferences
    group_settings = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Group-specific bot settings and preferences"
    )
    
    moderation_settings = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Moderation and anti-spam settings"
    )
    
    language_preferences = Column(
        ARRAY(String),
        nullable=True,
        comment="Preferred languages in the group"
    )
    
    timezone = Column(
        String(50),
        nullable=True,
        comment="Group timezone (if set)"
    )
    
    # Rate limiting and anti-spam
    rate_limit_config = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Rate limiting configuration for this group"
    )
    
    spam_detection_config = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Spam detection settings"
    )
    
    recent_violations = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Recent rate limit/spam violations"
    )
    
    # Activity timing
    first_interaction = Column(
        "first_interaction",
        nullable=True,
        comment="First bot interaction in group"
    )
    
    last_activity = Column(
        "last_activity",
        nullable=True,
        index=True,
        comment="Last activity timestamp"
    )
    
    last_message_at = Column(
        "last_message_at",
        nullable=True,
        index=True,
        comment="Last message timestamp"
    )
    
    # Analytics and ML data
    peak_activity_hours = Column(
        JSONB,
        nullable=True,
        comment="Peak activity hours analysis"
    )
    
    member_interaction_patterns = Column(
        JSONB,
        nullable=True,
        comment="Member interaction patterns and clusters"
    )
    
    conversation_flow_state = Column(
        String(100),
        nullable=True,
        comment="Current conversation flow state"
    )
    
    # Relationships
    members = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")
    conversations = relationship("GroupConversation", back_populates="group", cascade="all, delete-orphan")
    analytics = relationship("GroupAnalytics", back_populates="group", cascade="all, delete-orphan")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_group_chat_id', 'telegram_chat_id'),
        Index('idx_group_status_activity', 'status', 'last_activity'),
        Index('idx_group_type_members', 'group_type', 'member_count'),
        Index('idx_group_engagement', 'engagement_score'),
        CheckConstraint('member_count >= 0', name='ck_group_member_count_positive'),
        CheckConstraint('active_member_count >= 0', name='ck_group_active_members_positive'),
        CheckConstraint('active_member_count <= member_count', name='ck_group_active_members_valid'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_group_engagement_range'),
        CheckConstraint('total_messages >= 0', name='ck_group_total_messages_positive'),
        CheckConstraint('bot_mentions >= 0', name='ck_group_bot_mentions_positive'),
    )
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def update_message_stats(self, is_bot_mentioned: bool = False) -> None:
        """Update message statistics."""
        self.total_messages += 1
        self.last_message_at = datetime.utcnow()
        
        if is_bot_mentioned:
            self.bot_mentions += 1
        
        self.update_activity()
    
    def get_setting(self, key: str, default=None):
        """Get group setting value."""
        if not self.group_settings:
            return default
        return self.group_settings.get(key, default)
    
    def set_setting(self, key: str, value) -> None:
        """Set group setting value."""
        if not self.group_settings:
            self.group_settings = {}
        
        self.group_settings[key] = value
        # Mark the field as modified for SQLAlchemy
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'group_settings')
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for the group."""
        now = datetime.utcnow()
        config = self.rate_limit_config or {}
        
        # Default rate limits based on message frequency
        frequency_limits = {
            MessageFrequency.LOW: {"messages_per_hour": 10, "mentions_per_hour": 3},
            MessageFrequency.MODERATE: {"messages_per_hour": 30, "mentions_per_hour": 10},
            MessageFrequency.HIGH: {"messages_per_hour": 60, "mentions_per_hour": 20},
            MessageFrequency.VERY_HIGH: {"messages_per_hour": 120, "mentions_per_hour": 40}
        }
        
        limits = frequency_limits.get(self.message_frequency, frequency_limits[MessageFrequency.MODERATE])
        limits.update(config.get("custom_limits", {}))
        
        return {
            "limits": limits,
            "current_frequency": self.message_frequency,
            "recent_violations": len(self.recent_violations or []),
            "is_restricted": self.status == GroupStatus.RESTRICTED
        }
    
    def add_rate_limit_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Add a rate limit violation record."""
        if not self.recent_violations:
            self.recent_violations = []
        
        violation = {
            "type": violation_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        self.recent_violations.append(violation)
        
        # Keep only recent violations (last 100)
        if len(self.recent_violations) > 100:
            self.recent_violations = self.recent_violations[-50:]
        
        # Mark the field as modified for SQLAlchemy
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'recent_violations')
    
    def calculate_engagement_score(self) -> float:
        """Calculate and update engagement score based on activity metrics."""
        try:
            if self.member_count == 0:
                return 0.0
            
            # Base factors for engagement calculation
            factors = {
                "active_member_ratio": self.active_member_count / self.member_count if self.member_count > 0 else 0,
                "message_per_member_ratio": min(self.total_messages / max(self.member_count, 1) / 100, 1.0),
                "bot_interaction_ratio": min(self.bot_mentions / max(self.total_messages, 1) * 10, 1.0),
                "recent_activity": 1.0 if self.last_activity and (datetime.utcnow() - self.last_activity).days < 1 else 0.5
            }
            
            # Weighted calculation
            weights = {"active_member_ratio": 0.3, "message_per_member_ratio": 0.25, 
                      "bot_interaction_ratio": 0.25, "recent_activity": 0.2}
            
            score = sum(factors[key] * weights[key] for key in factors)
            
            # Apply frequency bonus/penalty
            frequency_modifiers = {
                MessageFrequency.LOW: 0.9,
                MessageFrequency.MODERATE: 1.0,
                MessageFrequency.HIGH: 1.1,
                MessageFrequency.VERY_HIGH: 0.95  # Too high might indicate spam
            }
            
            score *= frequency_modifiers.get(self.message_frequency, 1.0)
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))
            
            self.engagement_score = score
            return score
            
        except Exception:
            return 0.0
    
    def get_group_summary(self) -> Dict[str, Any]:
        """Get comprehensive group summary."""
        return {
            "group_id": str(self.id),
            "telegram_chat_id": self.telegram_chat_id,
            "title": self.title,
            "type": self.group_type,
            "status": self.status,
            "member_count": self.member_count,
            "active_members": self.active_member_count,
            "total_messages": self.total_messages,
            "bot_mentions": self.bot_mentions,
            "engagement_score": self.engagement_score,
            "message_frequency": self.message_frequency,
            "is_bot_admin": self.is_bot_admin,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class GroupMember(FullAuditModel):
    """
    Group member model for tracking individual user participation in groups.
    
    Maintains member-specific data, roles, and interaction patterns
    within the context of a specific group.
    """
    
    __tablename__ = "group_members"
    
    # Relationships
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("group_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Group this member belongs to"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who is a member of this group"
    )
    
    telegram_user_id = Column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram user ID for quick lookups"
    )
    
    # Member role and status
    role = Column(
        SQLEnum(MemberRole, name="member_role"),
        default=MemberRole.MEMBER,
        nullable=False,
        index=True,
        comment="Member role in the group"
    )
    
    status = Column(
        String(20),
        default="active",
        nullable=False,
        index=True,
        comment="Member status (active, restricted, etc.)"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether member is currently active"
    )
    
    # Member activity metrics
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total messages from this member in the group"
    )
    
    mention_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Times this member mentioned the bot"
    )
    
    last_message_at = Column(
        "last_message_at",
        nullable=True,
        index=True,
        comment="Last message from this member"
    )
    
    last_seen_at = Column(
        "last_seen_at",
        nullable=True,
        index=True,
        comment="Last activity from this member"
    )
    
    # Member interaction patterns
    interaction_frequency = Column(
        SQLEnum(MessageFrequency, name="member_frequency"),
        default=MessageFrequency.LOW,
        nullable=False,
        comment="Member's interaction frequency"
    )
    
    engagement_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Member engagement score in this group (0-1)"
    )
    
    influence_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Member influence/popularity score (0-1)"
    )
    
    # Member preferences and context
    member_context = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Member-specific context and conversation state"
    )
    
    notification_preferences = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Member notification preferences"
    )
    
    # Behavioral analysis
    conversation_topics = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Topics this member frequently discusses"
    )
    
    interaction_patterns = Column(
        JSONB,
        nullable=True,
        comment="Behavioral patterns and preferences"
    )
    
    sentiment_profile = Column(
        JSONB,
        nullable=True,
        comment="Member's sentiment patterns in this group"
    )
    
    # Risk and moderation
    risk_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Risk assessment score (0-1)"
    )
    
    violation_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of policy violations"
    )
    
    last_violation_at = Column(
        "last_violation_at",
        nullable=True,
        comment="Last policy violation timestamp"
    )
    
    moderation_notes = Column(
        Text,
        nullable=True,
        comment="Moderation notes and history"
    )
    
    # Member permissions and restrictions
    permissions = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Member-specific permissions"
    )
    
    restrictions = Column(
        JSONB,
        nullable=True,
        comment="Member-specific restrictions"
    )
    
    # Timing
    joined_at = Column(
        "joined_at",
        nullable=True,
        index=True,
        comment="When member joined the group"
    )
    
    left_at = Column(
        "left_at",
        nullable=True,
        comment="When member left the group"
    )
    
    # Relationships
    group = relationship("GroupSession", back_populates="members")
    user = relationship("User", back_populates="group_memberships")
    
    # Database constraints and indexes
    __table_args__ = (
        UniqueConstraint('group_id', 'user_id', name='uq_group_member'),
        Index('idx_member_group_role', 'group_id', 'role'),
        Index('idx_member_activity', 'is_active', 'last_seen_at'),
        Index('idx_member_engagement', 'engagement_score'),
        Index('idx_member_risk', 'risk_score'),
        Index('idx_member_telegram_user', 'telegram_user_id'),
        CheckConstraint('message_count >= 0', name='ck_member_message_count_positive'),
        CheckConstraint('mention_count >= 0', name='ck_member_mention_count_positive'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_member_engagement_range'),
        CheckConstraint('influence_score >= 0 AND influence_score <= 1', name='ck_member_influence_range'),
        CheckConstraint('risk_score >= 0 AND risk_score <= 1', name='ck_member_risk_range'),
        CheckConstraint('violation_count >= 0', name='ck_member_violations_positive'),
    )
    
    def update_activity(self) -> None:
        """Update member activity timestamp."""
        self.last_seen_at = datetime.utcnow()
        self.is_active = True
    
    def add_message(self, is_bot_mention: bool = False) -> None:
        """Record a new message from this member."""
        self.message_count += 1
        self.last_message_at = datetime.utcnow()
        
        if is_bot_mention:
            self.mention_count += 1
        
        self.update_activity()
    
    def calculate_engagement_score(self, group_total_messages: int, group_member_count: int) -> float:
        """Calculate member engagement score within the group."""
        try:
            if group_total_messages == 0:
                return 0.0
            
            # Calculate various engagement factors
            factors = {
                "message_participation": min(self.message_count / max(group_total_messages / group_member_count, 1), 2.0),
                "bot_interaction": min(self.mention_count / max(self.message_count, 1) * 5, 1.0),
                "recent_activity": 1.0 if self.last_seen_at and (datetime.utcnow() - self.last_seen_at).days < 7 else 0.3,
                "consistency": min(self.message_count / max((datetime.utcnow() - self.joined_at).days, 1), 1.0) if self.joined_at else 0
            }
            
            # Weighted calculation
            weights = {"message_participation": 0.4, "bot_interaction": 0.2, 
                      "recent_activity": 0.25, "consistency": 0.15}
            
            score = sum(factors[key] * weights[key] for key in factors)
            
            # Role bonus
            role_bonuses = {
                MemberRole.CREATOR: 0.1,
                MemberRole.ADMIN: 0.05,
                MemberRole.MEMBER: 0.0,
                MemberRole.RESTRICTED: -0.2,
                MemberRole.LEFT: -0.5,
                MemberRole.BANNED: -1.0
            }
            
            score += role_bonuses.get(self.role, 0.0)
            
            # Risk penalty
            score *= (1.0 - self.risk_score * 0.5)
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))
            
            self.engagement_score = score
            return score
            
        except Exception:
            return 0.0
    
    def get_member_summary(self) -> Dict[str, Any]:
        """Get member summary for analytics."""
        active_days = (datetime.utcnow() - self.joined_at).days if self.joined_at else 0
        
        return {
            "member_id": str(self.id),
            "user_id": str(self.user_id),
            "telegram_user_id": self.telegram_user_id,
            "role": self.role,
            "status": self.status,
            "is_active": self.is_active,
            "message_count": self.message_count,
            "mention_count": self.mention_count,
            "engagement_score": self.engagement_score,
            "influence_score": self.influence_score,
            "risk_score": self.risk_score,
            "violation_count": self.violation_count,
            "active_days": active_days,
            "last_seen": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None
        }


class GroupConversation(FullAuditModel):
    """
    Group conversation model for tracking thematic conversation threads within groups.
    
    Represents focused discussion topics that span multiple messages and participants
    within a group chat environment.
    """
    
    __tablename__ = "group_conversations"
    
    # Relationships
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("group_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Group this conversation belongs to"
    )
    
    # Conversation metadata
    thread_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Thread identifier (can be message ID or topic hash)"
    )
    
    topic = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Conversation topic (ML-determined or user-defined)"
    )
    
    title = Column(
        String(255),
        nullable=True,
        comment="Conversation title"
    )
    
    status = Column(
        String(20),
        default="active",
        nullable=False,
        index=True,
        comment="Conversation status"
    )
    
    # Participation tracking
    participant_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of unique participants"
    )
    
    participant_ids = Column(
        ARRAY(BigInteger),
        nullable=True,
        comment="List of participant Telegram user IDs"
    )
    
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total messages in this conversation thread"
    )
    
    bot_interactions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Bot interactions/mentions in this thread"
    )
    
    # Conversation analytics
    engagement_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Conversation engagement score (0-1)"
    )
    
    sentiment_summary = Column(
        JSONB,
        nullable=True,
        comment="Sentiment analysis summary for the conversation"
    )
    
    complexity_score = Column(
        Float,
        nullable=True,
        comment="Conversation complexity score (0-1)"
    )
    
    toxicity_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Toxicity/conflict score (0-1)"
    )
    
    # Content analysis
    keywords = Column(
        ARRAY(String),
        nullable=True,
        comment="Key terms and topics discussed"
    )
    
    entities = Column(
        JSONB,
        nullable=True,
        comment="Named entities mentioned in conversation"
    )
    
    language_distribution = Column(
        JSONB,
        nullable=True,
        comment="Languages used in conversation"
    )
    
    # Timing and activity
    started_at = Column(
        "started_at",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Conversation start timestamp"
    )
    
    last_message_at = Column(
        "last_message_at",
        nullable=True,
        index=True,
        comment="Last message in conversation"
    )
    
    ended_at = Column(
        "ended_at",
        nullable=True,
        comment="Conversation end timestamp"
    )
    
    duration_seconds = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total conversation duration"
    )
    
    # Context and state
    conversation_context = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Conversation context and state data"
    )
    
    moderator_actions = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Moderation actions taken in this conversation"
    )
    
    # Relationships
    group = relationship("GroupSession", back_populates="conversations")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_group_conv_thread', 'group_id', 'thread_id'),
        Index('idx_group_conv_topic', 'group_id', 'topic'),
        Index('idx_group_conv_timing', 'started_at', 'last_message_at'),
        Index('idx_group_conv_engagement', 'engagement_score'),
        Index('idx_group_conv_toxicity', 'toxicity_score'),
        CheckConstraint('participant_count >= 0', name='ck_group_conv_participants_positive'),
        CheckConstraint('message_count >= 0', name='ck_group_conv_messages_positive'),
        CheckConstraint('bot_interactions >= 0', name='ck_group_conv_bot_interactions_positive'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_group_conv_engagement_range'),
        CheckConstraint('complexity_score >= 0 AND complexity_score <= 1', name='ck_group_conv_complexity_range'),
        CheckConstraint('toxicity_score >= 0 AND toxicity_score <= 1', name='ck_group_conv_toxicity_range'),
        CheckConstraint('duration_seconds >= 0', name='ck_group_conv_duration_positive'),
    )
    
    def add_message(self, user_id: int, is_bot_interaction: bool = False) -> None:
        """Add a message to this conversation thread."""
        self.message_count += 1
        self.last_message_at = datetime.utcnow()
        
        if is_bot_interaction:
            self.bot_interactions += 1
        
        # Add participant if new
        if not self.participant_ids:
            self.participant_ids = []
        
        if user_id not in self.participant_ids:
            self.participant_ids.append(user_id)
            self.participant_count = len(self.participant_ids)
            
            # Mark the field as modified for SQLAlchemy
            from sqlalchemy.orm import attributes
            attributes.flag_modified(self, 'participant_ids')
    
    def end_conversation(self) -> None:
        """End the conversation and calculate final metrics."""
        now = datetime.utcnow()
        self.ended_at = now
        self.status = "ended"
        
        if self.started_at:
            self.duration_seconds = int((now - self.started_at).total_seconds())
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary for analytics."""
        return {
            "conversation_id": str(self.id),
            "group_id": str(self.group_id),
            "thread_id": self.thread_id,
            "topic": self.topic,
            "participant_count": self.participant_count,
            "message_count": self.message_count,
            "bot_interactions": self.bot_interactions,
            "engagement_score": self.engagement_score,
            "complexity_score": self.complexity_score,
            "toxicity_score": self.toxicity_score,
            "duration_minutes": self.duration_seconds // 60 if self.duration_seconds else 0,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None
        }


class GroupAnalytics(FullAuditModel):
    """
    Group analytics model for storing aggregated metrics and insights.
    
    Captures time-series data and analytical insights about group behavior,
    engagement patterns, and performance metrics.
    """
    
    __tablename__ = "group_analytics"
    
    # Relationships
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("group_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Group these analytics belong to"
    )
    
    # Time period and scope
    period_type = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Analytics period (hourly, daily, weekly, monthly)"
    )
    
    period_start = Column(
        "period_start",
        nullable=False,
        index=True,
        comment="Analytics period start timestamp"
    )
    
    period_end = Column(
        "period_end",
        nullable=False,
        index=True,
        comment="Analytics period end timestamp"
    )
    
    # Message and activity metrics
    total_messages = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total messages in period"
    )
    
    unique_participants = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Unique participants in period"
    )
    
    bot_interactions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Bot interactions in period"
    )
    
    conversation_threads = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Conversation threads started in period"
    )
    
    # Engagement metrics
    average_engagement = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Average engagement score for period"
    )
    
    peak_activity_hour = Column(
        Integer,
        nullable=True,
        comment="Peak activity hour (0-23)"
    )
    
    activity_distribution = Column(
        JSONB,
        nullable=True,
        comment="Hourly activity distribution"
    )
    
    # Content and sentiment analysis
    sentiment_distribution = Column(
        JSONB,
        nullable=True,
        comment="Sentiment distribution (positive, negative, neutral)"
    )
    
    topic_distribution = Column(
        JSONB,
        nullable=True,
        comment="Topic distribution and frequencies"
    )
    
    language_usage = Column(
        JSONB,
        nullable=True,
        comment="Language usage statistics"
    )
    
    # Member analytics
    new_members = Column(
        Integer,
        default=0,
        nullable=False,
        comment="New members added in period"
    )
    
    members_left = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Members who left in period"
    )
    
    active_member_ratio = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Ratio of active to total members"
    )
    
    # Performance and moderation metrics
    response_time_avg = Column(
        Float,
        nullable=True,
        comment="Average bot response time in seconds"
    )
    
    violations_detected = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Policy violations detected"
    )
    
    moderation_actions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Moderation actions taken"
    )
    
    # Growth and trend metrics
    growth_rate = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Message/engagement growth rate vs previous period"
    )
    
    trend_direction = Column(
        String(20),
        nullable=True,
        comment="Trend direction (growing, stable, declining)"
    )
    
    # Additional insights
    insights = Column(
        JSONB,
        nullable=True,
        comment="ML-generated insights and recommendations"
    )
    
    anomalies = Column(
        JSONB,
        nullable=True,
        comment="Detected anomalies and unusual patterns"
    )
    
    # Relationships
    group = relationship("GroupSession", back_populates="analytics")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_analytics_group_period', 'group_id', 'period_type', 'period_start'),
        Index('idx_analytics_engagement', 'average_engagement'),
        Index('idx_analytics_growth', 'growth_rate'),
        CheckConstraint('total_messages >= 0', name='ck_analytics_total_messages_positive'),
        CheckConstraint('unique_participants >= 0', name='ck_analytics_participants_positive'),
        CheckConstraint('bot_interactions >= 0', name='ck_analytics_bot_interactions_positive'),
        CheckConstraint('conversation_threads >= 0', name='ck_analytics_threads_positive'),
        CheckConstraint('new_members >= 0', name='ck_analytics_new_members_positive'),
        CheckConstraint('members_left >= 0', name='ck_analytics_left_members_positive'),
        CheckConstraint('violations_detected >= 0', name='ck_analytics_violations_positive'),
        CheckConstraint('moderation_actions >= 0', name='ck_analytics_mod_actions_positive'),
        CheckConstraint('average_engagement >= 0 AND average_engagement <= 1', name='ck_analytics_engagement_range'),
        CheckConstraint('active_member_ratio >= 0 AND active_member_ratio <= 1', name='ck_analytics_ratio_range'),
        CheckConstraint('peak_activity_hour >= 0 AND peak_activity_hour <= 23', name='ck_analytics_hour_range'),
    )
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        return {
            "analytics_id": str(self.id),
            "group_id": str(self.group_id),
            "period": {
                "type": self.period_type,
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "activity": {
                "total_messages": self.total_messages,
                "unique_participants": self.unique_participants,
                "bot_interactions": self.bot_interactions,
                "conversation_threads": self.conversation_threads
            },
            "engagement": {
                "average_score": self.average_engagement,
                "peak_hour": self.peak_activity_hour,
                "active_member_ratio": self.active_member_ratio
            },
            "growth": {
                "rate": self.growth_rate,
                "direction": self.trend_direction,
                "new_members": self.new_members,
                "members_left": self.members_left
            },
            "moderation": {
                "violations_detected": self.violations_detected,
                "actions_taken": self.moderation_actions,
                "avg_response_time": self.response_time_avg
            }
        }