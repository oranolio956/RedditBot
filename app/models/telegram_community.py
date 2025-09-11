"""
Telegram Community Management Models
Advanced models for managing engagement across multiple Telegram communities.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, JSON, Float, Text,
    ForeignKey, Enum as SQLEnum, BigInteger, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database.base import BaseModel


class CommunityType(str, Enum):
    """Type of Telegram community"""
    SUPERGROUP = "supergroup"
    CHANNEL = "channel"
    BASIC_GROUP = "basic_group"
    PRIVATE_CHAT = "private_chat"


class EngagementStrategy(str, Enum):
    """Engagement strategy for different communities"""
    LURKER = "lurker"  # Mostly observe, minimal engagement
    PARTICIPANT = "participant"  # Regular helpful participation
    CONTRIBUTOR = "contributor"  # Active valuable contributions
    LEADER = "leader"  # Thought leadership and guidance


class CommunityStatus(str, Enum):
    """Community relationship status"""
    PENDING_JOIN = "pending_join"
    JOINED = "joined"
    ACTIVE = "active"
    LIMITED = "limited"  # Restricted by community
    LEFT = "left"
    BANNED = "banned"


class TelegramCommunity(BaseModel):
    """
    Community profile and engagement tracking for each Telegram group/channel
    """
    __tablename__ = "telegram_communities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("telegram_accounts.id"), nullable=False)
    
    # Community Identity
    chat_id = Column(BigInteger, nullable=False)
    title = Column(String(255), nullable=False)
    username = Column(String(100), nullable=True)  # Public username if available
    invite_link = Column(Text, nullable=True)
    community_type = Column(SQLEnum(CommunityType), nullable=False)
    
    # Community Metadata
    member_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    topics = Column(JSON, default=list)  # Main discussion topics
    language = Column(String(10), default="en")
    timezone = Column(String(50), nullable=True)
    
    # Engagement Strategy
    engagement_strategy = Column(SQLEnum(EngagementStrategy), default=EngagementStrategy.PARTICIPANT)
    status = Column(SQLEnum(CommunityStatus), default=CommunityStatus.PENDING_JOIN)
    join_date = Column(DateTime, nullable=True)
    last_activity_date = Column(DateTime, nullable=True)
    
    # Community Analysis
    activity_level = Column(String(20), default="unknown")  # low, medium, high, very_high
    moderation_strictness = Column(String(20), default="unknown")  # lenient, moderate, strict
    spam_tolerance = Column(String(20), default="unknown")  # high, medium, low
    
    # Personality Adaptation
    community_personality = Column(JSON, nullable=True)  # Adapted personality for this community
    communication_style = Column(String(50), default="adaptive")
    formality_level = Column(String(20), default="casual")  # formal, semi-formal, casual
    
    # Engagement Metrics
    messages_sent = Column(Integer, default=0)
    messages_received = Column(Integer, default=0)  # Responses to our messages
    positive_reactions = Column(Integer, default=0)
    negative_reactions = Column(Integer, default=0)
    mentions_received = Column(Integer, default=0)
    
    # Performance Tracking
    engagement_score = Column(Float, default=0.0)  # 0-100 community engagement score
    influence_score = Column(Float, default=0.0)  # 0-100 influence in community
    reputation_score = Column(Float, default=50.0)  # 0-100 reputation score
    trust_level = Column(Float, default=0.0)  # 0-100 trust from community
    
    # Safety & Compliance
    warning_count = Column(Integer, default=0)
    last_warning_date = Column(DateTime, nullable=True)
    restriction_level = Column(String(20), default="none")  # none, limited, restricted
    compliance_notes = Column(Text, nullable=True)
    
    # Activity Patterns
    peak_activity_hours = Column(JSON, default=list)  # Hours of highest activity
    typical_response_time = Column(Integer, nullable=True)  # Average response time in minutes
    last_message_date = Column(DateTime, nullable=True)
    message_frequency = Column(Float, default=0.0)  # Messages per day average
    
    # Content Strategy
    preferred_content_types = Column(JSON, default=list)  # text, images, links, etc.
    content_themes = Column(JSON, default=list)  # Topics that work well
    avoided_topics = Column(JSON, default=list)  # Topics to avoid
    
    # Relationship Building
    key_members = Column(JSON, default=list)  # Important community members
    allies = Column(JSON, default=list)  # Members who respond positively
    adversaries = Column(JSON, default=list)  # Members who respond negatively
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TelegramAccount", back_populates="communities")
    conversations = relationship("TelegramConversation", back_populates="community", cascade="all, delete-orphan")
    engagement_events = relationship("CommunityEngagementEvent", back_populates="community", cascade="all, delete-orphan")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('account_id', 'chat_id', name='unique_account_community'),
        Index('ix_communities_account', 'account_id'),
        Index('ix_communities_chat_id', 'chat_id'),
        Index('ix_communities_status', 'status'),
        Index('ix_communities_strategy', 'engagement_strategy'),
        Index('ix_communities_activity', 'last_activity_date'),
    )
    
    def __repr__(self):
        return f"<TelegramCommunity(title={self.title}, strategy={self.engagement_strategy})>"
    
    @property
    def is_active(self) -> bool:
        """Check if community engagement is active"""
        return (
            self.status == CommunityStatus.ACTIVE and
            self.warning_count < 3 and
            self.restriction_level == "none"
        )
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate (responses received / messages sent)"""
        if self.messages_sent == 0:
            return 0.0
        return (self.messages_received / self.messages_sent) * 100
    
    @property
    def reputation_trend(self) -> str:
        """Determine reputation trend"""
        if self.positive_reactions > self.negative_reactions * 2:
            return "improving"
        elif self.negative_reactions > self.positive_reactions * 2:
            return "declining"
        return "stable"
    
    def update_engagement_score(self):
        """Update community engagement score based on metrics"""
        # Base score from message engagement
        message_score = min(25, (self.messages_sent / 10) * 25)
        
        # Response rate score
        response_score = min(25, self.engagement_rate * 0.25)
        
        # Reaction score
        total_reactions = self.positive_reactions + self.negative_reactions
        if total_reactions > 0:
            reaction_ratio = self.positive_reactions / total_reactions
            reaction_score = reaction_ratio * 25
        else:
            reaction_score = 0
        
        # Trust and reputation score
        trust_score = (self.trust_level / 100) * 25
        
        self.engagement_score = message_score + response_score + reaction_score + trust_score
        return self.engagement_score
    
    def adapt_personality_to_community(self, base_personality: dict) -> dict:
        """Adapt base personality to community context"""
        adapted = base_personality.copy()
        
        # Adjust formality based on community
        if self.formality_level == "formal":
            adapted["formality"] = min(1.0, adapted.get("formality", 0.5) + 0.3)
        elif self.formality_level == "casual":
            adapted["formality"] = max(0.0, adapted.get("formality", 0.5) - 0.3)
        
        # Adjust engagement level based on strategy
        strategy_multipliers = {
            EngagementStrategy.LURKER: 0.2,
            EngagementStrategy.PARTICIPANT: 0.6,
            EngagementStrategy.CONTRIBUTOR: 0.8,
            EngagementStrategy.LEADER: 1.0
        }
        
        multiplier = strategy_multipliers.get(self.engagement_strategy, 0.6)
        adapted["engagement_level"] = adapted.get("engagement_level", 0.5) * multiplier
        
        # Store adapted personality
        self.community_personality = adapted
        return adapted


class CommunityEngagementEvent(BaseModel):
    """
    Track engagement events within communities
    """
    __tablename__ = "community_engagement_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id = Column(UUID(as_uuid=True), ForeignKey("telegram_communities.id"), nullable=False)
    
    # Event Details
    event_type = Column(String(50), nullable=False)  # message_sent, reaction_received, mention, etc.
    description = Column(Text, nullable=True)
    
    # Event Data
    message_id = Column(BigInteger, nullable=True)
    user_id = Column(BigInteger, nullable=True)  # Other user involved
    content_preview = Column(String(200), nullable=True)
    
    # Event Metrics
    sentiment = Column(String(20), nullable=True)  # positive, neutral, negative
    engagement_impact = Column(Float, default=0.0)  # Impact on engagement score
    reputation_impact = Column(Float, default=0.0)  # Impact on reputation
    
    # Event Context
    context_data = Column(JSON, nullable=True)
    response_time = Column(Integer, nullable=True)  # Response time in seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    community = relationship("TelegramCommunity", back_populates="engagement_events")
    
    # Indexes
    __table_args__ = (
        Index('ix_engagement_events_community', 'community_id'),
        Index('ix_engagement_events_type', 'event_type'),
        Index('ix_engagement_events_created', 'created_at'),
        Index('ix_engagement_events_sentiment', 'sentiment'),
    )
    
    def __repr__(self):
        return f"<EngagementEvent(type={self.event_type}, sentiment={self.sentiment})>"


class CommunityInsight(BaseModel):
    """
    AI-generated insights about community behavior and optimization opportunities
    """
    __tablename__ = "community_insights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id = Column(UUID(as_uuid=True), ForeignKey("telegram_communities.id"), nullable=False)
    
    # Insight Details
    insight_type = Column(String(50), nullable=False)  # behavioral, content, timing, strategy
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Insight Data
    confidence_score = Column(Float, nullable=False)  # 0-100 confidence in insight
    impact_potential = Column(String(20), nullable=False)  # low, medium, high
    implementation_difficulty = Column(String(20), nullable=False)  # easy, medium, hard
    
    # Recommendations
    recommended_actions = Column(JSON, default=list)
    expected_outcomes = Column(JSON, default=list)
    
    # Status
    is_implemented = Column(Boolean, default=False)
    implementation_date = Column(DateTime, nullable=True)
    results = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # When insight becomes stale
    
    # Relationships
    community = relationship("TelegramCommunity", backref="insights")
    
    # Indexes
    __table_args__ = (
        Index('ix_insights_community', 'community_id'),
        Index('ix_insights_type', 'insight_type'),
        Index('ix_insights_impact', 'impact_potential'),
        Index('ix_insights_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<CommunityInsight(type={self.insight_type}, confidence={self.confidence_score})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if insight has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score for implementing this insight"""
        impact_weights = {"low": 1, "medium": 2, "high": 3}
        difficulty_weights = {"easy": 3, "medium": 2, "hard": 1}
        
        impact_score = impact_weights.get(self.impact_potential, 1)
        difficulty_score = difficulty_weights.get(self.implementation_difficulty, 1)
        
        return (self.confidence_score / 100) * impact_score * difficulty_score