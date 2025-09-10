"""
Engagement Models

Database models for tracking user engagement patterns, behavioral analysis,
and proactive outreach campaigns for the Reddit bot system.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, Text, JSON, 
    Float, DateTime, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.database.base import FullAuditModel


class EngagementType(str, Enum):
    """Types of engagement activities."""
    MESSAGE = "message"
    COMMAND = "command"
    CALLBACK = "callback"
    VOICE_MESSAGE = "voice_message"
    DOCUMENT = "document"
    STICKER = "sticker"
    REACTION = "reaction"
    INLINE_QUERY = "inline_query"


class SentimentType(str, Enum):
    """Sentiment classification types."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    UNKNOWN = "unknown"


class OutreachType(str, Enum):
    """Types of proactive outreach."""
    MILESTONE_CELEBRATION = "milestone_celebration"
    RE_ENGAGEMENT = "re_engagement"
    PERSONALIZED_CHECKIN = "personalized_checkin"
    FEATURE_SUGGESTION = "feature_suggestion"
    MOOD_SUPPORT = "mood_support"
    TOPIC_FOLLOW_UP = "topic_follow_up"
    ACHIEVEMENT_UNLOCK = "achievement_unlock"


class OutreachStatus(str, Enum):
    """Status of outreach campaigns."""
    SCHEDULED = "scheduled"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    RESPONDED = "responded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserEngagement(FullAuditModel):
    """
    User engagement tracking model.
    
    Stores detailed interaction patterns and behavioral data
    for proactive engagement analysis.
    """
    
    __tablename__ = "user_engagements"
    
    # User reference
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user record"
    )
    
    telegram_id = Column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram user ID for quick lookup"
    )
    
    # Engagement metadata
    engagement_type = Column(
        SQLEnum(EngagementType),
        nullable=False,
        index=True,
        comment="Type of engagement activity"
    )
    
    interaction_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When the interaction occurred"
    )
    
    # Context data
    message_text = Column(
        Text,
        nullable=True,
        comment="Text content of the interaction"
    )
    
    command_name = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Command name if applicable"
    )
    
    session_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Session identifier for grouping interactions"
    )
    
    # Sentiment analysis results
    sentiment_score = Column(
        Float,
        nullable=True,
        comment="Sentiment score (-1 to 1)"
    )
    
    sentiment_type = Column(
        SQLEnum(SentimentType),
        nullable=True,
        index=True,
        comment="Classified sentiment type"
    )
    
    # Behavioral metrics
    response_time_seconds = Column(
        Integer,
        nullable=True,
        comment="Time to respond to bot message"
    )
    
    message_length = Column(
        Integer,
        nullable=True,
        comment="Character count of user message"
    )
    
    contains_emoji = Column(
        Boolean,
        default=False,
        comment="Whether message contains emoji"
    )
    
    contains_question = Column(
        Boolean,
        default=False,
        comment="Whether message contains question"
    )
    
    # Topic and intent analysis
    detected_topics = Column(
        JSONB,
        nullable=True,
        comment="AI-detected topics in the interaction"
    )
    
    user_intent = Column(
        String(100),
        nullable=True,
        comment="Classified user intent"
    )
    
    mood_indicators = Column(
        JSONB,
        nullable=True,
        comment="Detected mood indicators and confidence scores"
    )
    
    # Engagement quality metrics
    engagement_quality_score = Column(
        Float,
        nullable=True,
        comment="Calculated engagement quality (0-1)"
    )
    
    is_meaningful_interaction = Column(
        Boolean,
        default=True,
        comment="Whether interaction was meaningful or just noise"
    )
    
    # Context preservation
    conversation_context = Column(
        JSONB,
        nullable=True,
        comment="Relevant conversation context for this interaction"
    )
    
    previous_bot_message = Column(
        Text,
        nullable=True,
        comment="The bot message this interaction responds to"
    )
    
    # Relationship to user
    # user = relationship("User", back_populates="engagements")
    
    # Database indexes for performance
    __table_args__ = (
        Index('idx_engagement_user_timestamp', 'user_id', 'interaction_timestamp'),
        Index('idx_engagement_telegram_timestamp', 'telegram_id', 'interaction_timestamp'),
        Index('idx_engagement_type_timestamp', 'engagement_type', 'interaction_timestamp'),
        Index('idx_engagement_sentiment', 'sentiment_type', 'sentiment_score'),
        Index('idx_engagement_session', 'session_id', 'interaction_timestamp'),
        Index('idx_engagement_command', 'command_name', 'interaction_timestamp'),
        Index('idx_engagement_quality', 'engagement_quality_score', 'interaction_timestamp'),
    )


class UserBehaviorPattern(FullAuditModel):
    """
    Aggregated user behavior patterns.
    
    Stores computed behavioral insights and engagement patterns
    for proactive engagement decisions.
    """
    
    __tablename__ = "user_behavior_patterns"
    
    # User reference
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="Reference to user record"
    )
    
    telegram_id = Column(
        BigInteger,
        nullable=False,
        unique=True,
        index=True,
        comment="Telegram user ID"
    )
    
    # Activity patterns
    total_interactions = Column(
        Integer,
        default=0,
        comment="Total number of interactions"
    )
    
    daily_interaction_average = Column(
        Float,
        nullable=True,
        comment="Average interactions per day"
    )
    
    most_active_hour = Column(
        Integer,
        nullable=True,
        comment="Hour of day (0-23) when most active"
    )
    
    most_active_day = Column(
        Integer,
        nullable=True,
        comment="Day of week (0-6) when most active"
    )
    
    average_session_length_minutes = Column(
        Float,
        nullable=True,
        comment="Average session duration in minutes"
    )
    
    # Engagement quality metrics
    average_sentiment_score = Column(
        Float,
        nullable=True,
        comment="Average sentiment across all interactions"
    )
    
    dominant_sentiment = Column(
        SQLEnum(SentimentType),
        nullable=True,
        comment="Most common sentiment type"
    )
    
    engagement_quality_trend = Column(
        Float,
        nullable=True,
        comment="Trend in engagement quality (-1 to 1)"
    )
    
    response_time_average_seconds = Column(
        Float,
        nullable=True,
        comment="Average response time to bot messages"
    )
    
    # Interaction preferences
    preferred_interaction_types = Column(
        JSONB,
        nullable=True,
        comment="Ranked list of preferred interaction types"
    )
    
    favorite_commands = Column(
        JSONB,
        nullable=True,
        comment="Most used commands with usage counts"
    )
    
    topic_interests = Column(
        JSONB,
        nullable=True,
        comment="Topics of interest with relevance scores"
    )
    
    # Behavioral indicators
    is_highly_engaged = Column(
        Boolean,
        default=False,
        comment="Whether user shows high engagement patterns"
    )
    
    shows_declining_engagement = Column(
        Boolean,
        default=False,
        comment="Whether engagement is declining"
    )
    
    needs_re_engagement = Column(
        Boolean,
        default=False,
        comment="Whether user needs re-engagement outreach"
    )
    
    churn_risk_score = Column(
        Float,
        nullable=True,
        comment="Predicted churn risk (0-1)"
    )
    
    # Timing insights
    optimal_outreach_hour = Column(
        Integer,
        nullable=True,
        comment="Best hour for proactive outreach"
    )
    
    days_since_last_interaction = Column(
        Integer,
        nullable=True,
        comment="Days since last meaningful interaction"
    )
    
    longest_absence_days = Column(
        Integer,
        nullable=True,
        comment="Longest period without interaction"
    )
    
    # Milestone tracking
    milestones_achieved = Column(
        JSONB,
        nullable=True,
        comment="List of achieved milestones with timestamps"
    )
    
    next_milestone_target = Column(
        String(100),
        nullable=True,
        comment="Next milestone user is approaching"
    )
    
    milestone_progress_percent = Column(
        Float,
        nullable=True,
        comment="Progress toward next milestone (0-100)"
    )
    
    # Last analysis timestamp
    last_pattern_analysis = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When patterns were last computed"
    )
    
    # Analysis metadata
    pattern_analysis_version = Column(
        String(20),
        nullable=True,
        comment="Version of analysis algorithm used"
    )
    
    # Database indexes
    __table_args__ = (
        Index('idx_behavior_churn_risk', 'churn_risk_score', 'updated_at'),
        Index('idx_behavior_engagement_trend', 'engagement_quality_trend', 'updated_at'),
        Index('idx_behavior_needs_reengagement', 'needs_re_engagement', 'updated_at'),
        Index('idx_behavior_last_interaction', 'days_since_last_interaction'),
        Index('idx_behavior_analysis_date', 'last_pattern_analysis'),
    )


class ProactiveOutreach(FullAuditModel):
    """
    Proactive outreach campaign tracking.
    
    Manages and tracks proactive engagement campaigns
    sent to users based on behavioral analysis.
    """
    
    __tablename__ = "proactive_outreaches"
    
    # User reference
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user record"
    )
    
    telegram_id = Column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram user ID"
    )
    
    # Outreach configuration
    outreach_type = Column(
        SQLEnum(OutreachType),
        nullable=False,
        index=True,
        comment="Type of proactive outreach"
    )
    
    campaign_id = Column(
        String(100),
        nullable=True,
        index=True,
        comment="Campaign identifier for grouping outreaches"
    )
    
    priority_score = Column(
        Float,
        nullable=True,
        comment="Outreach priority score (0-1)"
    )
    
    # Scheduling
    scheduled_for = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When outreach should be sent"
    )
    
    optimal_timing_used = Column(
        Boolean,
        default=True,
        comment="Whether optimal timing was considered"
    )
    
    # Content
    message_template = Column(
        String(200),
        nullable=True,
        comment="Template used for the message"
    )
    
    message_content = Column(
        Text,
        nullable=False,
        comment="Actual message content sent"
    )
    
    personalization_data = Column(
        JSONB,
        nullable=True,
        comment="Data used for personalizing the message"
    )
    
    # Context that triggered outreach
    trigger_event = Column(
        String(100),
        nullable=True,
        comment="Event or condition that triggered this outreach"
    )
    
    trigger_data = Column(
        JSONB,
        nullable=True,
        comment="Additional data about the trigger"
    )
    
    behavioral_indicators = Column(
        JSONB,
        nullable=True,
        comment="Behavioral patterns that led to this outreach"
    )
    
    # Delivery tracking
    status = Column(
        SQLEnum(OutreachStatus),
        default=OutreachStatus.SCHEDULED,
        nullable=False,
        index=True,
        comment="Current status of the outreach"
    )
    
    sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When message was actually sent"
    )
    
    delivered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When message was delivered"
    )
    
    read_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When message was read by user"
    )
    
    # Response tracking
    user_responded = Column(
        Boolean,
        default=False,
        comment="Whether user responded to the outreach"
    )
    
    response_time_minutes = Column(
        Integer,
        nullable=True,
        comment="Minutes between outreach and user response"
    )
    
    response_sentiment = Column(
        SQLEnum(SentimentType),
        nullable=True,
        comment="Sentiment of user response"
    )
    
    response_content = Column(
        Text,
        nullable=True,
        comment="User's response content"
    )
    
    # Effectiveness metrics
    engagement_improvement = Column(
        Boolean,
        nullable=True,
        comment="Whether outreach improved engagement"
    )
    
    led_to_extended_session = Column(
        Boolean,
        default=False,
        comment="Whether outreach led to extended conversation"
    )
    
    effectiveness_score = Column(
        Float,
        nullable=True,
        comment="Calculated effectiveness score (0-1)"
    )
    
    # Follow-up tracking
    follow_up_needed = Column(
        Boolean,
        default=False,
        comment="Whether follow-up outreach is needed"
    )
    
    next_follow_up_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When next follow-up should occur"
    )
    
    # Error handling
    failure_reason = Column(
        Text,
        nullable=True,
        comment="Reason for failure if status is failed"
    )
    
    retry_count = Column(
        Integer,
        default=0,
        comment="Number of retry attempts"
    )
    
    max_retries = Column(
        Integer,
        default=3,
        comment="Maximum number of retry attempts"
    )
    
    # Database indexes for performance
    __table_args__ = (
        Index('idx_outreach_user_scheduled', 'user_id', 'scheduled_for'),
        Index('idx_outreach_telegram_scheduled', 'telegram_id', 'scheduled_for'),
        Index('idx_outreach_status_scheduled', 'status', 'scheduled_for'),
        Index('idx_outreach_type_priority', 'outreach_type', 'priority_score'),
        Index('idx_outreach_campaign', 'campaign_id', 'created_at'),
        Index('idx_outreach_effectiveness', 'effectiveness_score', 'outreach_type'),
        Index('idx_outreach_followup', 'follow_up_needed', 'next_follow_up_at'),
    )


class EngagementMilestone(FullAuditModel):
    """
    Engagement milestone definitions and tracking.
    
    Defines various milestones users can achieve and tracks
    when they reach them for celebration outreach.
    """
    
    __tablename__ = "engagement_milestones"
    
    # Milestone definition
    milestone_name = Column(
        String(100),
        nullable=False,
        unique=True,
        comment="Unique name for the milestone"
    )
    
    display_name = Column(
        String(200),
        nullable=False,
        comment="User-friendly display name"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Description of the milestone"
    )
    
    # Milestone criteria
    metric_name = Column(
        String(100),
        nullable=False,
        comment="Name of metric to track"
    )
    
    target_value = Column(
        Float,
        nullable=False,
        comment="Target value to achieve milestone"
    )
    
    metric_type = Column(
        String(50),
        nullable=False,
        comment="Type of metric (count, duration, score, etc.)"
    )
    
    # Milestone configuration
    is_active = Column(
        Boolean,
        default=True,
        comment="Whether milestone is currently active"
    )
    
    celebration_template = Column(
        String(200),
        nullable=True,
        comment="Message template for milestone celebration"
    )
    
    reward_type = Column(
        String(50),
        nullable=True,
        comment="Type of reward for achieving milestone"
    )
    
    reward_data = Column(
        JSONB,
        nullable=True,
        comment="Reward configuration data"
    )
    
    # Milestone metadata
    category = Column(
        String(50),
        nullable=True,
        comment="Milestone category for organization"
    )
    
    difficulty_level = Column(
        Integer,
        nullable=True,
        comment="Difficulty level (1-5)"
    )
    
    estimated_days_to_achieve = Column(
        Integer,
        nullable=True,
        comment="Estimated days for average user to achieve"
    )
    
    # Usage statistics
    total_achievements = Column(
        Integer,
        default=0,
        comment="Total number of users who achieved this milestone"
    )
    
    average_days_to_achieve = Column(
        Float,
        nullable=True,
        comment="Actual average days to achieve"
    )
    
    last_achieved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this milestone was last achieved"
    )
    
    # Database indexes
    __table_args__ = (
        Index('idx_milestone_active', 'is_active', 'metric_name'),
        Index('idx_milestone_category', 'category', 'difficulty_level'),
        Index('idx_milestone_achievements', 'total_achievements', 'updated_at'),
    )


class UserMilestoneProgress(FullAuditModel):
    """
    User progress toward milestones.
    
    Tracks individual user progress toward various milestones
    for proactive engagement and celebration timing.
    """
    
    __tablename__ = "user_milestone_progress"
    
    # References
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user record"
    )
    
    milestone_id = Column(
        UUID(as_uuid=True),
        ForeignKey("engagement_milestones.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to milestone record"
    )
    
    telegram_id = Column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram user ID"
    )
    
    # Progress tracking
    current_value = Column(
        Float,
        default=0.0,
        comment="Current progress value"
    )
    
    target_value = Column(
        Float,
        nullable=False,
        comment="Target value to achieve milestone"
    )
    
    progress_percentage = Column(
        Float,
        nullable=True,
        comment="Progress as percentage (0-100)"
    )
    
    # Status
    is_achieved = Column(
        Boolean,
        default=False,
        comment="Whether milestone has been achieved"
    )
    
    achieved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When milestone was achieved"
    )
    
    days_to_achieve = Column(
        Integer,
        nullable=True,
        comment="Days taken to achieve milestone"
    )
    
    # Celebration tracking
    celebration_sent = Column(
        Boolean,
        default=False,
        comment="Whether celebration message was sent"
    )
    
    celebration_sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When celebration message was sent"
    )
    
    # Progress metadata
    last_updated_value = Column(
        Float,
        nullable=True,
        comment="Previous value before last update"
    )
    
    progress_velocity = Column(
        Float,
        nullable=True,
        comment="Rate of progress (value increase per day)"
    )
    
    estimated_completion_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Estimated date to achieve milestone"
    )
    
    # Milestone relationships
    # user = relationship("User", back_populates="milestone_progress")
    # milestone = relationship("EngagementMilestone", back_populates="user_progress")
    
    # Database indexes
    __table_args__ = (
        Index('idx_milestone_progress_user', 'user_id', 'is_achieved'),
        Index('idx_milestone_progress_telegram', 'telegram_id', 'is_achieved'),
        Index('idx_milestone_progress_achieved', 'is_achieved', 'achieved_at'),
        Index('idx_milestone_progress_completion', 'estimated_completion_date', 'progress_percentage'),
        Index('idx_milestone_progress_celebration', 'celebration_sent', 'is_achieved'),
        # Composite unique constraint
        Index('idx_milestone_progress_unique', 'user_id', 'milestone_id', unique=True),
    )
    
    def update_progress(self, new_value: float) -> bool:
        """
        Update milestone progress.
        
        Args:
            new_value: New progress value
            
        Returns:
            True if milestone was achieved with this update
        """
        self.last_updated_value = self.current_value
        self.current_value = new_value
        self.progress_percentage = min((new_value / self.target_value) * 100, 100.0)
        
        # Calculate progress velocity
        if self.updated_at and self.last_updated_value is not None:
            days_since_update = (datetime.utcnow() - self.updated_at).days
            if days_since_update > 0:
                value_increase = new_value - self.last_updated_value
                self.progress_velocity = value_increase / days_since_update
        
        # Check if milestone achieved
        if not self.is_achieved and new_value >= self.target_value:
            self.is_achieved = True
            self.achieved_at = datetime.utcnow()
            
            if self.created_at:
                self.days_to_achieve = (datetime.utcnow() - self.created_at).days
            
            return True
        
        # Update estimated completion date
        if self.progress_velocity and self.progress_velocity > 0 and not self.is_achieved:
            remaining_value = self.target_value - new_value
            days_remaining = remaining_value / self.progress_velocity
            self.estimated_completion_date = datetime.utcnow() + timedelta(days=days_remaining)
        
        return False