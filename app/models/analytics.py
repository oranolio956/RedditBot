"""
Analytics Models

Defines models for tracking user activity, conversation analytics, and system metrics.
Supports comprehensive analytics and reporting for AI conversation insights.
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text, Date, ForeignKey,
    Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class ActivityType(str, Enum):
    """User activity type enumeration."""
    MESSAGE_SENT = "message_sent"
    COMMAND_EXECUTED = "command_executed"
    CALLBACK_PRESSED = "callback_pressed"
    INLINE_QUERY = "inline_query"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    FILE_UPLOADED = "file_uploaded"
    FEEDBACK_PROVIDED = "feedback_provided"
    SETTING_CHANGED = "setting_changed"
    ERROR_OCCURRED = "error_occurred"


class MetricType(str, Enum):
    """System metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AnalyticsPeriod(str, Enum):
    """Analytics aggregation period enumeration."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class UserActivity(FullAuditModel):
    """
    User activity tracking for detailed analytics.
    
    Records individual user actions and interactions for
    behavior analysis and usage patterns.
    """
    
    __tablename__ = "user_activities"
    
    # User reference
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who performed the activity"
    )
    
    # Activity details
    activity_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of activity performed"
    )
    
    activity_name = Column(
        String(100),
        nullable=False,
        comment="Specific name of the activity"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Detailed description of the activity"
    )
    
    # Context and metadata
    context_data = Column(
        JSONB,
        nullable=True,
        comment="Additional context data for the activity"
    )
    
    session_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Session ID when activity occurred"
    )
    
    conversation_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Conversation ID if applicable"
    )
    
    # Technical details
    ip_address = Column(
        String(45),
        nullable=True,
        comment="IP address of user"
    )
    
    user_agent = Column(
        String(500),
        nullable=True,
        comment="User agent string"
    )
    
    platform = Column(
        String(50),
        nullable=True,
        comment="Platform used (web, mobile, desktop, etc.)"
    )
    
    # Timing and performance
    activity_timestamp = Column(
        "activity_timestamp",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When the activity occurred"
    )
    
    processing_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken to process activity in milliseconds"
    )
    
    response_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken to respond to activity in milliseconds"
    )
    
    # Success and error tracking
    success = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the activity was successful"
    )
    
    error_code = Column(
        String(50),
        nullable=True,
        comment="Error code if activity failed"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if activity failed"
    )
    
    # Analytics aggregation fields
    date_partition = Column(
        Date,
        nullable=False,
        index=True,
        comment="Date for partitioning and aggregation"
    )
    
    hour_partition = Column(
        Integer,
        nullable=False,
        index=True,
        comment="Hour of day (0-23) for time-based analytics"
    )
    
    # Relationships
    user = relationship("User", back_populates="activities")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_activity_user_type', 'user_id', 'activity_type'),
        Index('idx_activity_timestamp', 'activity_timestamp'),
        Index('idx_activity_date_hour', 'date_partition', 'hour_partition'),
        Index('idx_activity_session', 'session_id'),
        Index('idx_activity_success', 'success'),
        CheckConstraint('processing_time_ms >= 0', name='ck_processing_time_positive'),
        CheckConstraint('response_time_ms >= 0', name='ck_response_time_positive'),
        CheckConstraint('hour_partition >= 0 AND hour_partition <= 23', name='ck_hour_partition_range'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-set partition fields
        now = datetime.utcnow()
        if not self.date_partition:
            self.date_partition = now.date()
        if not self.hour_partition:
            self.hour_partition = now.hour
        if not self.activity_timestamp:
            self.activity_timestamp = now
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get activity summary for reporting."""
        return {
            "activity_id": str(self.id),
            "user_id": str(self.user_id),
            "activity_type": self.activity_type,
            "activity_name": self.activity_name,
            "success": self.success,
            "processing_time_ms": self.processing_time_ms,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.activity_timestamp.isoformat() if self.activity_timestamp else None,
        }


class ConversationAnalytics(FullAuditModel):
    """
    Conversation-level analytics and insights.
    
    Aggregates conversation data for analysis of conversation quality,
    user engagement, and AI performance.
    """
    
    __tablename__ = "conversation_analytics"
    
    # Conversation reference
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Conversation being analyzed"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User in the conversation"
    )
    
    session_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Session the conversation belongs to"
    )
    
    # Basic conversation metrics
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of messages in conversation"
    )
    
    user_message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of messages from user"
    )
    
    bot_message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of messages from bot"
    )
    
    # Conversation duration and timing
    conversation_duration_seconds = Column(
        Integer,
        nullable=True,
        comment="Total conversation duration in seconds"
    )
    
    average_response_time_ms = Column(
        Float,
        nullable=True,
        comment="Average bot response time in milliseconds"
    )
    
    user_response_time_ms = Column(
        Float,
        nullable=True,
        comment="Average user response time in milliseconds"
    )
    
    # Engagement metrics
    engagement_score = Column(
        Float,
        nullable=True,
        comment="Overall engagement score (0-1)"
    )
    
    interaction_intensity = Column(
        Float,
        nullable=True,
        comment="Messages per minute during active periods"
    )
    
    conversation_completion_rate = Column(
        Float,
        nullable=True,
        comment="How much of intended conversation was completed (0-1)"
    )
    
    # Content analysis
    average_message_length = Column(
        Float,
        nullable=True,
        comment="Average message length in characters"
    )
    
    total_content_length = Column(
        Integer,
        nullable=True,
        comment="Total content length in characters"
    )
    
    unique_words_count = Column(
        Integer,
        nullable=True,
        comment="Number of unique words used"
    )
    
    vocabulary_richness = Column(
        Float,
        nullable=True,
        comment="Vocabulary richness score (unique words / total words)"
    )
    
    # Sentiment and emotion analysis
    average_sentiment_score = Column(
        Float,
        nullable=True,
        comment="Average sentiment score (-1 to 1)"
    )
    
    sentiment_variance = Column(
        Float,
        nullable=True,
        comment="Variance in sentiment throughout conversation"
    )
    
    emotion_distribution = Column(
        JSONB,
        nullable=True,
        comment="Distribution of emotions detected"
    )
    
    mood_progression = Column(
        JSONB,
        nullable=True,
        comment="How mood changed during conversation"
    )
    
    # Topic and intent analysis
    primary_topics = Column(
        ARRAY(String),
        nullable=True,
        comment="Primary topics discussed"
    )
    
    topic_transitions = Column(
        Integer,
        nullable=True,
        comment="Number of topic transitions"
    )
    
    intent_distribution = Column(
        JSONB,
        nullable=True,
        comment="Distribution of user intents"
    )
    
    goal_achievement_score = Column(
        Float,
        nullable=True,
        comment="How well conversation goals were achieved (0-1)"
    )
    
    # Quality metrics
    coherence_score = Column(
        Float,
        nullable=True,
        comment="Conversation coherence score (0-1)"
    )
    
    relevance_score = Column(
        Float,
        nullable=True,
        comment="Response relevance score (0-1)"
    )
    
    helpfulness_score = Column(
        Float,
        nullable=True,
        comment="Perceived helpfulness score (0-1)"
    )
    
    # User feedback
    user_satisfaction_score = Column(
        Float,
        nullable=True,
        comment="User satisfaction score (0-1)"
    )
    
    explicit_feedback_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of explicit feedback items received"
    )
    
    positive_feedback_ratio = Column(
        Float,
        nullable=True,
        comment="Ratio of positive to total feedback"
    )
    
    # Risk and safety metrics
    risk_incidents_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of risk incidents detected"
    )
    
    safety_score = Column(
        Float,
        nullable=True,
        comment="Overall safety score (0-1)"
    )
    
    content_moderation_flags = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of content moderation flags"
    )
    
    # Technical performance
    error_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of errors during conversation"
    )
    
    average_processing_time = Column(
        Float,
        nullable=True,
        comment="Average message processing time in milliseconds"
    )
    
    model_versions_used = Column(
        JSONB,
        nullable=True,
        comment="AI model versions used during conversation"
    )
    
    # Advanced analytics
    conversation_complexity = Column(
        Float,
        nullable=True,
        comment="Conversation complexity score (0-1)"
    )
    
    personality_adaptation_score = Column(
        Float,
        nullable=True,
        comment="How well AI adapted to user personality (0-1)"
    )
    
    learning_effectiveness = Column(
        Float,
        nullable=True,
        comment="How effective the conversation was for AI learning (0-1)"
    )
    
    # Relationships
    conversation = relationship("Conversation", back_populates="analytics")
    user = relationship("User", back_populates="conversation_analytics")
    
    # Database constraints
    __table_args__ = (
        Index('idx_conv_analytics_user', 'user_id'),
        Index('idx_conv_analytics_session', 'session_id'),
        Index('idx_conv_analytics_engagement', 'engagement_score'),
        Index('idx_conv_analytics_satisfaction', 'user_satisfaction_score'),
        CheckConstraint('message_count >= 0', name='ck_message_count_positive'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='ck_engagement_range'),
        CheckConstraint('user_satisfaction_score >= 0 AND user_satisfaction_score <= 1', name='ck_satisfaction_range'),
        CheckConstraint('average_sentiment_score >= -1 AND average_sentiment_score <= 1', name='ck_sentiment_range'),
        CheckConstraint('coherence_score >= 0 AND coherence_score <= 1', name='ck_coherence_range'),
        CheckConstraint('safety_score >= 0 AND safety_score <= 1', name='ck_safety_range'),
    )
    
    def calculate_metrics(self, conversation_data: Dict[str, Any]) -> None:
        """Calculate all analytics metrics from conversation data."""
        # Basic metrics
        self.message_count = conversation_data.get('total_messages', 0)
        self.user_message_count = conversation_data.get('user_messages', 0)
        self.bot_message_count = conversation_data.get('bot_messages', 0)
        
        # Duration and timing
        if 'start_time' in conversation_data and 'end_time' in conversation_data:
            duration = conversation_data['end_time'] - conversation_data['start_time']
            self.conversation_duration_seconds = int(duration.total_seconds())
        
        # Response times
        response_times = conversation_data.get('response_times', [])
        if response_times:
            self.average_response_time_ms = sum(response_times) / len(response_times)
        
        # Engagement calculation
        self.engagement_score = self._calculate_engagement_score(conversation_data)
        
        # Content analysis
        self._analyze_content(conversation_data)
        
        # Sentiment analysis
        self._analyze_sentiment(conversation_data)
        
        # Quality metrics
        self._calculate_quality_metrics(conversation_data)
    
    def _calculate_engagement_score(self, data: Dict[str, Any]) -> float:
        """Calculate user engagement score."""
        factors = []
        
        # Message frequency
        if self.conversation_duration_seconds and self.conversation_duration_seconds > 0:
            msg_per_min = (self.message_count / self.conversation_duration_seconds) * 60
            factors.append(min(1.0, msg_per_min / 10))  # Normalize to 10 messages/min max
        
        # Response consistency
        response_times = data.get('response_times', [])
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            consistency = 1.0 / (1.0 + (avg_time / 10000))  # Normalize around 10s
            factors.append(consistency)
        
        # Conversation completion
        if 'completion_indicators' in data:
            completion = data['completion_indicators'].get('completion_score', 0.5)
            factors.append(completion)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _analyze_content(self, data: Dict[str, Any]) -> None:
        """Analyze conversation content."""
        messages = data.get('messages', [])
        if not messages:
            return
        
        total_length = 0
        word_set = set()
        total_words = 0
        
        for msg in messages:
            content = msg.get('content', '')
            total_length += len(content)
            words = content.split()
            total_words += len(words)
            word_set.update(word.lower() for word in words)
        
        self.total_content_length = total_length
        self.average_message_length = total_length / len(messages)
        self.unique_words_count = len(word_set)
        
        if total_words > 0:
            self.vocabulary_richness = len(word_set) / total_words
    
    def _analyze_sentiment(self, data: Dict[str, Any]) -> None:
        """Analyze sentiment patterns."""
        sentiments = data.get('sentiment_scores', [])
        if not sentiments:
            return
        
        self.average_sentiment_score = sum(sentiments) / len(sentiments)
        
        # Calculate variance
        mean = self.average_sentiment_score
        variance = sum((s - mean) ** 2 for s in sentiments) / len(sentiments)
        self.sentiment_variance = variance
    
    def _calculate_quality_metrics(self, data: Dict[str, Any]) -> None:
        """Calculate conversation quality metrics."""
        # Coherence based on topic consistency
        topics = data.get('topics', [])
        if topics:
            unique_topics = len(set(topics))
            self.topic_transitions = max(0, unique_topics - 1)
            # Higher coherence for fewer topic jumps
            self.coherence_score = max(0.0, 1.0 - (self.topic_transitions / len(topics)))
        
        # Relevance based on intent matching
        intents = data.get('intents', [])
        if intents:
            matched_intents = sum(1 for intent in intents if intent.get('matched', False))
            self.relevance_score = matched_intents / len(intents)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return {
            "conversation_id": str(self.conversation_id),
            "message_count": self.message_count,
            "duration_minutes": self.conversation_duration_seconds // 60 if self.conversation_duration_seconds else None,
            "engagement_score": self.engagement_score,
            "satisfaction_score": self.user_satisfaction_score,
            "sentiment_score": self.average_sentiment_score,
            "coherence_score": self.coherence_score,
            "safety_score": self.safety_score,
            "response_time_ms": self.average_response_time_ms,
            "topic_count": len(self.primary_topics) if self.primary_topics else 0,
            "risk_incidents": self.risk_incidents_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SystemMetrics(BaseModel):
    """
    System-wide metrics and performance monitoring.
    
    Tracks system health, performance, and usage statistics
    for monitoring and alerting.
    """
    
    __tablename__ = "system_metrics"
    
    # Metric identification
    metric_name = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Name of the metric"
    )
    
    metric_type = Column(
        String(20),
        nullable=False,
        comment="Type of metric (counter, gauge, histogram, etc.)"
    )
    
    category = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Metric category (performance, usage, error, etc.)"
    )
    
    # Metric values
    value = Column(
        Float,
        nullable=False,
        comment="Metric value"
    )
    
    count = Column(
        BigInteger,
        default=1,
        nullable=False,
        comment="Number of observations for this metric"
    )
    
    min_value = Column(
        Float,
        nullable=True,
        comment="Minimum value observed"
    )
    
    max_value = Column(
        Float,
        nullable=True,
        comment="Maximum value observed"
    )
    
    sum_value = Column(
        Float,
        nullable=True,
        comment="Sum of all values (for calculating averages)"
    )
    
    # Metadata and labels
    labels = Column(
        JSONB,
        nullable=True,
        comment="Metric labels and dimensions"
    )
    
    tags = Column(
        ARRAY(String),
        nullable=True,
        comment="Metric tags for filtering"
    )
    
    # Time partitioning
    timestamp = Column(
        "timestamp",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="Metric timestamp"
    )
    
    date_partition = Column(
        Date,
        nullable=False,
        index=True,
        comment="Date partition for aggregation"
    )
    
    hour_partition = Column(
        Integer,
        nullable=False,
        index=True,
        comment="Hour partition (0-23)"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_metrics_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_metrics_category_date', 'category', 'date_partition'),
        Index('idx_metrics_type', 'metric_type'),
        Index('idx_metrics_partition', 'date_partition', 'hour_partition'),
        UniqueConstraint('metric_name', 'timestamp', 'labels', name='uq_metric_point'),
        CheckConstraint('count >= 0', name='ck_count_positive'),
        CheckConstraint('hour_partition >= 0 AND hour_partition <= 23', name='ck_hour_range'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-set partition fields
        now = datetime.utcnow()
        if not self.timestamp:
            self.timestamp = now
        if not self.date_partition:
            self.date_partition = now.date()
        if not self.hour_partition:
            self.hour_partition = now.hour
    
    def update_value(self, new_value: float) -> None:
        """Update metric with new value."""
        if self.metric_type == MetricType.GAUGE:
            # Gauge - replace value
            self.value = new_value
        elif self.metric_type == MetricType.COUNTER:
            # Counter - increment
            self.value += new_value
        else:
            # For other types, update statistics
            self.count += 1
            self.sum_value = (self.sum_value or 0) + new_value
            self.value = self.sum_value / self.count  # Average
            
            if self.min_value is None or new_value < self.min_value:
                self.min_value = new_value
            if self.max_value is None or new_value > self.max_value:
                self.max_value = new_value
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get metric summary."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "category": self.category,
            "value": self.value,
            "count": self.count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# Update User and Conversation models to include analytics relationships
User.activities = relationship("UserActivity", back_populates="user", cascade="all, delete-orphan")
User.conversation_analytics = relationship("ConversationAnalytics", back_populates="user", cascade="all, delete-orphan")
Conversation.analytics = relationship("ConversationAnalytics", back_populates="conversation", uselist=False, cascade="all, delete-orphan")