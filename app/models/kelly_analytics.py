"""
Kelly AI Analytics Database Models

SQLAlchemy models for Phase 3 analytics system including conversation analysis,
performance metrics, revenue attribution, and business intelligence.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database.connection import Base

class ConversationAnalytics(Base):
    """Model for storing detailed conversation analytics and AI insights"""
    __tablename__ = "kelly_conversation_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # Conversation metadata
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    channel_type = Column(String(50), nullable=False)  # telegram, whatsapp, discord, etc.
    channel_id = Column(String(100), nullable=True)
    
    # Conversation metrics
    total_messages = Column(Integer, nullable=False, default=0)
    ai_messages = Column(Integer, nullable=False, default=0)
    human_messages = Column(Integer, nullable=False, default=0)
    human_intervention_count = Column(Integer, nullable=False, default=0)
    
    # Timing metrics
    started_at = Column(DateTime, nullable=False, index=True)
    ended_at = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration_minutes = Column(Float, nullable=True)
    avg_response_time_seconds = Column(Float, nullable=True)
    
    # Quality metrics
    conversation_quality_score = Column(Float, nullable=False, default=0.0)
    ai_confidence_avg = Column(Float, nullable=True)
    ai_confidence_min = Column(Float, nullable=True)
    ai_confidence_max = Column(Float, nullable=True)
    safety_score_avg = Column(Float, nullable=False, default=1.0)
    safety_score_min = Column(Float, nullable=True)
    
    # Sentiment analysis
    sentiment_overall = Column(String(20), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    emotion_primary = Column(String(50), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    
    # Topic analysis
    topics_detected = Column(JSON, nullable=True)  # List of topics with confidence scores
    topic_primary = Column(String(100), nullable=True)
    intent_detected = Column(String(100), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    
    # Engagement metrics
    user_satisfaction_score = Column(Float, nullable=True)  # 1-5 rating if available
    user_feedback_positive = Column(Boolean, nullable=True)
    resolution_achieved = Column(Boolean, nullable=False, default=False)
    escalation_required = Column(Boolean, nullable=False, default=False)
    
    # Business metrics
    lead_qualified = Column(Boolean, nullable=False, default=False)
    lead_score = Column(Float, nullable=True)
    conversion_event = Column(Boolean, nullable=False, default=False)
    revenue_attributed = Column(Float, nullable=True)
    
    # Patterns and insights
    conversation_pattern = Column(String(100), nullable=True)
    success_pattern_match = Column(Boolean, nullable=False, default=False)
    anomaly_detected = Column(Boolean, nullable=False, default=False)
    anomaly_type = Column(String(100), nullable=True)
    anomaly_confidence = Column(Float, nullable=True)
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional insights
    metadata = Column(JSON, nullable=True)
    ai_insights = Column(JSON, nullable=True)  # AI-generated insights about conversation
    key_moments = Column(JSON, nullable=True)  # Important moments in conversation
    
    # Performance indexes
    __table_args__ = (
        Index('idx_account_started', 'account_id', 'started_at'),
        Index('idx_user_started', 'user_id', 'started_at'),
        Index('idx_quality_score', 'conversation_quality_score'),
        Index('idx_lead_qualified', 'lead_qualified', 'started_at'),
        Index('idx_conversion_event', 'conversion_event', 'started_at'),
        Index('idx_anomaly_detected', 'anomaly_detected', 'started_at'),
    )

class PerformanceMetric(Base):
    """Model for storing team and AI performance metrics"""
    __tablename__ = "kelly_performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(50), nullable=False, index=True)  # team, ai, comparison, efficiency
    
    # Metric context
    account_id = Column(String(50), nullable=True, index=True)
    team_id = Column(String(50), nullable=True, index=True)
    operator_id = Column(String(50), nullable=True, index=True)
    
    # Time period
    period_type = Column(String(20), nullable=False)  # hour, day, week, month
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Performance metrics
    conversations_handled = Column(Integer, nullable=False, default=0)
    avg_response_time_seconds = Column(Float, nullable=True)
    median_response_time_seconds = Column(Float, nullable=True)
    p95_response_time_seconds = Column(Float, nullable=True)
    
    # Quality metrics
    avg_quality_score = Column(Float, nullable=True)
    customer_satisfaction_avg = Column(Float, nullable=True)
    resolution_rate = Column(Float, nullable=True)  # Percentage
    escalation_rate = Column(Float, nullable=True)  # Percentage
    
    # AI vs Human comparison
    ai_handled_count = Column(Integer, nullable=False, default=0)
    human_handled_count = Column(Integer, nullable=False, default=0)
    ai_success_rate = Column(Float, nullable=True)
    human_success_rate = Column(Float, nullable=True)
    ai_efficiency_score = Column(Float, nullable=True)
    human_efficiency_score = Column(Float, nullable=True)
    
    # Cost metrics
    total_cost = Column(Float, nullable=True)
    cost_per_conversation = Column(Float, nullable=True)
    ai_cost = Column(Float, nullable=True)
    human_cost = Column(Float, nullable=True)
    cost_savings = Column(Float, nullable=True)
    
    # Business impact
    revenue_generated = Column(Float, nullable=True)
    leads_qualified = Column(Integer, nullable=False, default=0)
    conversions_achieved = Column(Integer, nullable=False, default=0)
    roi_calculated = Column(Float, nullable=True)
    
    # Error and intervention tracking
    error_count = Column(Integer, nullable=False, default=0)
    intervention_count = Column(Integer, nullable=False, default=0)
    safety_incidents = Column(Integer, nullable=False, default=0)
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_metric_type_period', 'metric_type', 'period_start'),
        Index('idx_account_period', 'account_id', 'period_start'),
        Index('idx_team_period', 'team_id', 'period_start'),
        Index('idx_operator_period', 'operator_id', 'period_start'),
    )

class RevenueAttribution(Base):
    """Model for tracking revenue attribution to conversations and AI interactions"""
    __tablename__ = "kelly_revenue_attribution"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Attribution context
    conversation_id = Column(String(100), nullable=False, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    
    # Revenue details
    revenue_amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False, default='USD')
    attribution_type = Column(String(50), nullable=False)  # direct, assisted, influenced
    attribution_confidence = Column(Float, nullable=False, default=1.0)
    
    # Attribution source
    touchpoint_type = Column(String(50), nullable=False)  # conversation, recommendation, follow_up
    ai_contribution = Column(Float, nullable=True)  # Percentage of AI contribution
    human_contribution = Column(Float, nullable=True)  # Percentage of human contribution
    
    # Timeline
    conversation_started_at = Column(DateTime, nullable=False)
    conversion_occurred_at = Column(DateTime, nullable=False, index=True)
    attribution_delay_hours = Column(Float, nullable=True)
    
    # Deal/transaction details
    deal_id = Column(String(100), nullable=True, index=True)
    transaction_id = Column(String(100), nullable=True)
    product_category = Column(String(100), nullable=True)
    deal_size_category = Column(String(50), nullable=True)  # small, medium, large, enterprise
    
    # Customer journey stage
    journey_stage = Column(String(50), nullable=False)  # awareness, consideration, decision, retention
    customer_lifecycle_stage = Column(String(50), nullable=True)  # new, returning, expanding, churning
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(String(50), nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    attribution_factors = Column(JSON, nullable=True)  # Factors that contributed to attribution
    
    # Performance indexes
    __table_args__ = (
        Index('idx_account_conversion', 'account_id', 'conversion_occurred_at'),
        Index('idx_user_conversion', 'user_id', 'conversion_occurred_at'),
        Index('idx_deal_attribution', 'deal_id', 'attribution_type'),
        Index('idx_journey_stage', 'journey_stage', 'conversion_occurred_at'),
    )

class QualityScore(Base):
    """Model for storing detailed conversation quality assessments"""
    __tablename__ = "kelly_quality_scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(100), nullable=False, index=True)
    
    # Score context
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    message_id = Column(String(100), nullable=True)  # Specific message if applicable
    
    # Overall quality score
    overall_score = Column(Float, nullable=False)  # 0.0 - 1.0
    overall_grade = Column(String(10), nullable=False)  # A+, A, B+, B, C+, C, D, F
    
    # Component scores
    relevance_score = Column(Float, nullable=False, default=0.0)
    helpfulness_score = Column(Float, nullable=False, default=0.0)
    accuracy_score = Column(Float, nullable=False, default=0.0)
    empathy_score = Column(Float, nullable=False, default=0.0)
    professionalism_score = Column(Float, nullable=False, default=0.0)
    efficiency_score = Column(Float, nullable=False, default=0.0)
    
    # AI-specific scores
    ai_confidence = Column(Float, nullable=True)
    ai_uncertainty_level = Column(Float, nullable=True)
    hallucination_risk = Column(Float, nullable=False, default=0.0)
    coherence_score = Column(Float, nullable=True)
    
    # Safety and compliance scores
    safety_score = Column(Float, nullable=False, default=1.0)
    bias_score = Column(Float, nullable=False, default=0.0)
    toxicity_score = Column(Float, nullable=False, default=0.0)
    privacy_compliance_score = Column(Float, nullable=False, default=1.0)
    
    # Scoring methodology
    scoring_method = Column(String(50), nullable=False)  # ai_automatic, human_review, hybrid
    scored_by = Column(String(50), nullable=True)  # AI model or human reviewer ID
    scoring_model_version = Column(String(50), nullable=True)
    
    # Context factors
    conversation_complexity = Column(Float, nullable=True)
    user_satisfaction_indicator = Column(Float, nullable=True)
    resolution_achieved = Column(Boolean, nullable=True)
    
    # Performance tracking
    scored_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    conversation_timestamp = Column(DateTime, nullable=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    scoring_details = Column(JSON, nullable=True)  # Detailed breakdown of scoring
    improvement_suggestions = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_conversation_scored', 'conversation_id', 'scored_at'),
        Index('idx_account_overall_score', 'account_id', 'overall_score'),
        Index('idx_scoring_method', 'scoring_method', 'scored_at'),
        Index('idx_safety_score', 'safety_score', 'scored_at'),
    )

class TrendAnalysis(Base):
    """Model for storing trend analysis data and forecasting"""
    __tablename__ = "kelly_trend_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Trend context
    trend_type = Column(String(50), nullable=False, index=True)  # performance, quality, volume, satisfaction
    metric_name = Column(String(100), nullable=False, index=True)
    
    # Scope
    account_id = Column(String(50), nullable=True, index=True)
    team_id = Column(String(50), nullable=True, index=True)
    global_trend = Column(Boolean, nullable=False, default=False)
    
    # Time period
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly, quarterly
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Trend data
    current_value = Column(Float, nullable=False)
    previous_value = Column(Float, nullable=True)
    change_amount = Column(Float, nullable=True)
    change_percentage = Column(Float, nullable=True)
    trend_direction = Column(String(20), nullable=False)  # up, down, stable, volatile
    
    # Statistical analysis
    moving_average_7d = Column(Float, nullable=True)
    moving_average_30d = Column(Float, nullable=True)
    standard_deviation = Column(Float, nullable=True)
    volatility_score = Column(Float, nullable=True)
    
    # Forecasting
    forecast_next_period = Column(Float, nullable=True)
    forecast_confidence = Column(Float, nullable=True)
    forecast_method = Column(String(50), nullable=True)  # linear, exponential, ml_model
    
    # Significance
    statistical_significance = Column(Float, nullable=True)
    practical_significance = Column(Boolean, nullable=False, default=False)
    alert_threshold_breached = Column(Boolean, nullable=False, default=False)
    
    # Performance tracking
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    historical_data = Column(JSON, nullable=True)  # Historical data points used
    forecast_details = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_trend_type_period', 'trend_type', 'period_start'),
        Index('idx_metric_period', 'metric_name', 'period_start'),
        Index('idx_account_trend', 'account_id', 'trend_type', 'period_start'),
        Index('idx_global_trends', 'global_trend', 'trend_type', 'period_start'),
    )

class CustomAnalyticsQuery(Base):
    """Model for storing custom analytics queries and results"""
    __tablename__ = "kelly_custom_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Query context
    query_name = Column(String(200), nullable=False)
    query_type = Column(String(50), nullable=False)  # filter, aggregation, comparison, forecast
    created_by = Column(String(50), nullable=False, index=True)
    
    # Query parameters
    filters = Column(JSON, nullable=True)  # Filter conditions
    aggregations = Column(JSON, nullable=True)  # Aggregation functions
    grouping = Column(JSON, nullable=True)  # Group by fields
    time_range = Column(JSON, nullable=False)  # Time range for query
    
    # Execution details
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    execution_time_ms = Column(Integer, nullable=True)
    rows_returned = Column(Integer, nullable=True)
    
    # Results cache
    results = Column(JSON, nullable=True)  # Query results (if cacheable)
    results_hash = Column(String(64), nullable=True, index=True)  # Hash for deduplication
    cache_expires_at = Column(DateTime, nullable=True)
    
    # Query metadata
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Tags for organization
    shared = Column(Boolean, nullable=False, default=False)
    shared_with = Column(JSON, nullable=True)  # List of user IDs
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_executed = Column(DateTime, nullable=False, default=datetime.utcnow)
    execution_count = Column(Integer, nullable=False, default=1)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    query_sql = Column(Text, nullable=True)  # Generated SQL for debugging
    
    # Performance indexes
    __table_args__ = (
        Index('idx_created_by_executed', 'created_by', 'executed_at'),
        Index('idx_query_type', 'query_type', 'executed_at'),
        Index('idx_results_hash', 'results_hash'),
        Index('idx_shared_queries', 'shared', 'executed_at'),
    )

def create_kelly_analytics_tables():
    """Create all Kelly analytics tables"""
    from app.database.connection import engine
    Base.metadata.create_all(bind=engine)