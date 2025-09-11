"""
Kelly AI Intelligence Database Models

SQLAlchemy models for AI-powered conversation intelligence, pattern recognition,
recommendations, and anomaly detection.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database.connection import Base

class ConversationInsight(Base):
    """AI-generated insights about conversations and interactions"""
    __tablename__ = "kelly_conversation_insights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(100), nullable=False, index=True)
    
    # Insight context
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    message_id = Column(String(100), nullable=True)  # Specific message if applicable
    
    # Insight classification
    insight_type = Column(String(50), nullable=False, index=True)  # pattern, anomaly, opportunity, risk, coaching
    insight_category = Column(String(50), nullable=False, index=True)  # sentiment, intent, behavior, performance
    priority = Column(String(20), nullable=False, default='medium')  # low, medium, high, critical
    
    # Insight details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    
    # AI confidence and reasoning
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0
    ai_model_used = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    reasoning = Column(Text, nullable=True)
    
    # Supporting evidence
    evidence_data = Column(JSON, nullable=True)  # Data that supports the insight
    related_messages = Column(JSON, nullable=True)  # IDs of related messages
    patterns_detected = Column(JSON, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    
    # Impact assessment
    impact_score = Column(Float, nullable=False, default=0.0)  # 0.0 - 1.0
    business_impact = Column(String(50), nullable=True)  # positive, negative, neutral
    customer_impact = Column(String(50), nullable=True)
    
    # Actionability
    actionable = Column(Boolean, nullable=False, default=True)
    recommended_actions = Column(JSON, nullable=True)
    urgency_level = Column(String(20), nullable=False, default='normal')
    
    # Status tracking
    status = Column(String(30), nullable=False, default='new')  # new, reviewed, acted_on, dismissed
    reviewed_by = Column(String(50), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    acted_on = Column(Boolean, nullable=False, default=False)
    action_taken = Column(Text, nullable=True)
    
    # Performance tracking
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True)  # Some insights may expire
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    conversation_context = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_conversation_generated', 'conversation_id', 'generated_at'),
        Index('idx_insight_type_priority', 'insight_type', 'priority'),
        Index('idx_account_category', 'account_id', 'insight_category'),
        Index('idx_status_urgency', 'status', 'urgency_level'),
        Index('idx_confidence_score', 'confidence_score'),
    )

class ConversationPattern(Base):
    """Successful conversation patterns identified by AI"""
    __tablename__ = "kelly_conversation_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Pattern identification
    pattern_name = Column(String(200), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)  # success, failure, escalation, conversion
    pattern_category = Column(String(50), nullable=False)  # opening, objection_handling, closing, follow_up
    
    # Pattern details
    description = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    pattern_signature = Column(JSON, nullable=False)  # Key characteristics of the pattern
    
    # Pattern discovery
    discovered_by = Column(String(50), nullable=False)  # ai_analysis, human_expert, hybrid
    discovery_method = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0
    
    # Pattern effectiveness
    success_rate = Column(Float, nullable=False)  # 0.0 - 1.0
    sample_size = Column(Integer, nullable=False)
    statistical_significance = Column(Float, nullable=True)
    
    # Context applicability
    applicable_scenarios = Column(JSON, nullable=True)
    industry_applicability = Column(JSON, nullable=True)
    customer_segment = Column(JSON, nullable=True)
    conversation_stage = Column(JSON, nullable=True)
    
    # Performance metrics
    avg_conversation_quality = Column(Float, nullable=True)
    avg_customer_satisfaction = Column(Float, nullable=True)
    avg_resolution_time_minutes = Column(Float, nullable=True)
    conversion_rate = Column(Float, nullable=True)
    
    # Pattern examples
    example_conversations = Column(JSON, nullable=True)  # IDs of example conversations
    positive_examples = Column(JSON, nullable=True)
    negative_examples = Column(JSON, nullable=True)
    
    # Usage tracking
    times_recommended = Column(Integer, nullable=False, default=0)
    times_used = Column(Integer, nullable=False, default=0)
    success_when_used = Column(Integer, nullable=False, default=0)
    
    # Pattern lifecycle
    status = Column(String(30), nullable=False, default='active')  # active, deprecated, experimental
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_validated = Column(DateTime, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    validation_results = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_pattern_type_success', 'pattern_type', 'success_rate'),
        Index('idx_category_confidence', 'pattern_category', 'confidence_score'),
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_applicability', 'applicable_scenarios'),
    )

class AiRecommendation(Base):
    """AI coaching recommendations for improving conversations"""
    __tablename__ = "kelly_ai_recommendations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Recommendation context
    conversation_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    
    # Recommendation details
    recommendation_type = Column(String(50), nullable=False, index=True)  # coaching, pattern, strategy, improvement
    category = Column(String(50), nullable=False)  # response_time, empathy, accuracy, efficiency
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Recommendation specifics
    specific_suggestion = Column(Text, nullable=False)
    example_implementation = Column(Text, nullable=True)
    alternative_approaches = Column(JSON, nullable=True)
    
    # AI analysis
    based_on_analysis = Column(JSON, nullable=False)  # What analysis led to this recommendation
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0
    ai_model_used = Column(String(100), nullable=False)
    
    # Expected impact
    expected_improvement = Column(Float, nullable=True)  # Expected improvement percentage
    impact_area = Column(String(100), nullable=False)  # quality, efficiency, satisfaction, conversion
    expected_outcome = Column(Text, nullable=True)
    
    # Recommendation priority
    priority = Column(String(20), nullable=False, default='medium')  # low, medium, high, critical
    urgency = Column(String(20), nullable=False, default='normal')
    difficulty_level = Column(String(20), nullable=False, default='medium')  # easy, medium, hard
    
    # Context and timing
    trigger_event = Column(String(100), nullable=True)  # What triggered this recommendation
    optimal_timing = Column(String(100), nullable=True)
    context_requirements = Column(JSON, nullable=True)
    
    # Status and tracking
    status = Column(String(30), nullable=False, default='pending')  # pending, viewed, implemented, dismissed
    viewed_at = Column(DateTime, nullable=True)
    implemented_at = Column(DateTime, nullable=True)
    dismissed_at = Column(DateTime, nullable=True)
    dismissal_reason = Column(String(200), nullable=True)
    
    # Effectiveness tracking
    implementation_success = Column(Boolean, nullable=True)
    actual_improvement = Column(Float, nullable=True)
    feedback_rating = Column(Integer, nullable=True)  # 1-5 rating
    feedback_comments = Column(Text, nullable=True)
    
    # Performance tracking
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    supporting_data = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_user_generated', 'user_id', 'generated_at'),
        Index('idx_conversation_type', 'conversation_id', 'recommendation_type'),
        Index('idx_priority_urgency', 'priority', 'urgency'),
        Index('idx_status_category', 'status', 'category'),
        Index('idx_confidence_impact', 'confidence_score', 'impact_area'),
    )

class AnomalyDetection(Base):
    """Detected anomalies in conversations and behavior patterns"""
    __tablename__ = "kelly_anomaly_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Anomaly context
    conversation_id = Column(String(100), nullable=True, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=True, index=True)
    
    # Anomaly classification
    anomaly_type = Column(String(50), nullable=False, index=True)  # behavior, performance, sentiment, pattern
    anomaly_category = Column(String(50), nullable=False)  # positive, negative, neutral, unknown
    severity = Column(String(20), nullable=False, default='medium')  # low, medium, high, critical
    
    # Anomaly details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    
    # Detection details
    detection_method = Column(String(100), nullable=False)  # statistical, ml_model, rule_based
    detection_model = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0
    anomaly_score = Column(Float, nullable=False)  # How anomalous (higher = more anomalous)
    
    # Expected vs actual
    expected_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=False)
    deviation_magnitude = Column(Float, nullable=False)
    statistical_significance = Column(Float, nullable=True)
    
    # Contextual information
    baseline_period = Column(JSON, nullable=True)  # Period used for baseline
    comparison_data = Column(JSON, nullable=True)
    contributing_factors = Column(JSON, nullable=True)
    
    # Impact assessment
    impact_level = Column(String(20), nullable=False, default='medium')
    potential_consequences = Column(JSON, nullable=True)
    affected_metrics = Column(JSON, nullable=True)
    
    # Root cause analysis
    potential_causes = Column(JSON, nullable=True)
    root_cause_identified = Column(Boolean, nullable=False, default=False)
    root_cause = Column(Text, nullable=True)
    root_cause_confidence = Column(Float, nullable=True)
    
    # Resolution tracking
    status = Column(String(30), nullable=False, default='new')  # new, investigating, resolved, false_positive
    investigated_by = Column(String(50), nullable=True)
    investigated_at = Column(DateTime, nullable=True)
    resolution = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Follow-up actions
    requires_action = Column(Boolean, nullable=False, default=True)
    recommended_actions = Column(JSON, nullable=True)
    actions_taken = Column(JSON, nullable=True)
    
    # Performance tracking
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    occurrence_time = Column(DateTime, nullable=False)  # When the anomaly actually occurred
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    detection_data = Column(JSON, nullable=True)  # Raw detection data
    
    # Performance indexes
    __table_args__ = (
        Index('idx_account_detected', 'account_id', 'detected_at'),
        Index('idx_anomaly_type_severity', 'anomaly_type', 'severity'),
        Index('idx_status_impact', 'status', 'impact_level'),
        Index('idx_confidence_score', 'confidence_score'),
        Index('idx_occurrence_time', 'occurrence_time'),
    )

class TopicAnalysis(Base):
    """Conversation topic analysis and trends"""
    __tablename__ = "kelly_topic_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Topic context
    conversation_id = Column(String(100), nullable=True, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=True, index=True)
    
    # Topic identification
    topic_name = Column(String(200), nullable=False, index=True)
    topic_category = Column(String(100), nullable=False, index=True)
    subtopics = Column(JSON, nullable=True)
    
    # Detection details
    detection_confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    detection_method = Column(String(50), nullable=False)  # nlp_model, keyword_matching, semantic_analysis
    model_used = Column(String(100), nullable=True)
    
    # Topic characteristics
    sentiment = Column(String(20), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)
    emotion = Column(String(50), nullable=True)
    urgency_level = Column(String(20), nullable=False, default='normal')
    
    # Topic context
    mentioned_entities = Column(JSON, nullable=True)  # Named entities mentioned
    keywords = Column(JSON, nullable=True)
    related_topics = Column(JSON, nullable=True)
    
    # Frequency and patterns
    mention_count = Column(Integer, nullable=False, default=1)
    first_mentioned_at = Column(DateTime, nullable=False)
    last_mentioned_at = Column(DateTime, nullable=False)
    discussion_duration_minutes = Column(Float, nullable=True)
    
    # Business relevance
    business_relevance = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0
    product_relevance = Column(Float, nullable=False, default=0.5)
    sales_relevance = Column(Float, nullable=False, default=0.5)
    support_relevance = Column(Float, nullable=False, default=0.5)
    
    # Topic outcome
    resolution_status = Column(String(50), nullable=True)  # resolved, pending, escalated
    satisfaction_impact = Column(Float, nullable=True)
    follow_up_required = Column(Boolean, nullable=False, default=False)
    
    # Performance tracking
    analyzed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    conversation_timestamp = Column(DateTime, nullable=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    analysis_details = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_topic_name_analyzed', 'topic_name', 'analyzed_at'),
        Index('idx_account_category', 'account_id', 'topic_category'),
        Index('idx_business_relevance', 'business_relevance'),
        Index('idx_conversation_timestamp', 'conversation_timestamp'),
        Index('idx_sentiment_urgency', 'sentiment', 'urgency_level'),
    )

class IntelligenceReport(Base):
    """Aggregated intelligence reports and insights"""
    __tablename__ = "kelly_intelligence_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Report details
    report_name = Column(String(200), nullable=False)
    report_type = Column(String(50), nullable=False, index=True)  # daily, weekly, monthly, custom, incident
    report_category = Column(String(50), nullable=False)  # performance, patterns, anomalies, insights
    
    # Report scope
    account_id = Column(String(50), nullable=True, index=True)
    team_id = Column(String(50), nullable=True, index=True)
    global_report = Column(Boolean, nullable=False, default=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # hour, day, week, month, quarter
    
    # Report content
    executive_summary = Column(Text, nullable=False)
    key_insights = Column(JSON, nullable=False)
    recommendations = Column(JSON, nullable=False)
    
    # Metrics summary
    total_conversations = Column(Integer, nullable=False, default=0)
    insights_generated = Column(Integer, nullable=False, default=0)
    patterns_discovered = Column(Integer, nullable=False, default=0)
    anomalies_detected = Column(Integer, nullable=False, default=0)
    
    # Quality metrics
    avg_conversation_quality = Column(Float, nullable=True)
    customer_satisfaction_avg = Column(Float, nullable=True)
    resolution_rate = Column(Float, nullable=True)
    escalation_rate = Column(Float, nullable=True)
    
    # AI performance
    ai_accuracy_score = Column(Float, nullable=True)
    ai_confidence_avg = Column(Float, nullable=True)
    human_intervention_rate = Column(Float, nullable=True)
    
    # Report generation
    generated_by = Column(String(50), nullable=False)  # ai_system, human_analyst, scheduled
    generation_method = Column(String(100), nullable=False)
    data_sources = Column(JSON, nullable=True)
    
    # Report status
    status = Column(String(30), nullable=False, default='draft')  # draft, published, archived
    published_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    
    # Access control
    visibility = Column(String(30), nullable=False, default='private')  # private, team, organization
    shared_with = Column(JSON, nullable=True)
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    detailed_data = Column(JSON, nullable=True)
    attachments = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_report_type_period', 'report_type', 'period_start'),
        Index('idx_account_created', 'account_id', 'created_at'),
        Index('idx_status_published', 'status', 'published_at'),
        Index('idx_global_report', 'global_report', 'period_start'),
    )

def create_kelly_intelligence_tables():
    """Create all Kelly intelligence tables"""
    from app.database.connection import engine
    Base.metadata.create_all(bind=engine)