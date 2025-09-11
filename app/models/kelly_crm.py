"""
Kelly AI CRM Database Models

SQLAlchemy models for customer relationship management including contacts,
leads, deals, pipeline management, and customer journey tracking.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database.connection import Base

class Contact(Base):
    """Enhanced contact profiles with AI-driven insights"""
    __tablename__ = "kelly_contacts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic contact information
    user_id = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(200), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    display_name = Column(String(200), nullable=True)
    
    # Platform identifiers
    telegram_id = Column(String(50), nullable=True, index=True)
    telegram_username = Column(String(100), nullable=True)
    whatsapp_id = Column(String(50), nullable=True)
    discord_id = Column(String(50), nullable=True)
    other_platform_ids = Column(JSON, nullable=True)
    
    # Contact details
    company = Column(String(200), nullable=True, index=True)
    job_title = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True, index=True)
    company_size = Column(String(50), nullable=True)  # startup, small, medium, large, enterprise
    
    # Location
    country = Column(String(100), nullable=True, index=True)
    state_region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    timezone = Column(String(50), nullable=True)
    
    # Contact status
    status = Column(String(50), nullable=False, default='active')  # active, inactive, blocked, archived
    contact_source = Column(String(100), nullable=True)  # organic, referral, campaign, etc.
    acquisition_channel = Column(String(100), nullable=True)
    
    # Relationship metrics
    relationship_strength = Column(Float, nullable=False, default=0.0)  # 0.0 - 1.0
    engagement_score = Column(Float, nullable=False, default=0.0)
    trust_level = Column(Float, nullable=False, default=0.5)
    satisfaction_score = Column(Float, nullable=True)
    
    # Communication preferences
    preferred_communication_channel = Column(String(50), nullable=True)
    communication_frequency = Column(String(50), nullable=True)  # daily, weekly, monthly, quarterly
    best_contact_time = Column(String(100), nullable=True)
    do_not_contact = Column(Boolean, nullable=False, default=False)
    
    # AI-driven insights
    personality_profile = Column(JSON, nullable=True)  # AI-derived personality traits
    communication_style = Column(String(100), nullable=True)
    interests = Column(JSON, nullable=True)  # Detected interests from conversations
    pain_points = Column(JSON, nullable=True)  # Identified pain points
    motivators = Column(JSON, nullable=True)  # What motivates this contact
    
    # Behavioral patterns
    avg_response_time_minutes = Column(Float, nullable=True)
    preferred_conversation_length = Column(String(50), nullable=True)  # short, medium, long
    conversation_style = Column(String(100), nullable=True)
    decision_making_style = Column(String(100), nullable=True)
    
    # Business context
    budget_range = Column(String(100), nullable=True)
    authority_level = Column(String(50), nullable=True)  # decision_maker, influencer, user, blocker
    need_urgency = Column(String(50), nullable=True)  # low, medium, high, urgent
    timeline = Column(String(100), nullable=True)
    
    # Lifecycle tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_contact_at = Column(DateTime, nullable=True, index=True)
    last_activity_at = Column(DateTime, nullable=True, index=True)
    
    # GDPR and privacy
    gdpr_consent = Column(Boolean, nullable=False, default=False)
    gdpr_consent_date = Column(DateTime, nullable=True)
    data_processing_consent = Column(Boolean, nullable=False, default=False)
    marketing_consent = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_company_industry', 'company', 'industry'),
        Index('idx_status_last_contact', 'status', 'last_contact_at'),
        Index('idx_engagement_score', 'engagement_score'),
        Index('idx_relationship_strength', 'relationship_strength'),
        Index('idx_authority_urgency', 'authority_level', 'need_urgency'),
    )

class LeadScore(Base):
    """AI-powered lead scoring and qualification"""
    __tablename__ = "kelly_lead_scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contact_id = Column(UUID(as_uuid=True), ForeignKey('kelly_contacts.id'), nullable=False, index=True)
    
    # Overall lead score
    overall_score = Column(Float, nullable=False)  # 0.0 - 100.0
    score_grade = Column(String(10), nullable=False)  # A+, A, B+, B, C+, C, D, F
    qualification_status = Column(String(50), nullable=False)  # qualified, unqualified, nurturing, investigating
    
    # Score components
    demographic_score = Column(Float, nullable=False, default=0.0)
    behavioral_score = Column(Float, nullable=False, default=0.0)
    engagement_score = Column(Float, nullable=False, default=0.0)
    fit_score = Column(Float, nullable=False, default=0.0)
    intent_score = Column(Float, nullable=False, default=0.0)
    timing_score = Column(Float, nullable=False, default=0.0)
    
    # BANT criteria
    budget_score = Column(Float, nullable=False, default=0.0)
    authority_score = Column(Float, nullable=False, default=0.0)
    need_score = Column(Float, nullable=False, default=0.0)
    timeline_score = Column(Float, nullable=False, default=0.0)
    
    # AI confidence and reasoning
    scoring_confidence = Column(Float, nullable=False, default=1.0)
    scoring_method = Column(String(50), nullable=False)  # ai_model, rule_based, hybrid
    model_version = Column(String(50), nullable=True)
    
    # Conversion prediction
    conversion_probability = Column(Float, nullable=True)  # 0.0 - 1.0
    predicted_deal_size = Column(Float, nullable=True)
    predicted_close_date = Column(DateTime, nullable=True)
    predicted_timeline_days = Column(Integer, nullable=True)
    
    # Historical tracking
    score_history = Column(JSON, nullable=True)  # Historical score changes
    previous_score = Column(Float, nullable=True)
    score_change = Column(Float, nullable=True)
    score_trend = Column(String(20), nullable=True)  # improving, declining, stable
    
    # Factors and triggers
    positive_indicators = Column(JSON, nullable=True)
    negative_indicators = Column(JSON, nullable=True)
    trigger_events = Column(JSON, nullable=True)
    risk_factors = Column(JSON, nullable=True)
    
    # Performance tracking
    scored_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    valid_until = Column(DateTime, nullable=True)
    recalculate_requested = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    scoring_details = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Relationships
    contact = relationship("Contact", backref="lead_scores")
    
    # Performance indexes
    __table_args__ = (
        Index('idx_overall_score', 'overall_score'),
        Index('idx_qualification_status', 'qualification_status', 'scored_at'),
        Index('idx_conversion_probability', 'conversion_probability'),
        Index('idx_score_grade', 'score_grade', 'scored_at'),
    )

class Deal(Base):
    """Pipeline and opportunity management"""
    __tablename__ = "kelly_deals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deal_name = Column(String(200), nullable=False)
    
    # Deal relationships
    contact_id = Column(UUID(as_uuid=True), ForeignKey('kelly_contacts.id'), nullable=False, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    
    # Deal details
    deal_value = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False, default='USD')
    probability = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0
    expected_close_date = Column(DateTime, nullable=False, index=True)
    
    # Pipeline stage
    stage = Column(String(100), nullable=False, index=True)
    stage_probability = Column(Float, nullable=True)
    stage_entered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    previous_stage = Column(String(100), nullable=True)
    
    # Deal source and attribution
    source = Column(String(100), nullable=True)  # conversation, referral, campaign, etc.
    source_conversation_id = Column(String(100), nullable=True, index=True)
    attribution_type = Column(String(50), nullable=True)  # direct, assisted, influenced
    ai_assisted = Column(Boolean, nullable=False, default=False)
    ai_contribution_score = Column(Float, nullable=True)
    
    # Products and services
    product_categories = Column(JSON, nullable=True)
    products_interested = Column(JSON, nullable=True)
    solution_type = Column(String(100), nullable=True)
    
    # Competitive intelligence
    competitors = Column(JSON, nullable=True)
    competitive_advantages = Column(JSON, nullable=True)
    decision_criteria = Column(JSON, nullable=True)
    
    # Deal progress
    activities_completed = Column(JSON, nullable=True)
    next_steps = Column(JSON, nullable=True)
    obstacles = Column(JSON, nullable=True)
    success_factors = Column(JSON, nullable=True)
    
    # Risk assessment
    risk_level = Column(String(20), nullable=False, default='medium')  # low, medium, high
    risk_factors = Column(JSON, nullable=True)
    churn_risk = Column(Float, nullable=True)  # 0.0 - 1.0
    
    # Deal team
    owner_id = Column(String(50), nullable=False, index=True)
    team_members = Column(JSON, nullable=True)
    stakeholders = Column(JSON, nullable=True)
    
    # Performance tracking
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    # Deal outcome
    status = Column(String(30), nullable=False, default='open')  # open, won, lost, on_hold
    close_reason = Column(String(100), nullable=True)
    actual_close_date = Column(DateTime, nullable=True)
    actual_deal_value = Column(Float, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Relationships
    contact = relationship("Contact", backref="deals")
    
    # Performance indexes
    __table_args__ = (
        Index('idx_stage_expected_close', 'stage', 'expected_close_date'),
        Index('idx_owner_status', 'owner_id', 'status'),
        Index('idx_deal_value', 'deal_value'),
        Index('idx_probability', 'probability'),
        Index('idx_source_conversation', 'source_conversation_id'),
        Index('idx_ai_assisted', 'ai_assisted', 'created_at'),
    )

class TouchPoint(Base):
    """Customer journey touchpoints and interactions"""
    __tablename__ = "kelly_touchpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Touchpoint relationships
    contact_id = Column(UUID(as_uuid=True), ForeignKey('kelly_contacts.id'), nullable=False, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    deal_id = Column(UUID(as_uuid=True), ForeignKey('kelly_deals.id'), nullable=True, index=True)
    
    # Touchpoint details
    touchpoint_type = Column(String(50), nullable=False, index=True)  # conversation, email, call, meeting, demo
    channel = Column(String(50), nullable=False, index=True)  # telegram, whatsapp, discord, email, phone
    direction = Column(String(20), nullable=False)  # inbound, outbound
    
    # Content and context
    subject = Column(String(200), nullable=True)
    summary = Column(Text, nullable=True)
    sentiment = Column(String(20), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)
    
    # Journey context
    journey_stage = Column(String(50), nullable=False, index=True)  # awareness, consideration, decision, retention
    customer_lifecycle_stage = Column(String(50), nullable=True)
    intent_detected = Column(String(100), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    
    # Engagement metrics
    engagement_quality = Column(Float, nullable=False, default=0.0)  # 0.0 - 1.0
    response_time_minutes = Column(Float, nullable=True)
    interaction_duration_minutes = Column(Float, nullable=True)
    follow_up_required = Column(Boolean, nullable=False, default=False)
    
    # AI involvement
    ai_handled = Column(Boolean, nullable=False, default=False)
    ai_confidence = Column(Float, nullable=True)
    human_intervention = Column(Boolean, nullable=False, default=False)
    
    # Outcome and impact
    outcome = Column(String(100), nullable=True)
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5
    issue_resolved = Column(Boolean, nullable=True)
    escalated = Column(Boolean, nullable=False, default=False)
    
    # Business impact
    influenced_deal = Column(Boolean, nullable=False, default=False)
    progression_indicator = Column(Boolean, nullable=False, default=False)
    regression_indicator = Column(Boolean, nullable=False, default=False)
    
    # Performance tracking
    occurred_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    conversation_data = Column(JSON, nullable=True)
    
    # Relationships
    contact = relationship("Contact", backref="touchpoints")
    deal = relationship("Deal", backref="touchpoints")
    
    # Performance indexes
    __table_args__ = (
        Index('idx_contact_occurred', 'contact_id', 'occurred_at'),
        Index('idx_journey_stage_occurred', 'journey_stage', 'occurred_at'),
        Index('idx_touchpoint_type_occurred', 'touchpoint_type', 'occurred_at'),
        Index('idx_ai_handled', 'ai_handled', 'occurred_at'),
        Index('idx_deal_touchpoints', 'deal_id', 'occurred_at'),
    )

class CustomerJourney(Base):
    """Complete customer journey mapping and analysis"""
    __tablename__ = "kelly_customer_journeys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contact_id = Column(UUID(as_uuid=True), ForeignKey('kelly_contacts.id'), nullable=False, unique=True, index=True)
    
    # Journey overview
    journey_status = Column(String(50), nullable=False, default='active')  # active, completed, abandoned, on_hold
    current_stage = Column(String(50), nullable=False, index=True)
    previous_stage = Column(String(50), nullable=True)
    stage_entered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Journey timeline
    journey_started_at = Column(DateTime, nullable=False, index=True)
    journey_updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    expected_completion_date = Column(DateTime, nullable=True)
    actual_completion_date = Column(DateTime, nullable=True)
    
    # Journey metrics
    total_touchpoints = Column(Integer, nullable=False, default=0)
    conversion_touchpoints = Column(Integer, nullable=False, default=0)
    avg_time_between_touchpoints_hours = Column(Float, nullable=True)
    longest_gap_hours = Column(Float, nullable=True)
    
    # Stage progression
    awareness_entered_at = Column(DateTime, nullable=True)
    consideration_entered_at = Column(DateTime, nullable=True)
    decision_entered_at = Column(DateTime, nullable=True)
    retention_entered_at = Column(DateTime, nullable=True)
    
    # Stage durations (in hours)
    awareness_duration_hours = Column(Float, nullable=True)
    consideration_duration_hours = Column(Float, nullable=True)
    decision_duration_hours = Column(Float, nullable=True)
    retention_duration_hours = Column(Float, nullable=True)
    
    # Journey patterns
    journey_pattern = Column(String(100), nullable=True)  # linear, cyclical, complex, abandoned
    pattern_confidence = Column(Float, nullable=True)
    similar_journeys_count = Column(Integer, nullable=True)
    
    # Conversion tracking
    converted = Column(Boolean, nullable=False, default=False)
    conversion_date = Column(DateTime, nullable=True)
    conversion_value = Column(Float, nullable=True)
    conversion_type = Column(String(50), nullable=True)
    
    # Drop-off analysis
    dropped_off = Column(Boolean, nullable=False, default=False)
    drop_off_stage = Column(String(50), nullable=True)
    drop_off_date = Column(DateTime, nullable=True)
    drop_off_reason = Column(String(200), nullable=True)
    
    # Engagement quality
    overall_engagement_score = Column(Float, nullable=False, default=0.0)
    engagement_trend = Column(String(20), nullable=True)  # increasing, decreasing, stable
    satisfaction_trend = Column(String(20), nullable=True)
    
    # AI insights
    ai_recommendations = Column(JSON, nullable=True)
    next_best_actions = Column(JSON, nullable=True)
    risk_indicators = Column(JSON, nullable=True)
    success_indicators = Column(JSON, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    journey_data = Column(JSON, nullable=True)  # Detailed journey analytics
    
    # Relationships
    contact = relationship("Contact", backref="customer_journey", uselist=False)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_current_stage', 'current_stage', 'stage_entered_at'),
        Index('idx_journey_status', 'journey_status', 'journey_started_at'),
        Index('idx_converted', 'converted', 'conversion_date'),
        Index('idx_dropped_off', 'dropped_off', 'drop_off_date'),
    )

def create_kelly_crm_tables():
    """Create all Kelly CRM tables"""
    from app.database.connection import engine
    Base.metadata.create_all(bind=engine)