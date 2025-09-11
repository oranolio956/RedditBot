"""
Kelly AI Monitoring Database Models

SQLAlchemy models for Kelly AI monitoring system including metrics,
activities, alerts, interventions, and emergency actions.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database.connection import Base

class SystemMetric(Base):
    """Model for storing real-time system metrics"""
    __tablename__ = "kelly_system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # gauge, counter, histogram
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Optional grouping fields
    account_id = Column(String(50), nullable=True, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(50), nullable=True, index=True)
    
    # Metadata for additional context
    metadata = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_account_metric_timestamp', 'account_id', 'metric_name', 'timestamp'),
        Index('idx_conversation_metric_timestamp', 'conversation_id', 'metric_name', 'timestamp'),
    )

class ActivityEvent(Base):
    """Model for storing activity feed events"""
    __tablename__ = "kelly_activity_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(100), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Event context
    account_id = Column(String(50), nullable=True, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(50), nullable=True, index=True)
    
    # Event metadata
    severity = Column(String(20), nullable=False, default='low', index=True)  # low, medium, high, critical
    category = Column(String(50), nullable=False, index=True)  # safety, performance, conversation, system
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    occurred_at = Column(DateTime, nullable=True)  # When the actual event occurred (may differ from creation)
    
    # Status tracking
    read_by_users = Column(JSON, nullable=True)  # List of user IDs who have read this
    archived = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_event_type_created', 'event_type', 'created_at'),
        Index('idx_account_severity_created', 'account_id', 'severity', 'created_at'),
        Index('idx_category_created', 'category', 'created_at'),
    )

class AlertInstance(Base):
    """Model for storing alert instances"""
    __tablename__ = "kelly_alert_instances"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(100), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Alert context
    account_id = Column(String(50), nullable=True, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(50), nullable=True, index=True)
    
    # Alert classification
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    category = Column(String(50), nullable=False, index=True)  # safety, performance, security, system
    priority = Column(Integer, nullable=False, default=3)  # 1=urgent, 2=high, 3=normal, 4=low
    
    # Status tracking
    status = Column(String(30), nullable=False, default='active', index=True)  # active, acknowledged, investigating, resolved, escalated
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    escalated_at = Column(DateTime, nullable=True)
    
    # People involved
    acknowledged_by = Column(String(50), nullable=True)
    resolved_by = Column(String(50), nullable=True)
    escalated_by = Column(String(50), nullable=True)
    assigned_to = Column(String(50), nullable=True, index=True)
    
    # Resolution details
    resolution_type = Column(String(50), nullable=True)  # fixed, false_positive, deferred, escalated
    resolution_notes = Column(Text, nullable=True)
    escalation_reason = Column(Text, nullable=True)
    escalation_level = Column(String(30), nullable=True)  # supervisor, admin, emergency
    
    # Metrics
    impact_score = Column(Float, nullable=False, default=1.0)
    auto_generated = Column(Boolean, nullable=False, default=True)
    requires_human_review = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    alert_data = Column(JSON, nullable=True)  # Original alert trigger data
    
    # Performance indexes
    __table_args__ = (
        Index('idx_status_severity_created', 'status', 'severity', 'created_at'),
        Index('idx_account_status_created', 'account_id', 'status', 'created_at'),
        Index('idx_assigned_status', 'assigned_to', 'status'),
        Index('idx_category_severity', 'category', 'severity'),
    )

class InterventionLog(Base):
    """Model for storing intervention logs"""
    __tablename__ = "kelly_intervention_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    intervention_type = Column(String(50), nullable=False, index=True)  # takeover, release, emergency_stop
    
    # Intervention context
    conversation_id = Column(String(100), nullable=False, index=True)
    account_id = Column(String(50), nullable=False, index=True)
    target_user_id = Column(String(50), nullable=False, index=True)
    
    # Human operator details
    operator_id = Column(String(50), nullable=False, index=True)
    operator_name = Column(String(100), nullable=False)
    operator_role = Column(String(50), nullable=True)
    
    # Intervention details
    reason = Column(Text, nullable=False)
    priority = Column(String(20), nullable=False, default='normal')  # low, normal, high, urgent
    intervention_trigger = Column(String(100), nullable=True)  # safety_alert, performance_issue, manual
    
    # Timing
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # AI context at intervention
    ai_confidence_at_start = Column(Float, nullable=True)
    ai_confidence_at_end = Column(Float, nullable=True)
    safety_score_at_start = Column(Float, nullable=True)
    safety_score_at_end = Column(Float, nullable=True)
    
    # Intervention outcome
    status = Column(String(30), nullable=False, default='active')  # active, completed, aborted, escalated
    outcome = Column(String(50), nullable=True)  # resolved, escalated, transferred, user_blocked
    success_rating = Column(Integer, nullable=True)  # 1-5 rating of intervention success
    
    # Messages during intervention
    messages_handled = Column(Integer, nullable=False, default=0)
    ai_suggestions_used = Column(Integer, nullable=False, default=0)
    
    # Handoff details
    handoff_summary = Column(Text, nullable=True)
    ai_handoff_instructions = Column(Text, nullable=True)
    follow_up_required = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    conversation_context = Column(JSON, nullable=True)  # Conversation state at intervention
    
    # Performance indexes
    __table_args__ = (
        Index('idx_conversation_started', 'conversation_id', 'started_at'),
        Index('idx_operator_started', 'operator_id', 'started_at'),
        Index('idx_account_intervention_type', 'account_id', 'intervention_type'),
        Index('idx_status_priority', 'status', 'priority'),
    )

class EmergencyAction(Base):
    """Model for storing emergency actions and audit trail"""
    __tablename__ = "kelly_emergency_actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action_type = Column(String(100), nullable=False, index=True)
    action_category = Column(String(50), nullable=False, index=True)  # conversation, account, system
    
    # Action context
    target_type = Column(String(50), nullable=False)  # conversation, account, system_wide
    target_id = Column(String(100), nullable=True)  # ID of target being acted upon
    
    # Action details
    reason = Column(Text, nullable=False)
    urgency = Column(String(20), nullable=False, default='normal')  # low, normal, high, urgent
    confirmation_required = Column(Boolean, nullable=False, default=False)
    confirmation_code = Column(String(100), nullable=True)
    
    # Who performed the action
    performed_by = Column(String(50), nullable=False, index=True)
    performed_by_name = Column(String(100), nullable=False)
    performed_by_role = Column(String(50), nullable=True)
    authorized_by = Column(String(50), nullable=True)  # If different from performer
    
    # Timing
    initiated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Action status
    status = Column(String(30), nullable=False, default='initiated')  # initiated, in_progress, completed, failed, aborted
    success = Column(Boolean, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Impact tracking
    affected_conversations = Column(Integer, nullable=False, default=0)
    affected_accounts = Column(Integer, nullable=False, default=0)
    affected_users = Column(Integer, nullable=False, default=0)
    
    # Rollback capability
    rollback_available = Column(Boolean, nullable=False, default=False)
    rollback_data = Column(JSON, nullable=True)  # Data needed for rollback
    rollback_performed = Column(Boolean, nullable=False, default=False)
    rollback_at = Column(DateTime, nullable=True)
    rollback_by = Column(String(50), nullable=True)
    
    # Notifications
    notifications_sent = Column(JSON, nullable=True)  # List of who was notified
    escalation_triggered = Column(Boolean, nullable=False, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    before_state = Column(JSON, nullable=True)  # State before action
    after_state = Column(JSON, nullable=True)  # State after action
    
    # Performance indexes
    __table_args__ = (
        Index('idx_action_type_initiated', 'action_type', 'initiated_at'),
        Index('idx_performed_by_initiated', 'performed_by', 'initiated_at'),
        Index('idx_status_urgency', 'status', 'urgency'),
        Index('idx_target_type_target_id', 'target_type', 'target_id'),
    )

class MonitoringSession(Base):
    """Model for tracking monitoring sessions"""
    __tablename__ = "kelly_monitoring_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_type = Column(String(50), nullable=False, index=True)  # dashboard, alert_monitoring, intervention_monitoring
    
    # Session owner
    user_id = Column(String(50), nullable=False, index=True)
    username = Column(String(100), nullable=False)
    user_role = Column(String(50), nullable=True)
    
    # Session details
    connection_id = Column(String(100), nullable=False, index=True)
    websocket_session_id = Column(String(100), nullable=True)
    
    # Session configuration
    monitored_accounts = Column(JSON, nullable=True)  # List of account IDs being monitored
    alert_filters = Column(JSON, nullable=True)  # Alert filtering preferences
    metric_subscriptions = Column(JSON, nullable=True)  # Which metrics to receive
    notification_preferences = Column(JSON, nullable=True)  # How to handle notifications
    
    # Timing
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    ended_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration_seconds = Column(Integer, nullable=True)
    
    # Session metrics
    messages_received = Column(Integer, nullable=False, default=0)
    alerts_handled = Column(Integer, nullable=False, default=0)
    interventions_performed = Column(Integer, nullable=False, default=0)
    
    # Session status
    status = Column(String(30), nullable=False, default='active')  # active, ended, disconnected, expired
    disconnect_reason = Column(String(100), nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    session_notes = Column(Text, nullable=True)
    
    # Performance indexes
    __table_args__ = (
        Index('idx_user_started', 'user_id', 'started_at'),
        Index('idx_session_type_status', 'session_type', 'status'),
        Index('idx_connection_status', 'connection_id', 'status'),
    )

class PerformanceBenchmark(Base):
    """Model for storing performance benchmark data"""
    __tablename__ = "kelly_performance_benchmarks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    benchmark_type = Column(String(50), nullable=False, index=True)  # response_time, throughput, error_rate
    
    # Measurement context
    account_id = Column(String(50), nullable=True, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    component = Column(String(100), nullable=False, index=True)  # api, claude_ai, database, redis
    
    # Metrics
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)  # ms, rps, percent, count
    
    # Statistical data
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    p50_value = Column(Float, nullable=True)
    p95_value = Column(Float, nullable=True)
    p99_value = Column(Float, nullable=True)
    
    # Sample information
    sample_size = Column(Integer, nullable=False, default=1)
    measurement_period_seconds = Column(Integer, nullable=False, default=1)
    
    # Timing
    measured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    raw_data = Column(JSON, nullable=True)  # Raw measurement data if needed
    
    # Performance indexes
    __table_args__ = (
        Index('idx_benchmark_component_measured', 'benchmark_type', 'component', 'measured_at'),
        Index('idx_account_benchmark_measured', 'account_id', 'benchmark_type', 'measured_at'),
    )

# Create all tables
def create_kelly_monitoring_tables():
    """Create all Kelly monitoring tables"""
    from app.database.connection import engine
    Base.metadata.create_all(bind=engine)