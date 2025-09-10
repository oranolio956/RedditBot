"""
Audit Models

Defines models for comprehensive audit logging, security monitoring,
and performance tracking for compliance and debugging purposes.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from ipaddress import ip_address

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text, JSON, ForeignKey,
    Index, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY, INET
from sqlalchemy.sql import func

from app.database.base import BaseModel


class AuditEventType(str, Enum):
    """Audit event type enumeration."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_EVENT = "error_event"
    PERFORMANCE_EVENT = "performance_event"
    SECURITY_EVENT = "security_event"


class SecurityEventType(str, Enum):
    """Security event type enumeration."""
    FAILED_LOGIN = "failed_login"
    SUCCESSFUL_LOGIN = "successful_login"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_INPUT = "malicious_input"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ACCOUNT_LOCKED = "account_locked"
    PASSWORD_RESET = "password_reset"


class SeverityLevel(str, Enum):
    """Event severity level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(BaseModel):
    """
    Comprehensive audit log for all system activities.
    
    Tracks user actions, system events, data access, and security incidents
    for compliance, debugging, and security monitoring.
    """
    
    __tablename__ = "audit_logs"
    
    # Event identification
    event_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique event identifier"
    )
    
    event_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of event being audited"
    )
    
    event_category = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Event category for grouping"
    )
    
    severity = Column(
        String(20),
        default=SeverityLevel.INFO,
        nullable=False,
        index=True,
        comment="Event severity level"
    )
    
    # Event details
    action = Column(
        String(100),
        nullable=False,
        comment="Specific action performed"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of the event"
    )
    
    resource_type = Column(
        String(50),
        nullable=True,
        comment="Type of resource affected"
    )
    
    resource_id = Column(
        String(100),
        nullable=True,
        index=True,
        comment="ID of resource affected"
    )
    
    # User and session context
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who initiated the event"
    )
    
    session_id = Column(
        String(128),
        nullable=True,
        index=True,
        comment="Session ID when event occurred"
    )
    
    # Request context
    ip_address = Column(
        INET,
        nullable=True,
        index=True,
        comment="IP address of the client"
    )
    
    user_agent = Column(
        String(500),
        nullable=True,
        comment="User agent string"
    )
    
    request_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="Request ID for correlation"
    )
    
    endpoint = Column(
        String(200),
        nullable=True,
        comment="API endpoint accessed"
    )
    
    http_method = Column(
        String(10),
        nullable=True,
        comment="HTTP method used"
    )
    
    # Event data and context
    event_data = Column(
        JSONB,
        nullable=True,
        comment="Additional event data and context"
    )
    
    before_values = Column(
        JSONB,
        nullable=True,
        comment="Values before modification (for data changes)"
    )
    
    after_values = Column(
        JSONB,
        nullable=True,
        comment="Values after modification (for data changes)"
    )
    
    audit_metadata = Column(
        JSONB,
        nullable=True,
        comment="Additional metadata about the event"
    )
    
    # Outcome and status
    success = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the action was successful"
    )
    
    error_code = Column(
        String(50),
        nullable=True,
        comment="Error code if action failed"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if action failed"
    )
    
    # Timing information
    timestamp = Column(
        "timestamp",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When the event occurred"
    )
    
    processing_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken to process the action"
    )
    
    # Compliance and retention
    retention_period_days = Column(
        Integer,
        default=2555,  # 7 years default
        nullable=False,
        comment="How long to retain this audit record"
    )
    
    is_pii_data = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this record contains PII"
    )
    
    compliance_flags = Column(
        ARRAY(String),
        nullable=True,
        comment="Compliance framework flags (GDPR, HIPAA, etc.)"
    )
    
    # Investigation and follow-up
    requires_investigation = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this event requires investigation"
    )
    
    investigation_status = Column(
        String(20),
        nullable=True,
        comment="Status of any investigation"
    )
    
    investigation_notes = Column(
        Text,
        nullable=True,
        comment="Investigation notes and findings"
    )
    
    # Relationships
    user = relationship("User", backref="audit_logs")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_event_type', 'event_type', 'event_category'),
        Index('idx_audit_severity', 'severity'),
        Index('idx_audit_success', 'success'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_ip', 'ip_address'),
        Index('idx_audit_investigation', 'requires_investigation'),
        CheckConstraint('processing_time_ms >= 0', name='ck_processing_time_positive'),
        CheckConstraint('retention_period_days > 0', name='ck_retention_positive'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-generate event ID if not provided
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())
    
    def set_before_after_values(self, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        """Set before and after values for data modification events."""
        self.before_values = before
        self.after_values = after
        
        # Calculate what changed
        changes = {}
        for key in set(before.keys()) | set(after.keys()):
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val != after_val:
                changes[key] = {
                    "from": before_val,
                    "to": after_val
                }
        
        if not self.event_data:
            self.event_data = {}
        self.event_data["changes"] = changes
    
    def add_context(self, key: str, value: Any) -> None:
        """Add contextual information to the event."""
        if not self.event_data:
            self.event_data = {}
        self.event_data[key] = value
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'event_data')
    
    def mark_for_investigation(self, reason: str) -> None:
        """Mark this event for investigation."""
        self.requires_investigation = True
        self.investigation_status = "pending"
        self.add_context("investigation_reason", reason)
    
    def is_retention_expired(self) -> bool:
        """Check if retention period has expired."""
        expiry_date = self.timestamp + timedelta(days=self.retention_period_days)
        return datetime.utcnow() > expiry_date
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit event summary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "action": self.action,
            "user_id": str(self.user_id) if self.user_id else None,
            "resource": f"{self.resource_type}:{self.resource_id}" if self.resource_type else None,
            "success": self.success,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "ip_address": str(self.ip_address) if self.ip_address else None,
            "requires_investigation": self.requires_investigation,
        }


class SecurityEvent(BaseModel):
    """
    Security-specific event tracking with enhanced monitoring capabilities.
    
    Specialized tracking for security incidents, threats, and
    anomalous behavior patterns.
    """
    
    __tablename__ = "security_events"
    
    # Event identification
    event_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique security event identifier"
    )
    
    event_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of security event"
    )
    
    threat_level = Column(
        String(20),
        default=SeverityLevel.INFO,
        nullable=False,
        index=True,
        comment="Threat level assessment"
    )
    
    # Event details
    title = Column(
        String(200),
        nullable=False,
        comment="Security event title"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of the security event"
    )
    
    attack_vector = Column(
        String(100),
        nullable=True,
        comment="Identified attack vector"
    )
    
    # Source information
    source_ip = Column(
        INET,
        nullable=True,
        index=True,
        comment="Source IP address of the event"
    )
    
    source_user_agent = Column(
        String(500),
        nullable=True,
        comment="User agent of the source"
    )
    
    source_location = Column(
        JSONB,
        nullable=True,
        comment="Geographical location of source (if available)"
    )
    
    # Target information
    target_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Target user (if applicable)"
    )
    
    target_resource = Column(
        String(200),
        nullable=True,
        comment="Target resource or endpoint"
    )
    
    target_data = Column(
        JSONB,
        nullable=True,
        comment="Information about targeted data"
    )
    
    # Attack characteristics
    attack_pattern = Column(
        JSONB,
        nullable=True,
        comment="Identified attack patterns and signatures"
    )
    
    payload_data = Column(
        Text,
        nullable=True,
        comment="Attack payload (sanitized for analysis)"
    )
    
    indicators_of_compromise = Column(
        ARRAY(String),
        nullable=True,
        comment="Indicators of compromise (IOCs)"
    )
    
    # Detection and response
    detection_method = Column(
        String(100),
        nullable=False,
        comment="How this event was detected"
    )
    
    detection_confidence = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Confidence in detection (0-1)"
    )
    
    automated_response = Column(
        JSONB,
        nullable=True,
        comment="Automated response actions taken"
    )
    
    manual_response = Column(
        JSONB,
        nullable=True,
        comment="Manual response actions taken"
    )
    
    # Impact assessment
    impact_score = Column(
        Float,
        nullable=True,
        comment="Impact assessment score (0-10)"
    )
    
    affected_systems = Column(
        ARRAY(String),
        nullable=True,
        comment="Systems affected by this event"
    )
    
    data_compromised = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether data was compromised"
    )
    
    service_disrupted = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether service was disrupted"
    )
    
    # Timing and correlation
    first_seen = Column(
        "first_seen",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When this event was first detected"
    )
    
    last_seen = Column(
        "last_seen",
        nullable=True,
        index=True,
        comment="When this event was last seen"
    )
    
    event_count = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Number of times this event has occurred"
    )
    
    correlation_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="ID for correlating related events"
    )
    
    # Status and workflow
    status = Column(
        String(20),
        default="open",
        nullable=False,
        index=True,
        comment="Event status (open, investigating, resolved, false_positive)"
    )
    
    assigned_to = Column(
        String(100),
        nullable=True,
        comment="Security analyst assigned to this event"
    )
    
    resolution_notes = Column(
        Text,
        nullable=True,
        comment="Notes on event resolution"
    )
    
    false_positive = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this was determined to be a false positive"
    )
    
    # Notification and alerting
    alert_sent = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether an alert was sent for this event"
    )
    
    notification_channels = Column(
        ARRAY(String),
        nullable=True,
        comment="Channels where notifications were sent"
    )
    
    # Relationships
    target_user = relationship("User", backref="security_events_targeted")
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_security_first_seen', 'first_seen'),
        Index('idx_security_threat_level', 'threat_level'),
        Index('idx_security_event_type', 'event_type'),
        Index('idx_security_source_ip', 'source_ip'),
        Index('idx_security_status', 'status'),
        Index('idx_security_correlation', 'correlation_id'),
        Index('idx_security_target_user', 'target_user_id'),
        CheckConstraint('detection_confidence >= 0 AND detection_confidence <= 1', name='ck_detection_confidence_range'),
        CheckConstraint('impact_score >= 0 AND impact_score <= 10', name='ck_impact_score_range'),
        CheckConstraint('event_count > 0', name='ck_event_count_positive'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-generate event ID if not provided
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())
    
    def update_occurrence(self) -> None:
        """Update event for repeated occurrences."""
        self.event_count += 1
        self.last_seen = datetime.utcnow()
    
    def escalate_threat_level(self, new_level: str, reason: str) -> None:
        """Escalate threat level with reason."""
        old_level = self.threat_level
        self.threat_level = new_level
        
        if not self.manual_response:
            self.manual_response = []
        
        self.manual_response.append({
            "action": "threat_level_escalation",
            "timestamp": datetime.utcnow().isoformat(),
            "from_level": old_level,
            "to_level": new_level,
            "reason": reason
        })
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'manual_response')
    
    def add_ioc(self, ioc_value: str, ioc_type: str) -> None:
        """Add an indicator of compromise."""
        if not self.indicators_of_compromise:
            self.indicators_of_compromise = []
        
        ioc_entry = f"{ioc_type}:{ioc_value}"
        if ioc_entry not in self.indicators_of_compromise:
            self.indicators_of_compromise.append(ioc_entry)
            
            # Mark field as modified
            from sqlalchemy.orm import attributes
            attributes.flag_modified(self, 'indicators_of_compromise')
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security event summary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "threat_level": self.threat_level,
            "title": self.title,
            "source_ip": str(self.source_ip) if self.source_ip else None,
            "target_user": str(self.target_user_id) if self.target_user_id else None,
            "impact_score": self.impact_score,
            "status": self.status,
            "event_count": self.event_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "data_compromised": self.data_compromised,
            "service_disrupted": self.service_disrupted,
        }


class PerformanceMetric(BaseModel):
    """
    Performance metrics tracking for system monitoring and optimization.
    
    Tracks response times, throughput, resource utilization, and
    other performance indicators.
    """
    
    __tablename__ = "performance_metrics"
    
    # Metric identification
    metric_name = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Name of the performance metric"
    )
    
    metric_category = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Category of metric (response_time, throughput, etc.)"
    )
    
    component = Column(
        String(100),
        nullable=False,
        index=True,
        comment="System component being measured"
    )
    
    # Metric values
    value = Column(
        Float,
        nullable=False,
        comment="Metric value"
    )
    
    unit = Column(
        String(20),
        nullable=False,
        comment="Unit of measurement (ms, req/s, MB, etc.)"
    )
    
    # Statistical data
    min_value = Column(
        Float,
        nullable=True,
        comment="Minimum value in measurement period"
    )
    
    max_value = Column(
        Float,
        nullable=True,
        comment="Maximum value in measurement period"
    )
    
    avg_value = Column(
        Float,
        nullable=True,
        comment="Average value in measurement period"
    )
    
    percentile_50 = Column(
        Float,
        nullable=True,
        comment="50th percentile value"
    )
    
    percentile_95 = Column(
        Float,
        nullable=True,
        comment="95th percentile value"
    )
    
    percentile_99 = Column(
        Float,
        nullable=True,
        comment="99th percentile value"
    )
    
    sample_count = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Number of samples in this measurement"
    )
    
    # Context and dimensions
    dimensions = Column(
        JSONB,
        nullable=True,
        comment="Metric dimensions (endpoint, user_type, etc.)"
    )
    
    tags = Column(
        ARRAY(String),
        nullable=True,
        comment="Metric tags for filtering and grouping"
    )
    
    # Threshold monitoring
    warning_threshold = Column(
        Float,
        nullable=True,
        comment="Warning threshold for this metric"
    )
    
    critical_threshold = Column(
        Float,
        nullable=True,
        comment="Critical threshold for this metric"
    )
    
    threshold_breached = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether a threshold was breached"
    )
    
    # Timing
    timestamp = Column(
        "timestamp",
        nullable=False,
        server_default=func.now(),
        index=True,
        comment="When the metric was recorded"
    )
    
    measurement_period_start = Column(
        "measurement_period_start",
        nullable=True,
        comment="Start of measurement period"
    )
    
    measurement_period_end = Column(
        "measurement_period_end",
        nullable=True,
        comment="End of measurement period"
    )
    
    # Additional metadata
    environment = Column(
        String(20),
        default="production",
        nullable=False,
        comment="Environment where metric was collected"
    )
    
    version = Column(
        String(20),
        nullable=True,
        comment="Application version when metric was collected"
    )
    
    # Database constraints and indexes
    __table_args__ = (
        Index('idx_perf_metric_name', 'metric_name', 'timestamp'),
        Index('idx_perf_component', 'component', 'timestamp'),
        Index('idx_perf_category', 'metric_category'),
        Index('idx_perf_threshold', 'threshold_breached'),
        Index('idx_perf_environment', 'environment'),
        CheckConstraint('sample_count > 0', name='ck_sample_count_positive'),
    )
    
    def check_thresholds(self) -> bool:
        """Check if metric value breaches any thresholds."""
        breached = False
        
        if self.critical_threshold and self.value >= self.critical_threshold:
            breached = True
        elif self.warning_threshold and self.value >= self.warning_threshold:
            breached = True
        
        self.threshold_breached = breached
        return breached
    
    def update_statistics(self, values: List[float]) -> None:
        """Update statistical values from a list of measurements."""
        if not values:
            return
        
        self.sample_count = len(values)
        self.min_value = min(values)
        self.max_value = max(values)
        self.avg_value = sum(values) / len(values)
        
        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        self.percentile_50 = sorted_values[int(n * 0.5)]
        self.percentile_95 = sorted_values[int(n * 0.95)]
        self.percentile_99 = sorted_values[int(n * 0.99)]
        
        # Use average as the main value
        self.value = self.avg_value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metric summary."""
        return {
            "metric_name": self.metric_name,
            "category": self.metric_category,
            "component": self.component,
            "value": self.value,
            "unit": self.unit,
            "avg_value": self.avg_value,
            "percentile_95": self.percentile_95,
            "sample_count": self.sample_count,
            "threshold_breached": self.threshold_breached,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "dimensions": self.dimensions,
        }