"""
Telegram Account Management Models
Advanced SQLAlchemy models for single Telegram account management across multiple communities.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, JSON, Float, Text,
    ForeignKey, Enum as SQLEnum, BigInteger, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database.base import BaseModel


class AccountStatus(str, Enum):
    """Account status enumeration"""
    INACTIVE = "inactive"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    LIMITED = "limited"
    SUSPENDED = "suspended"
    BANNED = "banned"


class SafetyLevel(str, Enum):
    """Safety level for account operations"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TelegramAccount(BaseModel):
    """
    Main Telegram account model for managing a single account across communities.
    Includes safety monitoring, compliance tracking, and performance metrics.
    """
    __tablename__ = "telegram_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Account Identity
    phone_number = Column(String(20), unique=True, nullable=False)
    telegram_id = Column(BigInteger, unique=True, nullable=True)  # Set after auth
    username = Column(String(50), unique=True, nullable=True)
    first_name = Column(String(64), nullable=False)
    last_name = Column(String(64), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Account Status & Safety
    status = Column(SQLEnum(AccountStatus), default=AccountStatus.INACTIVE)
    safety_level = Column(SQLEnum(SafetyLevel), default=SafetyLevel.CONSERVATIVE)
    is_verified = Column(Boolean, default=False)
    is_ai_disclosed = Column(Boolean, default=True)  # Legal compliance
    
    # Session Management
    session_string = Column(Text, nullable=True)  # Encrypted session data
    session_created_at = Column(DateTime, nullable=True)
    session_last_used = Column(DateTime, nullable=True)
    
    # Account Warming Protocol
    warming_started_at = Column(DateTime, nullable=True)
    warming_completed_at = Column(DateTime, nullable=True)
    warming_progress = Column(Float, default=0.0)  # 0-100 percentage
    warming_phase = Column(String(50), nullable=True)  # profile, contacts, groups, active
    
    # Safety Metrics
    messages_sent_today = Column(Integer, default=0)
    groups_joined_today = Column(Integer, default=0)
    dms_sent_today = Column(Integer, default=0)
    last_activity_reset = Column(DateTime, default=datetime.utcnow)
    
    # Risk Assessment
    flood_wait_count = Column(Integer, default=0)
    spam_warnings = Column(Integer, default=0)
    account_warnings = Column(Integer, default=0)
    last_warning_at = Column(DateTime, nullable=True)
    risk_score = Column(Float, default=0.0)  # 0-100 risk percentage
    
    # Performance Metrics
    total_messages_sent = Column(Integer, default=0)
    total_groups_active = Column(Integer, default=0)
    successful_engagements = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)  # Response success rate
    
    # Legal Compliance
    gdpr_consent_given = Column(Boolean, default=False)
    privacy_policy_accepted = Column(Boolean, default=False)
    data_retention_days = Column(Integer, default=30)
    compliance_review_due = Column(DateTime, nullable=True)
    
    # Configuration
    max_messages_per_day = Column(Integer, default=50)
    max_groups_per_day = Column(Integer, default=2)
    max_dms_per_day = Column(Integer, default=5)
    response_rate_target = Column(Float, default=0.3)  # 30% response rate
    
    # Personality Configuration
    personality_profile = Column(JSON, nullable=True)
    communication_style = Column(JSON, nullable=True)
    interests = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_health_check = Column(DateTime, nullable=True)
    
    # Relationships
    communities = relationship("TelegramCommunity", back_populates="account", cascade="all, delete-orphan")
    conversations = relationship("TelegramConversation", back_populates="account", cascade="all, delete-orphan")
    safety_events = relationship("AccountSafetyEvent", back_populates="account", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_telegram_accounts_status', 'status'),
        Index('ix_telegram_accounts_phone', 'phone_number'),
        Index('ix_telegram_accounts_username', 'username'),
        Index('ix_telegram_accounts_telegram_id', 'telegram_id'),
        Index('ix_telegram_accounts_last_activity', 'last_activity_reset'),
    )
    
    def __repr__(self):
        return f"<TelegramAccount(phone={self.phone_number}, status={self.status})>"
    
    @property
    def is_healthy(self) -> bool:
        """Check if account is in healthy state"""
        return (
            self.status in [AccountStatus.ACTIVE, AccountStatus.WARMING_UP] and
            self.risk_score < 50.0 and
            self.spam_warnings < 3
        )
    
    @property
    def daily_limits_reached(self) -> bool:
        """Check if daily activity limits are reached"""
        if self.last_activity_reset.date() != datetime.utcnow().date():
            return False
        
        return (
            self.messages_sent_today >= self.max_messages_per_day or
            self.groups_joined_today >= self.max_groups_per_day or
            self.dms_sent_today >= self.max_dms_per_day
        )
    
    @property
    def warming_completed(self) -> bool:
        """Check if account warming is completed"""
        return (
            self.warming_completed_at is not None and
            self.warming_progress >= 100.0
        )
    
    def reset_daily_counters(self):
        """Reset daily activity counters"""
        self.messages_sent_today = 0
        self.groups_joined_today = 0
        self.dms_sent_today = 0
        self.last_activity_reset = datetime.utcnow()
    
    def increment_risk_score(self, points: float, reason: str):
        """Increment risk score with reason tracking"""
        self.risk_score = min(100.0, self.risk_score + points)
        
        # Create safety event
        safety_event = AccountSafetyEvent(
            account_id=self.id,
            event_type="risk_increase",
            severity="medium" if points < 10 else "high",
            description=f"Risk increased by {points} points: {reason}",
            data={"points": points, "new_score": self.risk_score}
        )
        return safety_event


class AccountSafetyEvent(BaseModel):
    """
    Safety events and warnings for account monitoring
    """
    __tablename__ = "account_safety_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("telegram_accounts.id"), nullable=False)
    
    # Event Details
    event_type = Column(String(50), nullable=False)  # flood_wait, spam_warning, account_limit
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    description = Column(Text, nullable=False)
    
    # Event Data
    data = Column(JSON, nullable=True)  # Additional event-specific data
    telegram_error_code = Column(Integer, nullable=True)
    telegram_error_message = Column(Text, nullable=True)
    
    # Response Actions
    action_taken = Column(String(100), nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    account = relationship("TelegramAccount", back_populates="safety_events")
    
    # Indexes
    __table_args__ = (
        Index('ix_safety_events_account', 'account_id'),
        Index('ix_safety_events_type', 'event_type'),
        Index('ix_safety_events_severity', 'severity'),
        Index('ix_safety_events_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SafetyEvent(type={self.event_type}, severity={self.severity})>"


class AccountConfiguration(BaseModel):
    """
    Advanced configuration for Telegram account behavior
    """
    __tablename__ = "account_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("telegram_accounts.id"), nullable=False)
    
    # Timing Configuration
    min_response_delay = Column(Integer, default=2)  # seconds
    max_response_delay = Column(Integer, default=30)  # seconds
    typing_speed = Column(Float, default=0.05)  # seconds per character
    
    # Behavioral Configuration
    response_probability = Column(Float, default=0.3)  # 30% chance to respond
    proactive_messaging = Column(Boolean, default=False)
    engagement_style = Column(String(50), default="helpful")  # helpful, casual, professional
    
    # Anti-Detection Settings
    use_random_delays = Column(Boolean, default=True)
    simulate_typing = Column(Boolean, default=True)
    vary_online_patterns = Column(Boolean, default=True)
    use_mobile_behavior = Column(Boolean, default=True)
    
    # Content Filtering
    blocked_keywords = Column(JSON, default=list)
    required_keywords = Column(JSON, default=list)
    content_filters = Column(JSON, default=dict)
    
    # AI Integration
    llm_provider = Column(String(50), default="anthropic")
    model_name = Column(String(100), default="claude-3-sonnet")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=150)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TelegramAccount", backref="configuration", uselist=False)
    
    def __repr__(self):
        return f"<AccountConfig(account_id={self.account_id})>"