"""
Database Models Package

Centralized import module for all database models.
Provides easy access to all model classes throughout the application.
"""

from app.models.user import User
from app.models.conversation import Conversation, Message, ConversationSession
from app.models.personality import PersonalityProfile, PersonalityTrait, UserPersonalityMapping
from app.models.risk_assessment import RiskAssessment, RiskFactor, ConversationRisk
from app.models.analytics import UserActivity, ConversationAnalytics, SystemMetrics
from app.models.system_config import SystemConfiguration, FeatureFlag, RateLimitConfig
from app.models.audit import AuditLog, SecurityEvent, PerformanceMetric

__all__ = [
    # User models
    "User",
    
    # Conversation models
    "Conversation",
    "Message",
    "ConversationSession",
    
    # Personality models
    "PersonalityProfile",
    "PersonalityTrait",
    "UserPersonalityMapping",
    
    # Risk assessment models
    "RiskAssessment",
    "RiskFactor",
    "ConversationRisk",
    
    # Analytics models
    "UserActivity",
    "ConversationAnalytics",
    "SystemMetrics",
    
    # System configuration models
    "SystemConfiguration",
    "FeatureFlag",
    "RateLimitConfig",
    
    # Audit models
    "AuditLog",
    "SecurityEvent",
    "PerformanceMetric",
]