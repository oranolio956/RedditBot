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
from app.models.engagement import (
    UserEngagement, UserBehaviorPattern, ProactiveOutreach, 
    EngagementMilestone, UserMilestoneProgress, EngagementType, 
    SentimentType, OutreachType, OutreachStatus
)
from app.models.group_session import (
    GroupSession, GroupMember, GroupConversation, GroupAnalytics,
    GroupType, MemberRole, GroupStatus, MessageFrequency
)
from app.models.sharing import (
    ShareableContent, ContentShare, ReferralProgram, UserReferral,
    ViralMetrics, ShareableContentType, SocialPlatform
)
from app.models.emotional_intelligence import (
    EmotionalProfile, EmotionReading, EmpathyAssessment, EmotionalInteraction,
    EmotionTrajectory, BasicEmotion, EmotionIntensity, DetectionModality,
    EmotionRegulationStrategy, CrisisLevel
)
from app.models.synesthesia import (
    SynestheticProfile, SynestheticTranslation, CrossModalMapping,
    SynestheticExperience, HapticPattern, SynestheticResearchData,
    SynestheticCalibrationSession
)
from app.models.neural_dreams import (
    DreamProfile, DreamSession, DreamContent, BiometricReading,
    TherapeuticProtocol, DreamAnalysis, SafetyMonitoring, LucidDreamTraining,
    NeuroplasticityTracker, DreamState, TherapeuticProtocolType, CrisisLevel,
    BiometricDeviceType
)
from app.models.temporal_dilution import (
    TemporalProfile, TemporalSession, FlowStateSession, TemporalBiometricReading,
    TemporalSafetyMonitoring, TemporalLearningProtocol, CircadianIntegration,
    TemporalState, FlowStateType, TemporalCueType, CircadianPhase, SafetyLevel
)
from app.models.reality_synthesis import (
    RealityProfile, RealitySession, SpatialEnvironment, RealitySafetyMonitoring,
    CollaborativeRealitySession, PortalSystem, RealityLayer, SpatialComputingPlatform,
    RealityRenderingEngine, RealityTransitionType, TherapeuticRealityProtocol
)

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
    
    # Engagement models
    "UserEngagement",
    "UserBehaviorPattern",
    "ProactiveOutreach",
    "EngagementMilestone",
    "UserMilestoneProgress",
    "EngagementType",
    "SentimentType",
    "OutreachType",
    "OutreachStatus",
    
    # Group models
    "GroupSession",
    "GroupMember", 
    "GroupConversation",
    "GroupAnalytics",
    "GroupType",
    "MemberRole",
    "GroupStatus", 
    "MessageFrequency",
    
    # Viral sharing models
    "ShareableContent",
    "ContentShare",
    "ReferralProgram",
    "UserReferral",
    "ViralMetrics",
    "ShareableContentType",
    "SocialPlatform",
    
    # Emotional Intelligence models
    "EmotionalProfile",
    "EmotionReading",
    "EmpathyAssessment",
    "EmotionalInteraction",
    "EmotionTrajectory",
    "BasicEmotion",
    "EmotionIntensity",
    "DetectionModality",
    "EmotionRegulationStrategy",
    "CrisisLevel",
    
    # Digital Synesthesia models
    "SynestheticProfile",
    "SynestheticTranslation",
    "CrossModalMapping",
    "SynestheticExperience", 
    "HapticPattern",
    "SynestheticResearchData",
    "SynestheticCalibrationSession",
    
    # Neural Dreams models
    "DreamProfile",
    "DreamSession",
    "DreamContent",
    "BiometricReading",
    "TherapeuticProtocol",
    "DreamAnalysis",
    "SafetyMonitoring",
    "LucidDreamTraining",
    "NeuroplasticityTracker",
    "DreamState",
    "TherapeuticProtocolType",
    "CrisisLevel",
    "BiometricDeviceType",
    
    # Temporal Dilution models
    "TemporalProfile",
    "TemporalSession",
    "FlowStateSession",
    "TemporalBiometricReading",
    "TemporalSafetyMonitoring",
    "TemporalLearningProtocol",
    "CircadianIntegration",
    "TemporalState",
    "FlowStateType",
    "TemporalCueType",
    "CircadianPhase",
    "SafetyLevel",
    
    # Reality Synthesis models
    "RealityProfile",
    "RealitySession",
    "SpatialEnvironment",
    "RealitySafetyMonitoring",
    "CollaborativeRealitySession",
    "PortalSystem",
    "RealityLayer",
    "SpatialComputingPlatform",
    "RealityRenderingEngine",
    "RealityTransitionType",
    "TherapeuticRealityProtocol",
]