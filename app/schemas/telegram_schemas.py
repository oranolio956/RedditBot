"""
Telegram Management Schemas
Pydantic models for request/response validation and serialization.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from app.models.telegram_account import AccountStatus, SafetyLevel
from app.models.telegram_community import CommunityType, EngagementStrategy, CommunityStatus
from app.models.telegram_conversation import ConversationStatus, MessageType, MessageDirection


# Request Models
class CreateTelegramAccountRequest(BaseModel):
    """Request model for creating a new Telegram account"""
    
    phone_number: str = Field(..., description="Phone number with country code (+1234567890)")
    first_name: str = Field(..., min_length=1, max_length=64, description="First name")
    last_name: Optional[str] = Field(None, max_length=64, description="Last name")
    bio: Optional[str] = Field(None, max_length=500, description="Bio/description")
    
    safety_level: SafetyLevel = Field(SafetyLevel.CONSERVATIVE, description="Safety level")
    is_ai_disclosed: bool = Field(True, description="Whether AI nature is disclosed")
    gdpr_consent_given: bool = Field(False, description="GDPR consent status")
    privacy_policy_accepted: bool = Field(False, description="Privacy policy acceptance")
    
    personality_profile: Optional[Dict[str, Any]] = Field(None, description="Personality configuration")
    communication_style: Optional[Dict[str, Any]] = Field(None, description="Communication style")
    interests: Optional[List[str]] = Field(None, description="List of interests")
    
    max_messages_per_day: int = Field(50, ge=1, le=200, description="Daily message limit")
    max_groups_per_day: int = Field(2, ge=0, le=10, description="Daily group join limit")
    max_dms_per_day: int = Field(5, ge=0, le=50, description="Daily DM limit")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not v.startswith('+'):
            raise ValueError('Phone number must start with +')
        if len(v) < 10:
            raise ValueError('Phone number too short')
        return v


class JoinCommunityRequest(BaseModel):
    """Request model for joining a Telegram community"""
    
    chat_id: int = Field(..., description="Telegram chat ID")
    title: str = Field(..., min_length=1, max_length=255, description="Community title")
    username: Optional[str] = Field(None, max_length=100, description="Community username")
    invite_link: Optional[str] = Field(None, description="Invite link used")
    community_type: CommunityType = Field(..., description="Type of community")
    engagement_strategy: Optional[EngagementStrategy] = Field(
        EngagementStrategy.PARTICIPANT, 
        description="Engagement strategy"
    )
    
    @validator('invite_link')
    def validate_invite_link(cls, v):
        if v and not (v.startswith('https://t.me/') or v.startswith('https://telegram.me/')):
            raise ValueError('Invalid Telegram invite link')
        return v


class SendMessageRequest(BaseModel):
    """Request model for sending a message"""
    
    chat_id: int = Field(..., description="Target chat ID")
    content: str = Field(..., min_length=1, max_length=4096, description="Message content")
    reply_to_message_id: Optional[int] = Field(None, description="Message ID to reply to")
    
    # Safety options
    force_send: bool = Field(False, description="Force send despite safety warnings")
    schedule_optimal_time: bool = Field(False, description="Schedule for optimal timing")
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        return v.strip()


class PersonalityAdaptationRequest(BaseModel):
    """Request model for personality adaptation"""
    
    context: Dict[str, Any] = Field(..., description="Context for adaptation")
    save_as_default: bool = Field(False, description="Save as default personality")
    community_id: Optional[str] = Field(None, description="Community to adapt for")
    
    @validator('context')
    def validate_context(cls, v):
        required_keys = ['situation', 'target_audience']
        if not all(key in v for key in required_keys):
            raise ValueError(f'Context must include: {required_keys}')
        return v


# Response Models
class TelegramAccountResponse(BaseModel):
    """Response model for Telegram account"""
    
    id: str
    phone_number: str
    status: AccountStatus
    safety_level: SafetyLevel
    is_healthy: bool
    risk_score: float
    engagement_rate: Optional[float] = None
    total_messages_sent: Optional[int] = None
    total_communities: Optional[int] = None
    created_at: datetime
    last_health_check: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class TelegramCommunityResponse(BaseModel):
    """Response model for Telegram community"""
    
    id: str
    title: str
    username: Optional[str] = None
    status: CommunityStatus
    engagement_strategy: EngagementStrategy
    engagement_score: float
    reputation_score: float
    trust_level: Optional[float] = None
    member_count: Optional[int] = None
    join_date: Optional[datetime] = None
    last_activity_date: Optional[datetime] = None
    warning_count: Optional[int] = None
    
    class Config:
        use_enum_values = True


class ConversationSummaryResponse(BaseModel):
    """Response model for conversation summary"""
    
    id: str
    chat_id: int
    is_group: bool
    message_count: int
    our_message_count: Optional[int] = None
    last_message_date: Optional[datetime] = None
    engagement_score: float
    response_rate: Optional[float] = None
    status: ConversationStatus
    
    class Config:
        use_enum_values = True


class SafetyEventResponse(BaseModel):
    """Response model for safety events"""
    
    id: str
    event_type: str
    severity: str
    description: str
    created_at: datetime
    resolved: Optional[bool] = None
    resolved_at: Optional[datetime] = None


class AccountHealthResponse(BaseModel):
    """Response model for account health"""
    
    account_id: str
    is_healthy: bool
    health_score: float
    risk_score: float
    threat_level: str
    active_warnings: int
    daily_activity: Dict[str, Any]
    recommendations: List[str]
    last_check: datetime


class TelegramAccountDetailResponse(BaseModel):
    """Detailed response model for Telegram account"""
    
    account: TelegramAccountResponse
    status_data: Dict[str, Any]
    communities: List[TelegramCommunityResponse]
    recent_conversations: List[ConversationSummaryResponse]
    recent_safety_events: List[SafetyEventResponse]


# Analytics Models
class MessageStatistics(BaseModel):
    """Message statistics model"""
    
    total_sent: int
    total_received: int
    daily_average: float
    peak_hour: Optional[int] = None
    response_rate: float
    avg_length: Optional[float] = None


class EngagementStatistics(BaseModel):
    """Engagement statistics model"""
    
    total_interactions: int
    positive_reactions: int
    negative_reactions: int
    mentions_received: int
    conversations_started: int
    avg_engagement_score: float


class CommunityStatistics(BaseModel):
    """Community statistics model"""
    
    total_communities: int
    active_communities: int
    avg_reputation_score: float
    top_performing_community: Optional[str] = None
    communities_with_warnings: int


class SafetyStatistics(BaseModel):
    """Safety statistics model"""
    
    total_safety_events: int
    flood_waits: int
    spam_warnings: int
    risk_score_trend: str  # improving, declining, stable
    avg_risk_score: float
    days_without_incidents: int


class AccountAnalyticsResponse(BaseModel):
    """Comprehensive analytics response"""
    
    account_id: str
    period_days: int
    message_statistics: MessageStatistics
    engagement_statistics: EngagementStatistics
    community_statistics: CommunityStatistics
    safety_statistics: SafetyStatistics
    generated_at: datetime


# Community Insight Models
class CommunityInsightResponse(BaseModel):
    """Community insight response model"""
    
    insight_type: str
    title: str
    description: str
    confidence_score: float
    impact_potential: str
    implementation_difficulty: str
    recommended_actions: List[str]
    expected_outcomes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None


class CommunityAnalysisResponse(BaseModel):
    """Community analysis response model"""
    
    community_id: str
    community_title: str
    analysis_date: datetime
    
    activity_patterns: Dict[str, Any]
    communication_style: Dict[str, Any]
    content_preferences: Dict[str, Any]
    member_dynamics: Dict[str, Any]
    
    engagement_opportunities: List[str]
    risk_factors: List[str]
    personality_recommendations: Dict[str, float]


# Safety and Monitoring Models
class SafetyCheckRequest(BaseModel):
    """Safety check request model"""
    
    action_type: str = Field(..., description="Type of action to check")
    target_chat_id: Optional[int] = Field(None, description="Target chat ID")
    message_content: Optional[str] = Field(None, description="Message content to check")
    force_check: bool = Field(False, description="Force comprehensive check")


class SafetyCheckResponse(BaseModel):
    """Safety check response model"""
    
    is_safe: bool
    risk_score: float
    threat_level: str
    recommended_action: str
    issues_found: List[str]
    recommendations: List[str]
    estimated_wait_time: Optional[int] = None  # seconds
    checked_at: datetime


class RealTimeStatusUpdate(BaseModel):
    """Real-time status update model for WebSocket"""
    
    account_id: str
    status: AccountStatus
    health_score: float
    risk_score: float
    active_conversations: int
    messages_sent_today: int
    last_activity: Optional[datetime] = None
    current_action: Optional[str] = None
    timestamp: datetime
    
    class Config:
        use_enum_values = True


# Configuration Models
class AccountConfigurationRequest(BaseModel):
    """Account configuration update request"""
    
    safety_level: Optional[SafetyLevel] = None
    max_messages_per_day: Optional[int] = Field(None, ge=1, le=200)
    max_groups_per_day: Optional[int] = Field(None, ge=0, le=10)
    max_dms_per_day: Optional[int] = Field(None, ge=0, le=50)
    
    personality_profile: Optional[Dict[str, Any]] = None
    communication_style: Optional[Dict[str, Any]] = None
    interests: Optional[List[str]] = None
    
    response_rate_target: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Advanced timing settings
    min_response_delay: Optional[int] = Field(None, ge=1, le=300)
    max_response_delay: Optional[int] = Field(None, ge=5, le=3600)
    typing_speed: Optional[float] = Field(None, ge=0.01, le=0.1)


class CommunityConfigurationRequest(BaseModel):
    """Community configuration update request"""
    
    engagement_strategy: Optional[EngagementStrategy] = None
    formality_level: Optional[str] = Field(None, regex="^(formal|semi-formal|casual)$")
    response_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    topics: Optional[List[str]] = None
    avoided_topics: Optional[List[str]] = None
    
    personality_adaptations: Optional[Dict[str, float]] = None
    
    @validator('topics')
    def validate_topics(cls, v):
        if v and len(v) > 20:
            raise ValueError('Maximum 20 topics allowed')
        return v


# Batch Operation Models
class BatchMessageRequest(BaseModel):
    """Batch message sending request"""
    
    messages: List[SendMessageRequest] = Field(..., max_items=10)
    schedule_optimally: bool = Field(True, description="Schedule for optimal timing")
    respect_rate_limits: bool = Field(True, description="Respect rate limits")
    
    @validator('messages')
    def validate_messages(cls, v):
        if len(v) == 0:
            raise ValueError('At least one message required')
        return v


class BatchOperationResponse(BaseModel):
    """Batch operation response"""
    
    operation_id: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_pending: int
    
    results: List[Dict[str, Any]]
    started_at: datetime
    estimated_completion: Optional[datetime] = None


# Error Models
class TelegramError(BaseModel):
    """Telegram error model"""
    
    error_type: str
    error_code: Optional[int] = None
    error_message: str
    context: Dict[str, Any]
    timestamp: datetime
    recommended_action: str


class ValidationError(BaseModel):
    """Validation error model"""
    
    field: str
    message: str
    invalid_value: Any


# Export all models
__all__ = [
    # Request models
    'CreateTelegramAccountRequest',
    'JoinCommunityRequest', 
    'SendMessageRequest',
    'PersonalityAdaptationRequest',
    'SafetyCheckRequest',
    'AccountConfigurationRequest',
    'CommunityConfigurationRequest',
    'BatchMessageRequest',
    
    # Response models
    'TelegramAccountResponse',
    'TelegramCommunityResponse',
    'ConversationSummaryResponse',
    'SafetyEventResponse',
    'AccountHealthResponse',
    'TelegramAccountDetailResponse',
    'SafetyCheckResponse',
    'RealTimeStatusUpdate',
    'BatchOperationResponse',
    
    # Analytics models
    'MessageStatistics',
    'EngagementStatistics', 
    'CommunityStatistics',
    'SafetyStatistics',
    'AccountAnalyticsResponse',
    
    # Insight models
    'CommunityInsightResponse',
    'CommunityAnalysisResponse',
    
    # Error models
    'TelegramError',
    'ValidationError'
]