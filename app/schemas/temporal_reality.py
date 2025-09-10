"""
Temporal-Reality Fusion Schemas

Comprehensive Pydantic schemas for temporal dilution, reality synthesis,
and integrated fusion experiences. Provides validation, serialization,
and API interface definitions for all temporal-reality features.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from app.models.temporal_dilution import (
    TemporalState, FlowStateType, TemporalCueType, CircadianPhase, 
    BiometricDeviceType, SafetyLevel as TemporalSafetyLevel
)
from app.models.reality_synthesis import (
    RealityLayer, SpatialComputingPlatform, RealityRenderingEngine,
    RealityTransitionType, TherapeuticRealityProtocol, SafetyLevel as RealitySafetyLevel
)

# ============================================================================
# TEMPORAL DILUTION SCHEMAS
# ============================================================================

class TemporalProfileCreate(BaseModel):
    """Schema for creating temporal dilution profile"""
    user_id: int
    baseline_time_perception_accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Accuracy of time estimation (0.0-1.0)"
    )
    natural_rhythm_period: Optional[float] = Field(
        None, gt=0, description="Natural temporal rhythm period in seconds"
    )
    temporal_sensitivity_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Sensitivity to temporal changes"
    )
    chronotype_classification: Optional[str] = Field(
        None, description="Morning lark, night owl, etc."
    )
    flow_state_frequency: Optional[float] = Field(
        None, ge=0.0, description="Natural flow state frequency per week"
    )
    optimal_flow_duration: Optional[float] = Field(
        None, gt=0, description="Optimal flow state duration in minutes"
    )
    preferred_temporal_state: Optional[TemporalState] = None
    optimal_learning_time_ratio: Optional[float] = Field(
        None, gt=0, description="Preferred subjective:objective time ratio"
    )
    maximum_safe_temporal_distortion: Optional[float] = Field(
        None, gt=0, le=5.0, description="Maximum safe time dilation ratio"
    )
    disorientation_sensitivity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Susceptibility to temporal confusion"
    )
    preferred_temporal_cue_types: Optional[List[Dict[str, Any]]] = None
    contraindications: Optional[List[Dict[str, Any]]] = None

    class Config:
        use_enum_values = True

class TemporalProfileResponse(BaseModel):
    """Schema for temporal profile responses"""
    id: int
    user_id: int
    baseline_time_perception_accuracy: Optional[float]
    natural_rhythm_period: Optional[float]
    temporal_sensitivity_score: Optional[float]
    chronotype_classification: Optional[str]
    flow_state_frequency: Optional[float]
    optimal_flow_duration: Optional[float]
    preferred_temporal_state: Optional[TemporalState]
    optimal_learning_time_ratio: Optional[float]
    maximum_safe_temporal_distortion: Optional[float]
    disorientation_sensitivity: Optional[float]
    preferred_temporal_cue_types: Optional[List[Dict[str, Any]]]
    contraindications: Optional[List[Dict[str, Any]]]
    created_at: datetime
    updated_at: Optional[datetime]
    last_temporal_session_at: Optional[datetime]
    profile_calibration_status: Optional[str]

    class Config:
        from_attributes = True
        use_enum_values = True

class TemporalExperienceRequest(BaseModel):
    """Schema for creating temporal dilution experience"""
    profile_id: int
    target_temporal_state: TemporalState
    target_dilation_ratio: float = Field(
        ge=0.5, le=5.0, description="Desired subjective:objective time ratio"
    )
    duration_minutes: int = Field(
        ge=5, le=180, description="Session duration in minutes"
    )
    flow_state_type: FlowStateType
    temporal_cues: List[TemporalCueType]
    safety_thresholds: Optional[Dict[str, float]] = None
    personalization_level: float = Field(
        ge=0.0, le=1.0, description="Level of personalization to apply"
    )
    circadian_optimization: bool = Field(
        default=True, description="Optimize for circadian rhythm"
    )

    @validator('temporal_cues')
    def validate_temporal_cues(cls, v):
        if len(v) == 0:
            raise ValueError('At least one temporal cue must be specified')
        return v

    class Config:
        use_enum_values = True

class TemporalSessionResponse(BaseModel):
    """Schema for temporal session responses"""
    id: int
    temporal_profile_id: int
    session_uuid: str
    session_type: FlowStateType
    target_temporal_state: TemporalState
    planned_duration_minutes: Optional[int]
    target_time_dilation_ratio: Optional[float]
    temporal_cue_sequence: Optional[List[Dict[str, Any]]]
    ai_model_version: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True
        use_enum_values = True

class TemporalStateResponse(BaseModel):
    """Schema for temporal state readings"""
    timestamp: datetime
    session_uuid: str
    detected_state: TemporalState
    confidence: float = Field(ge=0.0, le=1.0)
    time_dilation_ratio: float
    flow_depth: float = Field(ge=0.0, le=1.0)
    safety_level: TemporalSafetyLevel
    biometric_data: Dict[str, float]
    intervention_needed: bool

    class Config:
        use_enum_values = True

class FlowStateInductionRequest(BaseModel):
    """Schema for flow state induction request"""
    flow_type: FlowStateType
    target_depth: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Desired flow state depth"
    )
    neural_entrainment_enabled: bool = True
    temporal_enhancement: bool = True

    class Config:
        use_enum_values = True

class FlowStateResponse(BaseModel):
    """Schema for flow state induction results"""
    session_uuid: str
    flow_type: FlowStateType
    flow_state_achieved: bool
    time_to_flow_entry: Optional[float]
    flow_depth_score: float = Field(ge=0.0, le=1.0)
    neural_entrainment_success: bool
    temporal_dilation_applied: bool
    timestamp: datetime

    class Config:
        use_enum_values = True

class TemporalSafetyResponse(BaseModel):
    """Schema for temporal safety status"""
    session_uuid: str
    current_safety_level: TemporalSafetyLevel
    safety_trajectory: Optional[str]
    temporal_confusion_markers: Optional[List[Dict[str, Any]]]
    safety_interventions_triggered: Optional[List[Dict[str, Any]]]
    intervention_effectiveness: Optional[Dict[str, Any]]
    monitoring_started_at: datetime
    last_updated: datetime

    class Config:
        use_enum_values = True

# ============================================================================
# REALITY SYNTHESIS SCHEMAS
# ============================================================================

class RealityProfileCreate(BaseModel):
    """Schema for creating reality synthesis profile"""
    user_id: int
    supported_platforms: Optional[List[Dict[str, Any]]] = None
    platform_preferences: Optional[Dict[str, Any]] = None
    hardware_performance_profile: Optional[Dict[str, Any]] = None
    spatial_awareness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Spatial understanding ability"
    )
    depth_perception_accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="3D depth judgment ability"
    )
    motion_sickness_susceptibility: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Tendency to get motion sick"
    )
    preferred_reality_layers: Optional[List[str]] = None
    reality_immersion_tolerance: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Tolerance for deep immersion"
    )
    maximum_session_duration: Optional[float] = Field(
        None, gt=0, description="Maximum session duration in minutes"
    )
    comfort_zone_boundaries: Optional[Dict[str, Any]] = None
    therapeutic_goals: Optional[List[Dict[str, Any]]] = None
    contraindications: Optional[List[Dict[str, Any]]] = None
    accessibility_accommodations: Optional[Dict[str, Any]] = None

class RealityProfileResponse(BaseModel):
    """Schema for reality profile responses"""
    id: int
    user_id: int
    supported_platforms: Optional[List[Dict[str, Any]]]
    platform_preferences: Optional[Dict[str, Any]]
    hardware_performance_profile: Optional[Dict[str, Any]]
    spatial_awareness_score: Optional[float]
    depth_perception_accuracy: Optional[float]
    motion_sickness_susceptibility: Optional[float]
    preferred_reality_layers: Optional[List[str]]
    reality_immersion_tolerance: Optional[float]
    maximum_session_duration: Optional[float]
    comfort_zone_boundaries: Optional[Dict[str, Any]]
    therapeutic_goals: Optional[List[Dict[str, Any]]]
    contraindications: Optional[List[Dict[str, Any]]]
    accessibility_accommodations: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]
    last_reality_session_at: Optional[datetime]
    profile_calibration_status: Optional[str]

    class Config:
        from_attributes = True

class RealityExperienceRequest(BaseModel):
    """Schema for creating reality synthesis experience"""
    profile_id: int
    primary_reality_layer: RealityLayer
    target_reality_layers: Optional[List[RealityLayer]] = None
    rendering_engine: RealityRenderingEngine
    platform: SpatialComputingPlatform
    therapeutic_protocol: Optional[TherapeuticRealityProtocol] = None
    transition_types: Optional[List[RealityTransitionType]] = None
    safety_thresholds: Optional[Dict[str, float]] = None
    collaborative_enabled: bool = False
    duration_minutes: int = Field(
        ge=5, le=180, description="Session duration in minutes"
    )

    class Config:
        use_enum_values = True

class RealitySessionResponse(BaseModel):
    """Schema for reality session responses"""
    id: int
    reality_profile_id: int
    session_uuid: str
    primary_reality_layer: RealityLayer
    session_type: Optional[TherapeuticRealityProtocol]
    target_duration_minutes: Optional[int]
    rendering_engine: RealityRenderingEngine
    platform_used: SpatialComputingPlatform
    started_at: datetime
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True
        use_enum_values = True

class RealityStateResponse(BaseModel):
    """Schema for reality state readings"""
    timestamp: datetime
    session_uuid: str
    current_layer: RealityLayer
    presence_score: float = Field(ge=0.0, le=1.0)
    immersion_depth: float = Field(ge=0.0, le=1.0)
    spatial_awareness: float = Field(ge=0.0, le=1.0)
    safety_level: RealitySafetyLevel
    technical_performance: Dict[str, float]
    user_comfort: float = Field(ge=0.0, le=1.0)

    class Config:
        use_enum_values = True

class PortalTransitionRequest(BaseModel):
    """Schema for portal transition request"""
    source_layer: RealityLayer
    target_layer: RealityLayer
    transition_type: RealityTransitionType
    transition_duration: float = Field(
        ge=0.5, le=10.0, description="Transition duration in seconds"
    )
    spatial_anchors: Optional[List[Dict[str, Any]]] = None
    safety_protocols: Optional[List[str]] = None

    class Config:
        use_enum_values = True

class SpatialEnvironmentRequest(BaseModel):
    """Schema for creating spatial environment"""
    generation_prompt: str = Field(
        min_length=10, max_length=1000, description="Prompt for environment generation"
    )
    layer_type: RealityLayer
    therapeutic_purpose: Optional[TherapeuticRealityProtocol] = None
    environment_name: Optional[str] = None
    quality_preset: str = Field(
        default="standard", regex="^(draft|standard|high|ultra)$"
    )
    platform_targets: Optional[List[SpatialComputingPlatform]] = None

    @validator('generation_prompt')
    def validate_prompt_content(cls, v):
        # Basic content validation
        if any(word in v.lower() for word in ['violence', 'harm', 'inappropriate']):
            raise ValueError('Prompt contains inappropriate content')
        return v

    class Config:
        use_enum_values = True

class SpatialEnvironmentResponse(BaseModel):
    """Schema for spatial environment responses"""
    id: int
    environment_uuid: str
    environment_name: str
    environment_category: Optional[str]
    reality_layer_compatibility: Optional[List[str]]
    target_use_cases: Optional[List[str]]
    therapeutic_applications: Optional[List[str]]
    generation_prompt: Optional[str]
    ai_model_used: Optional[str]
    content_safety_rating: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class CollaborativeSessionRequest(BaseModel):
    """Schema for collaborative session creation"""
    host_profile_id: int
    max_participants: int = Field(ge=2, le=50)
    primary_reality_layer: RealityLayer
    target_reality_layers: Optional[List[RealityLayer]] = None
    rendering_engine: RealityRenderingEngine
    platform: SpatialComputingPlatform
    therapeutic_protocol: Optional[TherapeuticRealityProtocol] = None
    session_privacy_level: str = Field(
        default="invite_only", regex="^(public|private|invite_only)$"
    )
    duration_minutes: int = Field(ge=10, le=240)

    class Config:
        use_enum_values = True

class RealitySafetyResponse(BaseModel):
    """Schema for reality safety status"""
    session_uuid: str
    current_safety_level: RealitySafetyLevel
    safety_trajectory: Optional[str]
    reality_confusion_markers: Optional[List[Dict[str, Any]]]
    motion_sickness_events: Optional[List[Dict[str, Any]]]
    safety_interventions_triggered: Optional[List[Dict[str, Any]]]
    intervention_effectiveness: Optional[Dict[str, Any]]
    monitoring_started_at: datetime
    last_updated: datetime

    class Config:
        use_enum_values = True

# ============================================================================
# FUSION SYSTEM SCHEMAS
# ============================================================================

class TemporalRealityFusionRequest(BaseModel):
    """Schema for creating temporal-reality fusion experience"""
    temporal_profile_id: int
    reality_profile_id: int
    
    # Temporal configuration
    temporal_config: TemporalExperienceRequest
    
    # Reality configuration
    reality_config: RealityExperienceRequest
    
    # Fusion parameters
    synchronization_mode: str = Field(
        default="synchronized",
        regex="^(temporal_leads|reality_leads|synchronized)$"
    )
    temporal_reality_ratio: float = Field(
        ge=0.0, le=1.0, description="Balance between temporal and reality emphasis"
    )
    cross_modal_enhancement: bool = Field(
        default=True, description="Enable cross-modal temporal-reality enhancements"
    )
    therapeutic_integration_level: str = Field(
        default="basic",
        regex="^(basic|advanced|clinical)$"
    )
    safety_priority: str = Field(
        default="balanced",
        regex="^(temporal_first|reality_first|balanced)$"
    )

class TemporalRealityFusionResponse(BaseModel):
    """Schema for temporal-reality fusion session responses"""
    fusion_session_uuid: str
    temporal_session: TemporalSessionResponse
    reality_session: RealitySessionResponse
    synchronization_mode: str
    temporal_reality_ratio: float
    cross_modal_enhancement: bool
    therapeutic_integration_level: str
    safety_priority: str
    created_at: datetime

class TemporalRealityStateResponse(BaseModel):
    """Schema for fusion state readings"""
    timestamp: datetime
    fusion_session_uuid: str
    temporal_state: TemporalStateResponse
    reality_state: RealityStateResponse
    fusion_synchronization_score: float = Field(
        ge=0.0, le=1.0, description="How well temporal and reality are synchronized"
    )
    cross_modal_coherence: float = Field(
        ge=0.0, le=1.0, description="Coherence between time perception and spatial experience"
    )
    overall_safety_level: str
    therapeutic_progress_indicators: Dict[str, float]
    user_agency_level: float = Field(
        ge=0.0, le=1.0, description="User control over experience"
    )

class TherapeuticProtocolRequest(BaseModel):
    """Schema for therapeutic protocol execution"""
    protocol_name: str
    target_outcomes: Dict[str, float] = Field(
        description="Desired therapeutic outcomes with target values"
    )
    session_customization: Optional[Dict[str, Any]] = None
    professional_oversight_required: bool = False

    @validator('protocol_name')
    def validate_protocol_name(cls, v):
        valid_protocols = ['ptsd_temporal_reality', 'learning_acceleration', 'anxiety_management']
        if v not in valid_protocols:
            raise ValueError(f'Protocol must be one of: {valid_protocols}')
        return v

class TherapeuticProtocolResponse(BaseModel):
    """Schema for therapeutic protocol execution results"""
    fusion_session_uuid: str
    protocol_name: str
    execution_success: bool
    phases_completed: List[Dict[str, Any]]
    therapeutic_outcomes: Dict[str, float]
    safety_events: List[Dict[str, Any]]
    professional_recommendations: List[str]
    follow_up_required: bool
    session_timestamp: datetime

# ============================================================================
# COMMON SCHEMAS
# ============================================================================

class EmergencyResetRequest(BaseModel):
    """Schema for emergency reset requests"""
    reason: str = Field(
        min_length=5, max_length=500, description="Reason for emergency reset"
    )
    immediate_action_required: bool = True
    user_reported_issue: Optional[str] = None

class SystemHealthResponse(BaseModel):
    """Schema for system health check responses"""
    status: str
    service: str
    version: str
    timestamp: datetime
    features: Dict[str, bool]
    performance_metrics: Optional[Dict[str, float]] = None
    active_sessions_count: Optional[int] = None

class BiometricReadingResponse(BaseModel):
    """Schema for biometric data responses"""
    timestamp: datetime
    device_type: BiometricDeviceType
    eeg_data: Dict[str, float]
    temporal_indicators: Dict[str, float]
    flow_indicators: Dict[str, float]
    signal_quality: float

    class Config:
        use_enum_values = True

class SessionAnalyticsResponse(BaseModel):
    """Schema for session analytics"""
    session_uuid: str
    session_type: str
    duration_minutes: float
    success_metrics: Dict[str, float]
    user_satisfaction: Optional[float]
    therapeutic_outcomes: Optional[Dict[str, float]]
    safety_incidents: int
    system_performance: Dict[str, float]
    timestamp: datetime

# ============================================================================
# VALIDATION HELPERS
# ============================================================================

class SafetyValidationRequest(BaseModel):
    """Schema for safety validation requests"""
    user_id: int
    medical_history: Optional[Dict[str, Any]] = None
    current_medications: Optional[List[str]] = None
    previous_adverse_reactions: Optional[List[str]] = None
    consent_levels: Dict[str, bool]

    @validator('consent_levels')
    def validate_consent(cls, v):
        required_consents = ['temporal_manipulation', 'reality_synthesis', 'biometric_monitoring', 'data_collection']
        for consent in required_consents:
            if consent not in v or not v[consent]:
                raise ValueError(f'Consent required for: {consent}')
        return v

class PerformanceMetrics(BaseModel):
    """Schema for system performance metrics"""
    response_time_ms: float
    throughput_requests_per_second: float
    error_rate_percentage: float
    system_resource_usage: Dict[str, float]
    user_experience_score: float
    timestamp: datetime

# ============================================================================
# CONFIGURATION SCHEMAS
# ============================================================================

class TemporalEngineConfig(BaseModel):
    """Schema for temporal engine configuration"""
    max_dilation_ratio: float = Field(ge=1.0, le=5.0)
    max_session_duration_minutes: int = Field(ge=5, le=180)
    safety_monitoring_interval_seconds: int = Field(ge=1, le=60)
    neural_entrainment_enabled: bool = True
    circadian_optimization_enabled: bool = True

class RealityMatrixConfig(BaseModel):
    """Schema for reality matrix configuration"""
    supported_platforms: List[SpatialComputingPlatform]
    max_participants_collaborative: int = Field(ge=2, le=50)
    neural_radiance_fields_enabled: bool = True
    real_time_adaptation_enabled: bool = True
    safety_monitoring_interval_seconds: int = Field(ge=1, le=30)

    class Config:
        use_enum_values = True

class FusionSystemConfig(BaseModel):
    """Schema for fusion system configuration"""
    temporal_config: TemporalEngineConfig
    reality_config: RealityMatrixConfig
    synchronization_tolerance: float = Field(
        ge=0.1, le=1.0, description="Acceptable synchronization deviation"
    )
    cross_modal_enhancement_strength: float = Field(ge=0.0, le=1.0)
    therapeutic_protocols_enabled: bool = True
    emergency_intervention_enabled: bool = True
