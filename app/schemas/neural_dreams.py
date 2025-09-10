"""
Neural Dreams Schemas - Pydantic Models for API Validation

Comprehensive data validation schemas for the revolutionary dream therapy platform.
Ensures data integrity, security, and therapeutic safety across all operations.

Features:
- Therapeutic protocol validation with safety constraints
- Biometric data validation for clinical-grade processing  
- Cultural sensitivity and personalization parameters
- Crisis intervention and emergency response schemas
- Professional integration and consent management
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import re

# Enums for type safety and validation
class DreamStateEnum(str, Enum):
    """Validated dream and consciousness states"""
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    LUCID_DREAM = "lucid_dream"
    MEDITATIVE = "meditative"
    HYPNAGOGIC = "hypnagogic"
    HYPNOPOMPIC = "hypnopompic"

class TherapeuticProtocolEnum(str, Enum):
    """Evidence-based therapeutic interventions"""
    PTSD_PROCESSING = "ptsd_processing"
    NIGHTMARE_TRANSFORMATION = "nightmare_transformation"
    ANXIETY_REDUCTION = "anxiety_reduction"
    TRAUMA_INTEGRATION = "trauma_integration"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CREATIVE_ENHANCEMENT = "creative_enhancement"
    LUCID_TRAINING = "lucid_training"
    CONSCIOUSNESS_EXPANSION = "consciousness_expansion"

class BiometricDeviceEnum(str, Enum):
    """Supported biometric monitoring devices"""
    BRAINBIT_EEG = "brainbit_eeg"
    MUSE_EEG = "muse_eeg"
    HEART_RATE_MONITOR = "heart_rate_monitor"
    SKIN_CONDUCTANCE = "skin_conductance"
    BREATHING_SENSOR = "breathing_sensor"
    EYE_TRACKING = "eye_tracking"
    HAPTIC_SUIT = "haptic_suit"
    VR_HEADSET = "vr_headset"

class CrisisLevelEnum(str, Enum):
    """Crisis intervention urgency levels"""
    SAFE = "safe"
    MILD_DISTRESS = "mild_distress"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_RISK = "high_risk"
    EMERGENCY_INTERVENTION = "emergency_intervention"

# Core Request/Response Schemas

class DreamGenerationRequest(BaseModel):
    """
    Comprehensive request for generating therapeutic dream experiences.
    Includes safety validation and therapeutic protocol screening.
    """
    therapeutic_goals: List[TherapeuticProtocolEnum] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Primary therapeutic objectives for the dream session"
    )
    target_dream_state: DreamStateEnum = Field(
        default=DreamStateEnum.REM_SLEEP,
        description="Desired consciousness state for therapeutic intervention"
    )
    duration_minutes: int = Field(
        default=20,
        ge=5,
        le=120,
        description="Session duration in minutes (5-120 minutes)"
    )
    personalization_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Degree of personalization based on user history (0.0-1.0)"
    )
    
    # Safety and Constraints
    safety_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Safety parameters and contraindication screening"
    )
    triggering_content_filters: List[str] = Field(
        default_factory=list,
        description="Content types to avoid for psychological safety"
    )
    
    # Technical Configuration
    enable_biometric_adaptation: bool = Field(
        default=True,
        description="Enable real-time adaptation based on biometric feedback"
    )
    enable_multimodal_content: bool = Field(
        default=True,
        description="Generate visual, audio, haptic, and narrative content"
    )
    enable_real_time_adaptation: bool = Field(
        default=True,
        description="Allow dynamic content modification during session"
    )
    
    # Cultural and Personal Preferences
    cultural_considerations: List[str] = Field(
        default_factory=list,
        description="Cultural adaptations and sensitivity requirements"
    )
    preferred_themes: List[str] = Field(
        default_factory=list,
        description="Preferred dream themes (nature, memories, etc.)"
    )
    
    # Professional Integration
    therapist_oversight_required: bool = Field(
        default=False,
        description="Require licensed therapist supervision"
    )
    emergency_contact_info: Optional[Dict[str, str]] = Field(
        default=None,
        description="Emergency contact information for crisis intervention"
    )

    @validator('safety_constraints')
    def validate_safety_constraints(cls, v):
        """Validate safety constraint structure"""
        allowed_keys = {
            'max_stress_level', 'trauma_indicators', 'dissociation_risk',
            'medication_interactions', 'contraindications'
        }
        if not isinstance(v, dict):
            raise ValueError('Safety constraints must be a dictionary')
        
        invalid_keys = set(v.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f'Invalid safety constraint keys: {invalid_keys}')
        
        return v

    @validator('duration_minutes')
    def validate_duration_therapeutic_guidelines(cls, v, values):
        """Validate duration based on therapeutic protocols"""
        if 'therapeutic_goals' in values:
            goals = values['therapeutic_goals']
            
            # PTSD processing requires longer sessions
            if TherapeuticProtocolEnum.PTSD_PROCESSING in goals and v < 30:
                raise ValueError('PTSD processing requires minimum 30 minutes')
            
            # Nightmare transformation works better with shorter initial sessions
            if TherapeuticProtocolEnum.NIGHTMARE_TRANSFORMATION in goals and v > 45:
                raise ValueError('Nightmare transformation recommended ≤45 minutes initially')
        
        return v

class BiometricSubmissionRequest(BaseModel):
    """
    Real-time biometric data submission for therapeutic adaptation.
    Supports multiple device types and clinical-grade data validation.
    """
    session_uuid: str = Field(
        ...,
        regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        description="Active dream session UUID"
    )
    device_type: BiometricDeviceEnum = Field(
        ...,
        description="Type of biometric monitoring device"
    )
    timestamp: datetime = Field(
        ...,
        description="Precise timestamp of biometric reading"
    )
    
    # EEG Data (Primary consciousness indicators)
    eeg_data: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="EEG channel data with frequency band analysis"
    )
    
    # Cardiovascular Metrics
    heart_rate_bpm: Optional[int] = Field(
        default=None,
        ge=30,
        le=200,
        description="Heart rate in beats per minute"
    )
    heart_rate_variability: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="HRV RMSSD measure in milliseconds"
    )
    
    # Autonomic Nervous System
    skin_conductance_microsiemens: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Skin conductance in microsiemens"
    )
    breathing_rate_per_minute: Optional[float] = Field(
        default=None,
        ge=5.0,
        le=30.0,
        description="Breathing rate per minute"
    )
    
    # Environmental Context
    ambient_conditions: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Environmental factors affecting readings"
    )
    
    # Data Quality Indicators
    signal_quality_score: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Signal quality assessment (0.0-1.0)"
    )
    device_calibration_status: Optional[str] = Field(
        default="calibrated",
        description="Device calibration status"
    )

    @validator('eeg_data')
    def validate_eeg_data_structure(cls, v):
        """Validate EEG data format and channels"""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError('EEG data must be a dictionary')
        
        # Standard 10-20 system channels
        valid_channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8'}
        
        for channel, data in v.items():
            if channel not in valid_channels:
                raise ValueError(f'Invalid EEG channel: {channel}')
            
            if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
                raise ValueError(f'EEG data for {channel} must be list of numbers')
            
            # Check for reasonable voltage values (μV)
            if any(abs(x) > 500 for x in data):
                raise ValueError(f'EEG values for {channel} outside reasonable range (±500μV)')
        
        return v

    @root_validator
    def validate_device_data_consistency(cls, values):
        """Ensure biometric data matches device capabilities"""
        device_type = values.get('device_type')
        
        # EEG devices should provide EEG data
        if device_type in ['brainbit_eeg', 'muse_eeg'] and not values.get('eeg_data'):
            raise ValueError(f'{device_type} requires EEG data')
        
        # Heart rate monitors should provide cardiovascular data
        if device_type == 'heart_rate_monitor' and not values.get('heart_rate_bpm'):
            raise ValueError('Heart rate monitor requires heart_rate_bpm')
        
        return values

class LucidTrainingRequest(BaseModel):
    """
    Request for initiating AI-guided lucid dreaming training program.
    Includes personalized technique selection and progress tracking.
    """
    dream_profile_id: int = Field(
        ...,
        description="User's dream profile identifier"
    )
    training_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Personal preferences for training approach"
    )
    experience_level: str = Field(
        default="beginner",
        regex=r'^(beginner|intermediate|advanced)$',
        description="Current lucid dreaming experience level"
    )
    training_intensity: str = Field(
        default="moderate",
        regex=r'^(light|moderate|intensive)$',
        description="Desired training intensity level"
    )
    preferred_techniques: List[str] = Field(
        default_factory=list,
        description="Preferred lucid dreaming techniques (MILD, WILD, DILD, etc.)"
    )
    
    # Training Goals
    primary_goals: List[str] = Field(
        default_factory=lambda: ["achieve_lucidity"],
        description="Primary training objectives"
    )
    timeline_weeks: int = Field(
        default=8,
        ge=4,
        le=26,
        description="Desired training timeline in weeks (4-26 weeks)"
    )
    
    # Availability and Scheduling
    training_schedule: Dict[str, Any] = Field(
        default_factory=dict,
        description="Availability and preferred training times"
    )
    daily_commitment_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Daily training commitment in minutes"
    )

    @validator('training_preferences')
    def validate_training_preferences(cls, v):
        """Validate training preference structure"""
        allowed_preferences = {
            'meditation_experience', 'dream_recall_frequency', 'sleep_schedule_flexibility',
            'technology_comfort_level', 'group_vs_individual', 'cultural_considerations'
        }
        
        if not isinstance(v, dict):
            raise ValueError('Training preferences must be a dictionary')
        
        # Validate preference values
        for key, value in v.items():
            if key not in allowed_preferences:
                raise ValueError(f'Invalid training preference: {key}')
        
        return v

# Response Schemas

class DreamProfileResponse(BaseModel):
    """
    Comprehensive dream profile with therapeutic progress and personalization data.
    """
    profile_id: int
    user_id: int
    
    # Dream Pattern Analysis
    dream_patterns: Dict[str, Any] = Field(
        ...,
        description="Personal dream patterns and sleep analysis"
    )
    
    # Therapeutic Status
    therapeutic_status: Dict[str, Any] = Field(
        ...,
        description="Current therapeutic progress and goals"
    )
    
    # Personalization Settings
    personalization: Dict[str, Any] = Field(
        ...,
        description="Personal preferences and cultural adaptations"
    )
    
    # Safety and Monitoring
    safety_status: Dict[str, Any] = Field(
        ...,
        description="Current safety monitoring status"
    )
    
    # Professional Integration
    professional_integration: Dict[str, Any] = Field(
        ...,
        description="Licensed therapist integration status"
    )
    
    # Biometric Integration
    biometric_integration: Dict[str, Any] = Field(
        ...,
        description="Biometric device integration and monitoring capabilities"
    )

class DreamSessionResponse(BaseModel):
    """
    Complete dream session record with generated content and outcomes.
    """
    session_uuid: str
    session_id: int
    
    # Session Configuration
    session_type: TherapeuticProtocolEnum
    target_dream_state: DreamStateEnum
    duration_minutes: int
    
    # Generated Content
    dream_content: Dict[str, Any] = Field(
        ...,
        description="Generated multimodal dream content"
    )
    
    # Real-time Adaptation
    adaptation_data: Dict[str, Any] = Field(
        ...,
        description="Real-time adaptation history and triggers"
    )
    
    # Therapeutic Outcomes
    therapeutic_outcomes: Dict[str, Any] = Field(
        ...,
        description="Session therapeutic effectiveness and progress"
    )
    
    # Safety Monitoring
    safety_monitoring: Dict[str, Any] = Field(
        ...,
        description="Safety monitoring summary and interventions"
    )
    
    # Professional Review
    professional_review: Dict[str, Any] = Field(
        ...,
        description="Professional therapist review status and notes"
    )

class DreamAnalysisResponse(BaseModel):
    """
    Comprehensive AI analysis of dream session with therapeutic insights.
    """
    analysis_id: str
    session_uuid: str
    
    # Therapeutic Analysis
    therapeutic_outcomes: Dict[str, Any] = Field(
        ...,
        description="Detailed therapeutic outcome analysis"
    )
    
    # Symbolic Interpretation
    symbolic_interpretation: Dict[str, Any] = Field(
        ...,
        description="Dream symbolism and psychological interpretation"
    )
    
    # Progress Assessment
    progress_assessment: Dict[str, Any] = Field(
        ...,
        description="Therapeutic progress and milestone tracking"
    )
    
    # Integration Guidance
    integration_guidance: Dict[str, Any] = Field(
        ...,
        description="Personalized integration exercises and recommendations"
    )
    
    # Professional Insights
    professional_insights: Dict[str, Any] = Field(
        ...,
        description="Professional consultation recommendations and clinical insights"
    )
    
    # Analysis Metadata
    analysis_metadata: Dict[str, Any] = Field(
        ...,
        description="Analysis configuration and confidence metrics"
    )

class TherapeuticAssessmentResponse(BaseModel):
    """
    Pre-session therapeutic assessment with safety screening and recommendations.
    """
    assessment_id: str
    user_id: int
    
    # Trauma Assessment
    trauma_assessment: Dict[str, float] = Field(
        ...,
        description="Comprehensive trauma indicator assessment"
    )
    
    # Safety Evaluation
    safety_evaluation: Dict[str, Any] = Field(
        ...,
        description="Psychological safety assessment and risk factors"
    )
    
    # Therapeutic Readiness
    therapeutic_readiness: Dict[str, Any] = Field(
        ...,
        description="Readiness for therapeutic intervention assessment"
    )
    
    # Recommendations
    session_recommendations: Dict[str, Any] = Field(
        ...,
        description="Recommended session modifications and approaches"
    )
    
    # Professional Oversight
    professional_oversight: Dict[str, Any] = Field(
        ...,
        description="Professional supervision requirements and contacts"
    )

class CrisisInterventionResponse(BaseModel):
    """
    Crisis intervention plan and emergency response information.
    """
    intervention_id: str
    crisis_level: CrisisLevelEnum
    session_uuid: str
    
    # Immediate Actions
    immediate_actions: List[str] = Field(
        ...,
        description="Immediate crisis intervention actions taken"
    )
    
    # Professional Contacts
    professional_notifications: Dict[str, Any] = Field(
        ...,
        description="Professional contacts notified and response timeline"
    )
    
    # Safety Measures
    safety_measures: Dict[str, Any] = Field(
        ...,
        description="Safety measures implemented and monitoring status"
    )
    
    # Follow-up Plan
    follow_up_plan: Dict[str, Any] = Field(
        ...,
        description="Follow-up timeline and continued care recommendations"
    )
    
    # Emergency Resources
    emergency_resources: Dict[str, Any] = Field(
        ...,
        description="Emergency contact information and crisis resources"
    )

class SafetyMonitoringResponse(BaseModel):
    """
    Real-time safety monitoring status and psychological wellness indicators.
    """
    monitoring_id: str
    session_uuid: str
    current_safety_level: CrisisLevelEnum
    
    # Real-time Status
    psychological_indicators: Dict[str, Any] = Field(
        ...,
        description="Current psychological state indicators"
    )
    
    # Risk Assessment
    risk_assessment: Dict[str, Any] = Field(
        ...,
        description="Current risk factors and protective factors"
    )
    
    # Intervention Status
    intervention_status: Dict[str, Any] = Field(
        ...,
        description="Active interventions and their effectiveness"
    )
    
    # Monitoring Timeline
    monitoring_timeline: Dict[str, Any] = Field(
        ...,
        description="Safety monitoring history and trend analysis"
    )
    
    # Professional Alert Status
    professional_alert_status: Dict[str, Any] = Field(
        ...,
        description="Professional notification status and response"
    )

# Utility schemas for complex data structures

class BiometricReading(BaseModel):
    """Individual biometric reading with full context"""
    timestamp: datetime
    device_type: BiometricDeviceEnum
    readings: Dict[str, Union[float, int, List[float]]]
    signal_quality: float = Field(ge=0.0, le=1.0)
    processing_flags: List[str] = Field(default_factory=list)

class TherapeuticContent(BaseModel):
    """Generated therapeutic content with metadata"""
    content_type: str
    content_data: Dict[str, Any]
    therapeutic_annotations: Dict[str, Any]
    safety_rating: str
    cultural_adaptations: List[str] = Field(default_factory=list)
    
class AdaptationTrigger(BaseModel):
    """Real-time content adaptation trigger"""
    trigger_type: str
    biometric_threshold: Dict[str, float]
    adaptation_instructions: Dict[str, Any]
    safety_override: bool = False