"""
Emotional Intelligence Pydantic Schemas

Comprehensive data validation and serialization schemas for the revolutionary
emotional intelligence system, including:
- Multi-modal emotion analysis
- Empathetic response generation  
- Empathy assessment and development
- Crisis detection and intervention
- Emotional profile management
- Real-time coaching systems

All schemas include proper validation, documentation, and type safety.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

from app.models.emotional_intelligence import (
    BasicEmotion, EmotionIntensity, DetectionModality,
    EmotionRegulationStrategy, CrisisLevel
)


# Base Schemas

class EmotionalDimensionsSchema(BaseModel):
    """Schema for core emotional dimensions (Valence-Arousal-Dominance model)."""
    valence: float = Field(..., ge=-1, le=1, description="Pleasant/Unpleasant dimension (-1 to 1)")
    arousal: float = Field(..., ge=-1, le=1, description="High/Low activation (-1 to 1)")
    dominance: float = Field(..., ge=-1, le=1, description="Control/Submission (-1 to 1)")
    
    @validator('valence', 'arousal', 'dominance')
    def validate_dimension_range(cls, v):
        """Ensure dimensional scores are within valid range."""
        if not -1.0 <= v <= 1.0:
            raise ValueError("Emotional dimension scores must be between -1 and 1")
        return round(v, 3)  # Round to 3 decimal places for consistency


class PlutchikEmotionScoresSchema(BaseModel):
    """Schema for Plutchik's 8 basic emotion scores."""
    joy: float = Field(..., ge=0, le=1, description="Joy emotion score")
    sadness: float = Field(..., ge=0, le=1, description="Sadness emotion score")
    anger: float = Field(..., ge=0, le=1, description="Anger emotion score")
    fear: float = Field(..., ge=0, le=1, description="Fear emotion score")
    trust: float = Field(..., ge=0, le=1, description="Trust emotion score")
    disgust: float = Field(..., ge=0, le=1, description="Disgust emotion score")
    surprise: float = Field(..., ge=0, le=1, description="Surprise emotion score")
    anticipation: float = Field(..., ge=0, le=1, description="Anticipation emotion score")
    
    @root_validator(skip_on_failure=True)
    def validate_emotion_scores_sum(cls, values):
        """Ensure emotion scores are reasonable (don't need to sum to 1 for multi-label)."""
        total = sum(values.values())
        if total > 8.0:  # Maximum possible if all emotions at 1.0
            raise ValueError("Total emotion scores seem unrealistic")
        return values


# Emotion Detection Schemas

class VoiceFeaturesSchema(BaseModel):
    """Schema for voice prosodic features used in emotion detection."""
    pitch_mean: Optional[float] = Field(None, ge=0, description="Average pitch/fundamental frequency")
    pitch_std: Optional[float] = Field(None, ge=0, description="Pitch variation/standard deviation")
    intensity_mean: Optional[float] = Field(None, ge=0, description="Average vocal intensity")
    intensity_std: Optional[float] = Field(None, ge=0, description="Intensity variation")
    speaking_rate: Optional[float] = Field(None, ge=0, description="Words/syllables per minute")
    pause_duration: Optional[float] = Field(None, ge=0, description="Average pause duration")
    jitter: Optional[float] = Field(None, ge=0, le=1, description="Pitch variability (jitter)")
    shimmer: Optional[float] = Field(None, ge=0, le=1, description="Amplitude variability (shimmer)")
    harmonics_noise_ratio: Optional[float] = Field(None, description="Voice quality measure")


class BehavioralDataSchema(BaseModel):
    """Schema for behavioral patterns used in emotion detection."""
    typing_speed: Optional[float] = Field(None, ge=0, description="Characters per minute")
    typing_rhythm: Optional[List[float]] = Field(None, description="Inter-keystroke intervals")
    pause_patterns: Optional[List[float]] = Field(None, description="Pause durations between messages")
    message_length_variance: Optional[float] = Field(None, ge=0, description="Variance in message lengths")
    response_time_seconds: Optional[float] = Field(None, ge=0, description="Time to respond to messages")
    backspace_frequency: Optional[float] = Field(None, ge=0, description="Backspace usage rate")
    emoji_usage_pattern: Optional[Dict[str, int]] = Field(None, description="Emoji usage patterns")
    activity_patterns: Optional[Dict[str, Any]] = Field(None, description="General activity patterns")


class ConversationContextSchema(BaseModel):
    """Schema for conversation context information."""
    topic: Optional[str] = Field(None, max_length=200, description="Main conversation topic")
    social_context: Optional[str] = Field(None, description="Social setting (private, group, public)")
    time_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    conversation_length: Optional[int] = Field(None, ge=0, description="Number of messages in conversation")
    message_count: Optional[int] = Field(None, ge=0, description="Total messages in session")
    participants_count: Optional[int] = Field(None, ge=1, description="Number of conversation participants")
    conversation_type: Optional[str] = Field(None, description="Type of conversation (support, casual, etc.)")
    previous_emotions: Optional[List[str]] = Field(None, description="Recent emotions in conversation")
    emotional_history: Optional[Dict[str, Any]] = Field(None, description="User's emotional history context")


class EmotionAnalysisRequestSchema(BaseModel):
    """Schema for emotion analysis requests."""
    text_content: Optional[str] = Field(
        None, 
        max_length=5000, 
        description="Text content to analyze for emotions"
    )
    voice_features: Optional[VoiceFeaturesSchema] = Field(
        None, 
        description="Voice/audio features for prosodic analysis"
    )
    behavioral_data: Optional[BehavioralDataSchema] = Field(
        None, 
        description="User behavioral patterns for emotion inference"
    )
    conversation_context: Optional[ConversationContextSchema] = Field(
        None, 
        description="Current conversation context"
    )
    include_regulation_suggestions: bool = Field(
        True, 
        description="Include personalized emotion regulation strategies"
    )
    preferred_detection_modalities: Optional[List[DetectionModality]] = Field(
        None, 
        description="Preferred emotion detection methods"
    )
    
    @validator('text_content')
    def validate_text_content(cls, v):
        """Validate text content is meaningful."""
        if v is not None and len(v.strip()) == 0:
            return None  # Treat empty strings as None
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_at_least_one_input(cls, values):
        """Ensure at least one analysis input is provided."""
        inputs = [
            values.get('text_content'),
            values.get('voice_features'),
            values.get('behavioral_data')
        ]
        if not any(inputs):
            raise ValueError("At least one input (text, voice, or behavioral data) must be provided")
        return values


class EmotionAnalysisResultSchema(BaseModel):
    """Schema for emotion analysis results."""
    dimensions: EmotionalDimensionsSchema = Field(..., description="Core emotional dimensions")
    primary_emotion: BasicEmotion = Field(..., description="Primary detected emotion")
    secondary_emotion: Optional[BasicEmotion] = Field(None, description="Secondary emotion if present")
    emotion_intensity: EmotionIntensity = Field(..., description="Intensity of primary emotion")
    plutchik_scores: PlutchikEmotionScoresSchema = Field(..., description="All 8 basic emotion scores")
    
    # Detection metadata
    confidence_scores: Dict[str, float] = Field(..., description="Confidence by detection modality")
    detection_modalities: List[DetectionModality] = Field(..., description="Methods used for detection")
    processing_time_ms: int = Field(..., ge=0, description="Analysis processing time")
    analysis_quality: float = Field(..., ge=0, le=1, description="Overall analysis quality score")
    
    # Contextual insights
    crisis_indicators: Optional[Dict[str, Any]] = Field(None, description="Crisis risk indicators if detected")
    regulation_strategies: Optional[List[EmotionRegulationStrategy]] = Field(
        None, 
        description="Suggested emotion regulation strategies"
    )
    therapeutic_insights: Optional[Dict[str, Any]] = Field(
        None, 
        description="Therapeutic insights and recommendations"
    )
    
    class Config:
        use_enum_values = True


# Empathetic Response Schemas

class TherapeuticTechniqueSchema(BaseModel):
    """Schema for therapeutic techniques applied in responses."""
    technique_name: str = Field(..., description="Name of therapeutic technique")
    technique_description: str = Field(..., description="Brief description of technique")
    evidence_base: Optional[str] = Field(None, description="Evidence base (CBT, DBT, etc.)")
    application_context: str = Field(..., description="How technique was applied")


class EmotionalIntentSchema(BaseModel):
    """Schema for intended emotional impact of responses."""
    validation: Optional[float] = Field(None, ge=0, le=1, description="Validation intent strength")
    support: Optional[float] = Field(None, ge=0, le=1, description="Support intent strength")
    empowerment: Optional[float] = Field(None, ge=0, le=1, description="Empowerment intent strength")
    calming: Optional[float] = Field(None, ge=0, le=1, description="Calming intent strength")
    hope: Optional[float] = Field(None, ge=0, le=1, description="Hope instillation strength")
    connection: Optional[float] = Field(None, ge=0, le=1, description="Connection building strength")
    understanding: Optional[float] = Field(None, ge=0, le=1, description="Understanding demonstration strength")
    safety: Optional[float] = Field(None, ge=0, le=1, description="Safety assurance strength")


class EmpatheticResponseRequestSchema(BaseModel):
    """Schema for empathetic response generation requests."""
    user_message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="User's message to respond to"
    )
    emotion_analysis: Optional[EmotionAnalysisResultSchema] = Field(
        None, 
        description="Pre-computed emotion analysis"
    )
    conversation_context: Optional[ConversationContextSchema] = Field(
        None, 
        description="Current conversation context"
    )
    response_constraints: Optional[Dict[str, Any]] = Field(
        None, 
        description="Constraints for response generation"
    )
    personalization_level: Optional[int] = Field(
        3, 
        ge=1, 
        le=5, 
        description="Level of personalization (1=generic, 5=highly personalized)"
    )
    include_coaching_elements: bool = Field(
        True, 
        description="Include empathy coaching elements"
    )
    
    @validator('user_message')
    def validate_user_message(cls, v):
        """Validate user message is meaningful."""
        if not v or len(v.strip()) == 0:
            raise ValueError("User message cannot be empty")
        return v.strip()


class EmpatheticResponseResultSchema(BaseModel):
    """Schema for empathetic response generation results."""
    response_text: str = Field(..., description="Generated empathetic response")
    response_style: str = Field(..., description="Applied response style")
    
    # Therapeutic elements
    therapeutic_techniques: List[TherapeuticTechniqueSchema] = Field(
        ..., 
        description="Therapeutic techniques incorporated"
    )
    emotional_intent: EmotionalIntentSchema = Field(
        ..., 
        description="Intended emotional impact"
    )
    
    # Personalization and effectiveness
    personalization_applied: bool = Field(..., description="Whether personalization was applied")
    confidence_score: float = Field(..., ge=0, le=1, description="Response generation confidence")
    expected_effectiveness: float = Field(..., ge=0, le=1, description="Expected response effectiveness")
    
    # Additional support
    regulation_strategies: List[EmotionRegulationStrategy] = Field(
        ..., 
        description="Recommended emotion regulation strategies"
    )
    followup_suggestions: List[str] = Field(
        ..., 
        description="Suggested followup questions or topics"
    )
    crisis_level: CrisisLevel = Field(..., description="Assessed crisis level")
    
    # Metadata
    processing_time_ms: int = Field(..., ge=0, description="Response generation time")
    model_version: Optional[str] = Field(None, description="AI model version used")
    
    class Config:
        use_enum_values = True


# Empathy Assessment Schemas

class EmpathyDimensionScoresSchema(BaseModel):
    """Schema for multi-dimensional empathy scores."""
    cognitive_empathy: float = Field(..., ge=0, le=100, description="Understanding others' emotions")
    affective_empathy: float = Field(..., ge=0, le=100, description="Sharing others' emotions")
    compassionate_empathy: float = Field(..., ge=0, le=100, description="Taking action to help")
    perspective_taking: float = Field(..., ge=0, le=100, description="Seeing from others' viewpoints")
    empathic_concern: float = Field(..., ge=0, le=100, description="Other-oriented concern")
    personal_distress: float = Field(..., ge=0, le=100, description="Distress from others' suffering")
    fantasy: float = Field(..., ge=0, le=100, description="Emotional investment in fiction")
    
    @validator('*')
    def validate_percentage_scores(cls, v):
        """Ensure all scores are valid percentages."""
        if not 0 <= v <= 100:
            raise ValueError("Empathy dimension scores must be between 0 and 100")
        return round(v, 1)


class EmpathyAssessmentRequestSchema(BaseModel):
    """Schema for empathy assessment requests."""
    assessment_type: str = Field(
        "comprehensive", 
        description="Type of assessment (initial, comprehensive, behavioral)"
    )
    include_behavioral_analysis: bool = Field(
        True, 
        description="Include behavioral pattern analysis"
    )
    include_conversation_analysis: bool = Field(
        True, 
        description="Include conversation history analysis"
    )
    include_self_report: bool = Field(
        False, 
        description="Include self-report questionnaire"
    )
    time_period_days: Optional[int] = Field(
        30, 
        ge=7, 
        le=365, 
        description="Days of history to analyze"
    )


class EmpathyAssessmentResultSchema(BaseModel):
    """Schema for empathy assessment results."""
    overall_empathy_quotient: float = Field(..., ge=0, le=80, description="Baron-Cohen EQ score")
    dimension_scores: EmpathyDimensionScoresSchema = Field(..., description="Multi-dimensional scores")
    
    # Assessment metadata
    assessment_confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    assessment_method: str = Field(..., description="Primary assessment methodology")
    data_sources: List[str] = Field(..., description="Data sources used for assessment")
    
    # Development insights
    strengths: List[str] = Field(..., description="Identified empathy strengths")
    improvement_areas: List[str] = Field(..., description="Areas for empathy development")
    coaching_recommendations: List[str] = Field(..., description="Recommended coaching interventions")
    
    # Comparison and context
    peer_comparison: Optional[Dict[str, float]] = Field(
        None, 
        description="Comparison to peer group averages"
    )
    growth_potential: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Estimated growth potential"
    )
    
    # Detailed insights
    behavioral_indicators: Optional[Dict[str, Any]] = Field(
        None, 
        description="Specific behavioral empathy indicators observed"
    )
    conversation_analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Results from conversation empathy analysis"
    )


# Emotional Profile Schemas

class EmotionalProfileSchema(BaseModel):
    """Schema for comprehensive emotional profile."""
    user_id: str = Field(..., description="User identifier")
    
    # Baseline emotional characteristics
    baseline_dimensions: Optional[EmotionalDimensionsSchema] = Field(
        None, 
        description="User's baseline emotional dimensions"
    )
    dominant_emotions: Optional[Dict[str, float]] = Field(
        None, 
        description="Most frequently experienced emotions"
    )
    emotion_patterns: Optional[Dict[str, Any]] = Field(
        None, 
        description="Identified emotional patterns and cycles"
    )
    
    # Emotional intelligence metrics
    empathy_quotient: Optional[float] = Field(None, ge=0, le=80, description="Baron-Cohen EQ score")
    emotional_quotient: Optional[float] = Field(None, ge=0, le=200, description="Mayer-Salovey EQ score")
    alexithymia_score: Optional[float] = Field(None, ge=20, le=100, description="TAS-20 score")
    
    # Regulation and coping
    primary_regulation_strategies: Optional[List[EmotionRegulationStrategy]] = Field(
        None, 
        description="User's primary emotion regulation strategies"
    )
    regulation_effectiveness: Optional[Dict[str, float]] = Field(
        None, 
        description="Effectiveness scores for different strategies"
    )
    
    # Personality and attachment
    attachment_style: Optional[str] = Field(None, description="Attachment style classification")
    personality_correlations: Optional[Dict[str, float]] = Field(
        None, 
        description="Big Five personality correlations"
    )
    
    # Social and crisis information
    social_empathy_patterns: Optional[Dict[str, Any]] = Field(
        None, 
        description="Social emotional interaction patterns"
    )
    crisis_indicators: Optional[Dict[str, Any]] = Field(
        None, 
        description="Personal crisis warning signs"
    )
    support_preferences: Optional[Dict[str, Any]] = Field(
        None, 
        description="Preferred support types and approaches"
    )
    
    # Privacy and consent
    sharing_consent_level: int = Field(1, ge=1, le=5, description="Data sharing consent level")
    crisis_intervention_consent: bool = Field(True, description="Consent for crisis intervention")
    
    # Metadata
    profile_completeness: Optional[float] = Field(None, ge=0, le=1, description="Profile completeness score")
    last_updated: Optional[datetime] = Field(None, description="Last profile update timestamp")
    
    class Config:
        use_enum_values = True


class EmotionalProfileUpdateSchema(BaseModel):
    """Schema for updating emotional profile."""
    primary_regulation_strategies: Optional[List[EmotionRegulationStrategy]] = None
    attachment_style: Optional[str] = None
    sharing_consent_level: Optional[int] = Field(None, ge=1, le=5)
    crisis_intervention_consent: Optional[bool] = None
    support_preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


# Development Tracking Schemas

class EmpathyDevelopmentProgressSchema(BaseModel):
    """Schema for empathy development progress tracking."""
    user_id: str = Field(..., description="User identifier")
    tracking_period_days: int = Field(..., ge=1, description="Period tracked in days")
    
    # Progress metrics
    overall_improvement: float = Field(..., description="Overall improvement percentage")
    dimension_improvements: Dict[str, float] = Field(..., description="Improvement by empathy dimension")
    skill_development_rates: Dict[str, float] = Field(..., description="Rate of skill development")
    
    # Intervention effectiveness
    most_effective_interventions: List[str] = Field(..., description="Most effective coaching interventions")
    intervention_success_rates: Dict[str, float] = Field(..., description="Success rates by intervention type")
    
    # Milestones and achievements
    milestones_achieved: List[str] = Field(..., description="Empathy development milestones reached")
    next_milestone_goals: List[str] = Field(..., description="Next development goals")
    estimated_timeline: Optional[Dict[str, int]] = Field(None, description="Estimated timeline for goals")
    
    # Behavioral evidence
    behavioral_improvements: Optional[Dict[str, Any]] = Field(
        None, 
        description="Observable behavioral improvements"
    )
    peer_feedback_trends: Optional[Dict[str, Any]] = Field(
        None, 
        description="Trends in peer feedback if available"
    )
    
    # Recommendations
    development_recommendations: List[str] = Field(..., description="Personalized development recommendations")
    focus_areas: List[str] = Field(..., description="Areas to focus on next")


# System Health and Metrics Schemas

class EmotionalIntelligenceSystemMetricsSchema(BaseModel):
    """Schema for system-wide emotional intelligence metrics."""
    
    # Emotion detection metrics
    emotion_detection: Dict[str, Any] = Field(..., description="Emotion detection performance")
    
    # Response generation metrics  
    response_generation: Dict[str, Any] = Field(..., description="Response generation effectiveness")
    
    # Empathy development metrics
    empathy_development: Dict[str, Any] = Field(..., description="Empathy coaching effectiveness")
    
    # Crisis intervention metrics
    crisis_interventions: Dict[str, Any] = Field(..., description="Crisis detection and intervention stats")
    
    # Overall system health
    system_status: str = Field(..., description="Overall system operational status")
    performance_score: float = Field(..., ge=0, le=1, description="Overall performance score")
    timestamp: datetime = Field(..., description="Metrics timestamp")


# Error and Validation Schemas

class EmotionalIntelligenceErrorSchema(BaseModel):
    """Schema for emotional intelligence system errors."""
    error_type: str = Field(..., description="Type of error encountered")
    error_message: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Specific error code")
    component: str = Field(..., description="System component where error occurred")
    user_impact: str = Field(..., description="Impact on user experience")
    suggested_action: Optional[str] = Field(None, description="Suggested user action")
    support_available: bool = Field(True, description="Whether support resources are available")


# Export all schemas for easy importing
__all__ = [
    'EmotionalDimensionsSchema',
    'PlutchikEmotionScoresSchema',
    'VoiceFeaturesSchema',
    'BehavioralDataSchema',
    'ConversationContextSchema',
    'EmotionAnalysisRequestSchema',
    'EmotionAnalysisResultSchema',
    'TherapeuticTechniqueSchema',
    'EmotionalIntentSchema',
    'EmpatheticResponseRequestSchema',
    'EmpatheticResponseResultSchema',
    'EmpathyDimensionScoresSchema',
    'EmpathyAssessmentRequestSchema',
    'EmpathyAssessmentResultSchema',
    'EmotionalProfileSchema',
    'EmotionalProfileUpdateSchema',
    'EmpathyDevelopmentProgressSchema',
    'EmotionalIntelligenceSystemMetricsSchema',
    'EmotionalIntelligenceErrorSchema'
]