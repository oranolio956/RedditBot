"""
Digital Synesthesia Engine - Pydantic Schemas

Request/response schemas for synesthetic translation API endpoints.
Comprehensive data validation for multi-modal sensory experiences.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum


class ModalityTypeSchema(str, Enum):
    """Supported sensory modalities for synesthetic translation"""
    audio = "audio"
    visual = "visual" 
    haptic = "haptic"
    text = "text"
    emotion = "emotion"
    spatial = "spatial"
    temporal = "temporal"


class HapticDeviceTypeSchema(str, Enum):
    """Supported haptic device types"""
    ultraleap = "ultraleap"
    tanvas = "tanvas"
    vibration = "vibration"
    pneumatic = "pneumatic"
    electrotactile = "electrotactile"
    thermal = "thermal"


class SynestheticTranslationRequest(BaseModel):
    """Request for cross-modal synesthetic translation"""
    
    input_modality: ModalityTypeSchema = Field(..., description="Source modality type")
    target_modalities: List[ModalityTypeSchema] = Field(..., description="Desired output modalities")
    
    # Input data (flexible structure for different modalities)
    input_data: Dict[str, Any] = Field(..., description="Input stimulus data")
    
    # Optional parameters
    user_profile_id: Optional[str] = Field(None, description="User's synesthetic profile ID")
    intensity_factor: float = Field(1.0, ge=0.1, le=2.0, description="Intensity scaling factor")
    target_device: Optional[HapticDeviceTypeSchema] = Field(None, description="Target haptic device")
    include_spatial: bool = Field(True, description="Include 3D spatial visualization")
    real_time: bool = Field(False, description="Real-time streaming mode")
    
    # Quality preferences
    translation_quality: Literal["fast", "balanced", "high_quality"] = Field("balanced")
    authenticity_mode: bool = Field(True, description="Use research-based authentic mappings")
    
    class Config:
        schema_extra = {
            "example": {
                "input_modality": "audio",
                "target_modalities": ["visual", "haptic"],
                "input_data": {
                    "waveform": [0.1, 0.2, -0.1, 0.3],
                    "sample_rate": 22050,
                    "duration": 2.0
                },
                "intensity_factor": 1.2,
                "target_device": "vibration",
                "include_spatial": True
            }
        }


class ColorInfo(BaseModel):
    """Color representation with multiple formats"""
    
    hue: float = Field(..., ge=0, le=360, description="Hue in degrees (0-360)")
    saturation: float = Field(..., ge=0, le=1, description="Saturation (0-1)")
    brightness: float = Field(..., ge=0, le=1, description="Brightness/value (0-1)")
    
    rgb: Dict[str, int] = Field(..., description="RGB values (0-255)")
    hex: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$', description="Hex color code")
    
    # Optional metadata
    intensity_factor: Optional[float] = Field(None, description="Intensity scaling applied")
    emotion_source: Optional[str] = Field(None, description="Source emotion if applicable")
    frequency_source: Optional[float] = Field(None, description="Source frequency if applicable")


class VisualPattern(BaseModel):
    """Visual pattern description for synesthetic visualization"""
    
    colors: List[ColorInfo] = Field(..., description="Color palette")
    
    # Pattern characteristics
    pattern_type: str = Field(..., description="Type of visual pattern")
    movement_type: str = Field("static", description="Type of movement/animation")
    
    # Spatial properties
    shapes: List[Dict[str, Any]] = Field(default_factory=list, description="Geometric shapes")
    spatial_layout: Dict[str, Any] = Field(default_factory=dict, description="3D spatial arrangement")
    
    # Temporal properties
    durations: List[float] = Field(default_factory=list, description="Element durations in ms")
    intensities: List[float] = Field(default_factory=list, description="Element intensities")
    
    # Animation properties
    motion_vectors: List[Dict[str, float]] = Field(default_factory=list, description="Movement vectors")
    synchronization_points: List[float] = Field(default_factory=list, description="Sync timestamps")


class HapticSensation(BaseModel):
    """Single haptic sensation specification"""
    
    primitive_type: str = Field(..., description="Type of haptic primitive")
    intensity: float = Field(..., ge=0, le=1, description="Sensation intensity (0-1)")
    
    # Temporal properties
    duration_ms: float = Field(..., gt=0, description="Duration in milliseconds")
    fade_in_ms: float = Field(0.0, ge=0, description="Fade in time")
    fade_out_ms: float = Field(0.0, ge=0, description="Fade out time")
    
    # Frequency/oscillation (for vibration)
    frequency_hz: Optional[float] = Field(None, gt=0, description="Oscillation frequency")
    waveform: str = Field("sine", description="Waveform type")
    
    # Spatial properties
    spatial_position: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.5, "y": 0.5, "z": 0.0},
        description="3D position (normalized 0-1)"
    )


class HapticSequence(BaseModel):
    """Complete haptic sensation sequence"""
    
    sensations: List[HapticSensation] = Field(..., description="List of haptic sensations")
    total_duration_ms: float = Field(..., gt=0, description="Total sequence duration")
    
    # Sequence properties
    loop_count: int = Field(1, ge=1, description="Number of loops")
    synchronization_points: List[float] = Field(default_factory=list, description="Key timing points")
    
    # Device-specific rendering
    device_format: Optional[Dict[str, Any]] = Field(None, description="Device-specific format")


class EmotionalState(BaseModel):
    """Emotional state representation"""
    
    primary_emotion: str = Field(..., description="Primary detected emotion")
    intensity: float = Field(..., ge=0, le=1, description="Emotional intensity")
    
    # Secondary emotions
    secondary_emotions: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list, 
        description="Secondary emotions with intensities"
    )
    
    # Dimensional model (PAD - Pleasure, Arousal, Dominance)
    valence: float = Field(0.0, ge=-1, le=1, description="Emotional valence (negative to positive)")
    arousal: float = Field(0.5, ge=0, le=1, description="Emotional arousal (calm to excited)")
    dominance: float = Field(0.0, ge=-1, le=1, description="Emotional dominance (submissive to dominant)")
    
    # Temporal characteristics
    temporal_dynamics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Temporal evolution characteristics"
    )


class SpatialEnvironment(BaseModel):
    """3D spatial environment for synesthetic experience"""
    
    spatial_metaphor: str = Field(..., description="Spatial metaphor type")
    
    # Environment elements
    environmental_elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="3D objects and their properties"
    )
    
    # Lighting and atmosphere
    lighting_scheme: Dict[str, Any] = Field(
        default_factory=dict,
        description="Lighting configuration"
    )
    
    particle_systems: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Particle effect configurations"
    )
    
    atmospheric_effects: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fog, wind, and atmospheric effects"
    )
    
    # Spatial dimensions
    dimensions: Dict[str, float] = Field(
        default_factory=lambda: {"width": 10.0, "height": 10.0, "depth": 10.0},
        description="Environment bounding dimensions"
    )


class SynestheticTranslationResponse(BaseModel):
    """Response from synesthetic translation"""
    
    # Input information
    input_modality: ModalityTypeSchema
    target_modalities: List[ModalityTypeSchema]
    
    # Translation results
    visual_output: Optional[VisualPattern] = Field(None, description="Visual synesthetic patterns")
    haptic_output: Optional[HapticSequence] = Field(None, description="Haptic sensation sequence")  
    spatial_output: Optional[SpatialEnvironment] = Field(None, description="3D spatial environment")
    emotional_output: Optional[EmotionalState] = Field(None, description="Detected emotional state")
    
    # Quality metrics
    translation_confidence: float = Field(..., ge=0, le=1, description="Translation confidence score")
    processing_time_ms: float = Field(..., gt=0, description="Processing time in milliseconds")
    authenticity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Authenticity scores for different synesthetic types"
    )
    
    # Neural activation simulation
    neural_activation_pattern: Optional[Dict[str, float]] = Field(
        None, 
        description="Simulated brain region activations"
    )
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TextSynesthesiaRequest(BaseModel):
    """Request for text-to-synesthetic translation"""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Input text content")
    
    # Synesthetic translation types
    synesthesia_types: List[str] = Field(
        default_factory=lambda: ["grapheme_color", "lexical_gustatory"],
        description="Types of text synesthesia to generate"
    )
    
    # Optional parameters
    user_profile_id: Optional[str] = Field(None, description="User's synesthetic profile ID")
    include_spatial_layout: bool = Field(True, description="Include 3D semantic spatial layout")
    include_emotional_analysis: bool = Field(True, description="Include emotional synesthesia")
    
    # Processing options
    language: str = Field("en", description="Text language for processing")
    intensity_factor: float = Field(1.0, ge=0.1, le=2.0, description="Intensity scaling")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "The bright sun filled me with joy",
                "synesthesia_types": ["grapheme_color", "emotional_visual", "semantic_spatial"],
                "include_spatial_layout": True,
                "intensity_factor": 1.2
            }
        }


class GraphemeColor(BaseModel):
    """Individual character color mapping"""
    
    character: str = Field(..., min_length=1, max_length=1, description="Single character")
    color: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$', description="Hex color")
    
    hue: float = Field(..., ge=0, le=360)
    saturation: float = Field(..., ge=0, le=1) 
    brightness: float = Field(..., ge=0, le=1)
    
    personality_traits: List[str] = Field(default_factory=list, description="Character personality")


class WordTaste(BaseModel):
    """Word taste and texture profile"""
    
    word: str = Field(..., description="Word text")
    
    # Taste characteristics
    primary_taste: str = Field(..., description="Primary taste (sweet, sour, bitter, salty, umami)")
    intensity: float = Field(..., ge=0, le=1, description="Taste intensity")
    
    # Texture characteristics  
    texture: str = Field(..., description="Tactile texture description")
    temperature: str = Field(..., description="Temperature sensation")
    complexity: float = Field(..., ge=0, le=1, description="Taste complexity")
    
    # Emotional associations
    emotional_valence: float = Field(..., ge=-1, le=1, description="Emotional association")


class TextSynesthesiaResponse(BaseModel):
    """Response from text synesthetic translation"""
    
    # Input text info
    input_text: str
    text_length: int = Field(..., gt=0)
    word_count: int = Field(..., ge=0)
    
    # Grapheme-color synesthesia
    grapheme_colors: Optional[Dict[str, Any]] = Field(None, description="Character color mappings")
    
    # Lexical-gustatory synesthesia  
    word_tastes: Optional[Dict[str, Any]] = Field(None, description="Word taste profiles")
    
    # Semantic spatial mapping
    semantic_space: Optional[Dict[str, Any]] = Field(None, description="3D semantic layout")
    
    # Emotional visual patterns
    emotional_visuals: Optional[Dict[str, Any]] = Field(None, description="Emotion-based visuals")
    
    # Phoneme color mapping
    phoneme_colors: Optional[Dict[str, Any]] = Field(None, description="Sound-to-color mapping")
    
    # Word personality mapping
    word_personalities: Optional[Dict[str, Any]] = Field(None, description="Word character traits")
    
    # Processing metadata
    processing_time_ms: float = Field(..., gt=0)
    synesthesia_types: List[str] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AudioSynesthesiaRequest(BaseModel):
    """Request for audio-to-synesthetic translation"""
    
    # Audio data (base64 encoded or file reference)
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    audio_file_path: Optional[str] = Field(None, description="Server file path")
    
    # Audio parameters
    sample_rate: int = Field(22050, gt=0, description="Audio sample rate")
    duration_seconds: Optional[float] = Field(None, gt=0, description="Audio duration")
    
    # Translation options
    target_modalities: List[ModalityTypeSchema] = Field(
        default_factory=lambda: ["visual"],
        description="Target synesthetic modalities"
    )
    
    # Device-specific options
    haptic_device: Optional[HapticDeviceTypeSchema] = Field(None, description="Target haptic device")
    
    # Quality settings
    real_time_processing: bool = Field(False, description="Real-time streaming mode")
    latency_target_ms: float = Field(180.0, gt=0, description="Target processing latency")
    
    @validator('audio_data', 'audio_url', 'audio_file_path')
    def validate_audio_source(cls, v, values):
        """Ensure at least one audio source is provided"""
        audio_sources = [
            values.get('audio_data'),
            values.get('audio_url'), 
            v if v else None
        ]
        if not any(audio_sources):
            raise ValueError("At least one audio source must be provided")
        return v


class EmotionSynesthesiaRequest(BaseModel):
    """Request for emotion-to-synesthetic translation"""
    
    # Emotion input (text analysis or direct emotion specification)
    input_text: Optional[str] = Field(None, description="Text for emotion analysis")
    direct_emotion: Optional[str] = Field(None, description="Direct emotion specification")
    emotion_intensity: float = Field(0.5, ge=0, le=1, description="Emotion intensity")
    
    # Additional emotional context
    secondary_emotions: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Secondary emotions with intensities"
    )
    
    # Output preferences
    include_color_palette: bool = Field(True, description="Generate color palette")
    include_spatial_environment: bool = Field(True, description="Generate 3D environment")
    include_haptic_patterns: bool = Field(False, description="Generate haptic feedback")
    
    # Personalization
    user_profile_id: Optional[str] = Field(None, description="User synesthetic profile")
    cultural_context: Optional[str] = Field(None, description="Cultural color associations")
    
    @validator('input_text', 'direct_emotion')
    def validate_emotion_source(cls, v, values):
        """Ensure at least one emotion source is provided"""
        if not values.get('input_text') and not v:
            raise ValueError("Either input_text or direct_emotion must be provided")
        return v


class SynestheticProfileRequest(BaseModel):
    """Request to create or update synesthetic profile"""
    
    # User identification
    user_id: str = Field(..., description="User identifier")
    
    # Synesthetic preferences
    color_intensity: float = Field(0.7, ge=0, le=1, description="Color intensity preference")
    texture_sensitivity: float = Field(0.6, ge=0, le=1, description="Haptic sensitivity")  
    motion_amplitude: float = Field(0.8, ge=0, le=1, description="Motion intensity preference")
    
    # Preferred color palette
    preferred_colors: List[str] = Field(
        default_factory=lambda: ["#FF6B6B", "#4ECDC4", "#45B7D1"],
        description="Preferred color palette (hex codes)"
    )
    
    # Feature preferences
    haptic_feedback_enabled: bool = Field(True, description="Enable haptic feedback")
    spatial_audio_enabled: bool = Field(True, description="Enable spatial audio")
    
    # Learning and adaptation
    adaptation_rate: float = Field(0.1, ge=0, le=1, description="Learning adaptation rate")
    
    @validator('preferred_colors')
    def validate_hex_colors(cls, v):
        """Validate hex color format"""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        for color in v:
            if not hex_pattern.match(color):
                raise ValueError(f"Invalid hex color format: {color}")
        return v


class SynestheticProfileResponse(BaseModel):
    """Response with synesthetic profile information"""
    
    profile_id: str = Field(..., description="Profile identifier")
    user_id: str = Field(..., description="User identifier")
    
    # Current preferences
    color_intensity: float
    texture_sensitivity: float
    motion_amplitude: float
    preferred_colors: List[str]
    
    # Status information
    calibration_complete: bool = Field(..., description="Whether calibration is finished")
    learning_sessions: int = Field(..., description="Number of learning sessions")
    
    # Performance metrics
    translation_accuracy: Optional[float] = Field(None, description="Translation accuracy score")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction rating")
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CalibrationRequest(BaseModel):
    """Request for synesthetic calibration session"""
    
    profile_id: str = Field(..., description="Synesthetic profile ID")
    calibration_type: str = Field(..., description="Type of calibration (audio_visual, text_color, etc)")
    
    # Stimulus set for calibration
    stimulus_data: List[Dict[str, Any]] = Field(..., description="Calibration stimuli")
    
    # User responses
    user_responses: List[Dict[str, Any]] = Field(..., description="User synesthetic responses")
    
    # Session metadata
    session_duration_seconds: float = Field(..., gt=0, description="Session duration")
    completion_rate: float = Field(..., ge=0, le=1, description="Completion percentage")


class PerformanceStats(BaseModel):
    """Performance statistics for synesthetic engines"""
    
    total_translations: int = Field(..., ge=0, description="Total translations performed")
    average_latency_ms: float = Field(..., gt=0, description="Average processing time")
    target_latency_ms: float = Field(..., gt=0, description="Target processing time") 
    success_rate: float = Field(..., ge=0, le=1, description="Translation success rate")
    
    performance_status: Literal["optimal", "degraded", "critical"] = Field(
        ..., description="Overall performance status"
    )
    
    # Additional metrics
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    error_rate: Optional[float] = Field(None, ge=0, le=1, description="Error rate")
    
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response for synesthetic services"""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service health status")
    
    # Component status
    synesthetic_engine: bool = Field(..., description="Core engine status") 
    audio_translator: bool = Field(..., description="Audio-visual translator status")
    text_processor: bool = Field(..., description="Text synesthesia processor status")
    haptic_synthesizer: bool = Field(..., description="Haptic synthesizer status")
    emotion_detector: bool = Field(..., description="Emotion detection status")
    
    # Performance indicators
    response_time_ms: float = Field(..., gt=0, description="Health check response time")
    memory_usage_mb: float = Field(..., gt=0, description="Current memory usage")
    
    # Error information
    errors: List[str] = Field(default_factory=list, description="Current error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }