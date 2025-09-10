"""
Digital Synesthesia Engine - Database Models

Revolutionary AI system for cross-modal sensory translation.
Research foundation: Neural synesthesia mechanisms, VR/AR integration, real-time processing.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, Boolean, 
    ForeignKey, JSON, LargeBinary, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.database.base_class import Base


class SynestheticProfile(Base):
    """User's personalized synesthetic patterns and preferences"""
    __tablename__ = "synesthetic_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Core synesthetic mappings
    audio_visual_mappings = Column(JSONB, nullable=False, default=dict)
    text_color_mappings = Column(JSONB, nullable=False, default=dict)
    emotion_texture_mappings = Column(JSONB, nullable=False, default=dict)
    
    # Personalized intensity preferences (0.0-1.0)
    color_intensity = Column(Float, nullable=False, default=0.7)
    texture_sensitivity = Column(Float, nullable=False, default=0.6)
    motion_amplitude = Column(Float, nullable=False, default=0.8)
    
    # Cross-modal preferences
    preferred_color_palette = Column(ARRAY(String), nullable=False, default=lambda: ["#FF6B6B", "#4ECDC4", "#45B7D1"])
    haptic_feedback_enabled = Column(Boolean, nullable=False, default=True)
    spatial_audio_enabled = Column(Boolean, nullable=False, default=True)
    
    # Learning and adaptation
    adaptation_rate = Column(Float, nullable=False, default=0.1)
    learning_sessions = Column(Integer, nullable=False, default=0)
    calibration_complete = Column(Boolean, nullable=False, default=False)
    
    # Performance metrics
    translation_accuracy = Column(Float, nullable=True)
    user_satisfaction_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="synesthetic_profile")
    translations = relationship("SynestheticTranslation", back_populates="profile")
    experiences = relationship("SynestheticExperience", back_populates="profile")

    __table_args__ = (
        Index('idx_synesthetic_profile_user_id', 'user_id'),
        Index('idx_synesthetic_profile_calibration', 'calibration_complete'),
    )


class SynestheticTranslation(Base):
    """Real-time cross-modal translation sessions and results"""
    __tablename__ = "synesthetic_translations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("synesthetic_profiles.id"), nullable=False)
    
    # Input modality
    input_type = Column(String(50), nullable=False)  # 'audio', 'text', 'emotion', 'haptic'
    input_data = Column(JSONB, nullable=False)
    input_metadata = Column(JSONB, nullable=False, default=dict)
    
    # Output modality  
    output_type = Column(String(50), nullable=False)  # 'visual', 'haptic', 'audio', 'spatial'
    output_data = Column(JSONB, nullable=False)
    output_metadata = Column(JSONB, nullable=False, default=dict)
    
    # Translation quality metrics
    processing_latency_ms = Column(Float, nullable=False)
    translation_confidence = Column(Float, nullable=False)
    neural_activation_pattern = Column(JSONB, nullable=True)
    
    # Synesthetic authenticity scoring
    chromesthesia_score = Column(Float, nullable=True)  # Sound->Color
    lexical_gustatory_score = Column(Float, nullable=True)  # Words->Taste
    spatial_sequence_score = Column(Float, nullable=True)  # Numbers->Space
    
    # User feedback
    user_rating = Column(Float, nullable=True)
    subjective_quality = Column(String(20), nullable=True)  # 'excellent', 'good', 'poor'
    
    # Session context
    session_id = Column(String(100), nullable=True)
    device_type = Column(String(50), nullable=True)  # 'vr', 'ar', 'mobile', 'desktop'
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    profile = relationship("SynestheticProfile", back_populates="translations")
    
    __table_args__ = (
        Index('idx_translation_profile_id', 'profile_id'),
        Index('idx_translation_input_type', 'input_type'),
        Index('idx_translation_output_type', 'output_type'),
        Index('idx_translation_latency', 'processing_latency_ms'),
        Index('idx_translation_created_at', 'created_at'),
    )


class CrossModalMapping(Base):
    """Learned associations between different sensory modalities"""
    __tablename__ = "cross_modal_mappings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Source and target modalities
    source_modality = Column(String(50), nullable=False)
    target_modality = Column(String(50), nullable=False)
    
    # Mapping definition
    source_pattern = Column(JSONB, nullable=False)  # Input pattern characteristics
    target_pattern = Column(JSONB, nullable=False)  # Output pattern characteristics
    
    # Neural network weights for this mapping
    transformation_weights = Column(LargeBinary, nullable=True)
    embedding_vector = Column(ARRAY(Float), nullable=True)
    
    # Statistical measures
    confidence_score = Column(Float, nullable=False)
    usage_frequency = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=1.0)
    
    # Population vs individual mapping
    is_universal = Column(Boolean, nullable=False, default=False)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("synesthetic_profiles.id"), nullable=True)
    
    # Research validation
    scientific_basis = Column(String(500), nullable=True)
    research_citations = Column(ARRAY(String), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    profile = relationship("SynestheticProfile", foreign_keys=[profile_id])
    
    __table_args__ = (
        Index('idx_cross_modal_source', 'source_modality'),
        Index('idx_cross_modal_target', 'target_modality'),
        Index('idx_cross_modal_confidence', 'confidence_score'),
        Index('idx_cross_modal_universal', 'is_universal'),
        UniqueConstraint('source_modality', 'target_modality', 'profile_id', 
                        name='uq_cross_modal_mapping'),
    )


class SynestheticExperience(Base):
    """Complete multi-sensory synesthetic experiences and sessions"""
    __tablename__ = "synesthetic_experiences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("synesthetic_profiles.id"), nullable=False)
    
    # Experience metadata
    experience_name = Column(String(200), nullable=False)
    experience_type = Column(String(50), nullable=False)  # 'music_visualization', 'text_landscape', 'emotion_journey'
    description = Column(Text, nullable=True)
    
    # Multi-modal configuration
    active_modalities = Column(ARRAY(String), nullable=False)  # ['visual', 'haptic', 'audio']
    modality_intensities = Column(JSONB, nullable=False)
    cross_modal_interactions = Column(JSONB, nullable=False)
    
    # Immersive environment settings
    vr_environment_config = Column(JSONB, nullable=True)
    spatial_audio_config = Column(JSONB, nullable=True)
    haptic_feedback_config = Column(JSONB, nullable=True)
    
    # Session data
    duration_seconds = Column(Float, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Quality metrics
    immersion_score = Column(Float, nullable=True)  # 0.0-1.0
    synesthetic_authenticity = Column(Float, nullable=True)  # 0.0-1.0
    user_engagement_score = Column(Float, nullable=True)  # 0.0-1.0
    
    # Physiological measurements (if available)
    heart_rate_variability = Column(JSONB, nullable=True)
    eeg_data_summary = Column(JSONB, nullable=True)
    galvanic_skin_response = Column(JSONB, nullable=True)
    
    # Social features
    is_shareable = Column(Boolean, nullable=False, default=False)
    shared_with_users = Column(ARRAY(UUID), nullable=True)
    public_rating = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    profile = relationship("SynestheticProfile", back_populates="experiences")
    haptic_patterns = relationship("HapticPattern", back_populates="experience")
    
    __table_args__ = (
        Index('idx_experience_profile_id', 'profile_id'),
        Index('idx_experience_type', 'experience_type'),
        Index('idx_experience_shareable', 'is_shareable'),
        Index('idx_experience_created_at', 'created_at'),
    )


class HapticPattern(Base):
    """Tactile feedback patterns and intensities for synesthetic experiences"""
    __tablename__ = "haptic_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experience_id = Column(UUID(as_uuid=True), ForeignKey("synesthetic_experiences.id"), nullable=True)
    
    # Pattern identification
    pattern_name = Column(String(200), nullable=False)
    pattern_type = Column(String(50), nullable=False)  # 'vibration', 'texture', 'temperature', 'pressure'
    
    # Haptic parameters
    frequency_hz = Column(Float, nullable=True)  # For vibration patterns
    amplitude = Column(Float, nullable=False)  # 0.0-1.0
    duration_ms = Column(Float, nullable=False)
    
    # Spatial distribution (for multi-point haptic devices)
    contact_points = Column(JSONB, nullable=False)  # Array of {x, y, intensity}
    
    # Temporal evolution
    intensity_envelope = Column(ARRAY(Float), nullable=False)  # Intensity over time
    frequency_modulation = Column(ARRAY(Float), nullable=True)  # Frequency changes
    
    # Synesthetic mapping
    source_stimulus = Column(JSONB, nullable=False)  # What triggered this haptic pattern
    mapping_confidence = Column(Float, nullable=False)
    
    # Device compatibility
    compatible_devices = Column(ARRAY(String), nullable=False)  # ['ultrahaptics', 'tanvas', 'ultraleap']
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    experience = relationship("SynestheticExperience", back_populates="haptic_patterns")
    
    __table_args__ = (
        Index('idx_haptic_pattern_type', 'pattern_type'),
        Index('idx_haptic_pattern_experience', 'experience_id'),
        Index('idx_haptic_pattern_confidence', 'mapping_confidence'),
    )


class SynestheticResearchData(Base):
    """Research data collection for advancing synesthesia science"""
    __tablename__ = "synesthetic_research_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Research study information
    study_id = Column(String(100), nullable=True)
    researcher_id = Column(String(100), nullable=True)
    consent_given = Column(Boolean, nullable=False, default=False)
    
    # Anonymized user data
    demographic_data = Column(JSONB, nullable=True)  # Age range, location, etc.
    synesthetic_type = Column(String(100), nullable=True)  # Natural vs artificial
    
    # Experimental data
    stimulus_response_pairs = Column(JSONB, nullable=False)
    reaction_times_ms = Column(ARRAY(Float), nullable=True)
    accuracy_scores = Column(ARRAY(Float), nullable=True)
    
    # Neural activity patterns (anonymized)
    eeg_features = Column(JSONB, nullable=True)
    fmri_activation_patterns = Column(JSONB, nullable=True)
    
    # Phenomenological reports
    subjective_reports = Column(JSONB, nullable=True)
    qualitative_descriptions = Column(Text, nullable=True)
    
    # Privacy protection
    data_anonymized = Column(Boolean, nullable=False, default=True)
    retention_period_days = Column(Integer, nullable=False, default=365)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_research_study_id', 'study_id'),
        Index('idx_research_consent', 'consent_given'),
        Index('idx_research_anonymized', 'data_anonymized'),
    )


class SynestheticCalibrationSession(Base):
    """Calibration sessions for personalizing synesthetic mappings"""
    __tablename__ = "synesthetic_calibration_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("synesthetic_profiles.id"), nullable=False)
    
    # Session configuration
    calibration_type = Column(String(50), nullable=False)  # 'audio_visual', 'text_color', 'emotion_haptic'
    stimulus_set = Column(JSONB, nullable=False)  # Standardized stimuli used
    
    # User responses
    user_mappings = Column(JSONB, nullable=False)  # User's reported synesthetic experiences
    reaction_times = Column(ARRAY(Float), nullable=True)
    confidence_ratings = Column(ARRAY(Float), nullable=True)
    
    # Calibration results
    baseline_established = Column(Boolean, nullable=False, default=False)
    personalization_factor = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)  # How consistent user responses are
    
    # Session metadata
    session_duration_seconds = Column(Float, nullable=False)
    completion_rate = Column(Float, nullable=False)  # 0.0-1.0
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    profile = relationship("SynestheticProfile")
    
    __table_args__ = (
        Index('idx_calibration_profile_id', 'profile_id'),
        Index('idx_calibration_type', 'calibration_type'),
        Index('idx_calibration_completion', 'completion_rate'),
    )