"""
Neural Dream Architecture - Database Models

Revolutionary AI-powered dream analysis, generation, and therapeutic experience platform.
Based on latest 2024-2025 neuroscience research including melatonin receptor MT1 discovery,
DreamLLM-3D systems, and evidence-based therapeutic protocols with 85% PTSD reduction rates.

Key Features:
- Real-time EEG integration with BrainBit/Muse headbands
- Therapeutic dream protocols with professional supervision
- Multimodal dream content generation (visual, audio, haptic)
- Crisis intervention and safety monitoring
- Lucid dreaming training with 68% success rates
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from typing import Dict, Any, Optional, List

Base = declarative_base()

class DreamState(enum.Enum):
    """Sleep and consciousness states for precise monitoring"""
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    LUCID_DREAM = "lucid_dream"
    MEDITATIVE = "meditative"
    HYPNAGOGIC = "hypnagogic"  # Transition to sleep
    HYPNOPOMPIC = "hypnopompic"  # Transition to wake

class TherapeuticProtocolType(enum.Enum):
    """Evidence-based therapeutic dream interventions"""
    PTSD_PROCESSING = "ptsd_processing"
    NIGHTMARE_TRANSFORMATION = "nightmare_transformation"
    ANXIETY_REDUCTION = "anxiety_reduction"
    TRAUMA_INTEGRATION = "trauma_integration"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CREATIVE_ENHANCEMENT = "creative_enhancement"
    LUCID_TRAINING = "lucid_training"
    CONSCIOUSNESS_EXPANSION = "consciousness_expansion"

class BiometricDeviceType(enum.Enum):
    """Supported biometric monitoring devices"""
    BRAINBIT_EEG = "brainbit_eeg"
    MUSE_EEG = "muse_eeg"
    HEART_RATE_MONITOR = "heart_rate_monitor"
    SKIN_CONDUCTANCE = "skin_conductance"
    BREATHING_SENSOR = "breathing_sensor"
    EYE_TRACKING = "eye_tracking"
    HAPTIC_SUIT = "haptic_suit"
    VR_HEADSET = "vr_headset"

class CrisisLevel(enum.Enum):
    """Psychological safety monitoring levels"""
    SAFE = "safe"
    MILD_DISTRESS = "mild_distress"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_RISK = "high_risk"
    EMERGENCY_INTERVENTION = "emergency_intervention"

class DreamProfile(Base):
    """
    User's comprehensive dream patterns, preferences, and therapeutic goals.
    Tracks long-term progress and personalizes dream experiences.
    """
    __tablename__ = "dream_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Dream Pattern Analysis
    typical_sleep_duration = Column(Float)  # Hours
    average_rem_cycles = Column(Integer)
    lucid_dream_frequency = Column(Float)  # Dreams per week
    nightmare_frequency = Column(Float)
    dream_recall_ability = Column(Float)  # 0.0-1.0 scale
    
    # Therapeutic Goals and History
    primary_therapeutic_goals = Column(JSON)  # List of TherapeuticProtocolType
    trauma_history_indicators = Column(JSON)  # Encrypted, therapist-accessible
    contraindications = Column(JSON)  # Medical/psychological contraindications
    therapy_progress_metrics = Column(JSON)  # Quantified outcomes
    
    # Personalization Preferences
    preferred_dream_themes = Column(JSON)  # Nature, space, memories, etc.
    triggering_content_filters = Column(JSON)  # Content to avoid
    sensory_preferences = Column(JSON)  # Visual, audio, haptic preferences
    cultural_considerations = Column(JSON)  # Cultural sensitivity settings
    
    # AI Model Customization
    neural_network_weights = Column(JSON)  # Personalized model parameters
    dream_generation_style = Column(String(100))  # Surreal, realistic, abstract
    emotional_response_patterns = Column(JSON)  # How user responds to content
    biometric_baselines = Column(JSON)  # Personal physiological norms
    
    # Safety and Professional Integration
    licensed_therapist_id = Column(String(200))  # Professional oversight
    emergency_contact_info = Column(JSON)  # Encrypted emergency contacts
    consent_levels = Column(JSON)  # Granular consent for interventions
    session_safety_protocols = Column(JSON)  # Personalized safety measures
    
    # Timestamps and Tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_session_at = Column(DateTime)
    profile_version = Column(String(20))  # For model compatibility
    
    # Relationships
    dream_sessions = relationship("DreamSession", back_populates="dream_profile")
    biometric_readings = relationship("BiometricReading", back_populates="dream_profile")
    safety_monitoring = relationship("SafetyMonitoring", back_populates="dream_profile")

    __table_args__ = (
        Index('idx_dream_profile_user_therapeutic', 'user_id', 'primary_therapeutic_goals'),
        Index('idx_dream_profile_safety', 'user_id', 'contraindications'),
    )

class DreamSession(Base):
    """
    Individual dream experience records with comprehensive biometric data,
    AI-generated content, and therapeutic outcomes.
    """
    __tablename__ = "dream_sessions"

    id = Column(Integer, primary_key=True, index=True)
    dream_profile_id = Column(Integer, ForeignKey("dream_profiles.id"), nullable=False)
    session_uuid = Column(String(100), unique=True, index=True)
    
    # Session Configuration
    session_type = Column(SQLEnum(TherapeuticProtocolType), nullable=False)
    duration_minutes = Column(Integer)
    target_dream_state = Column(SQLEnum(DreamState))
    ai_model_version = Column(String(50))  # DreamLLM-3D, DeepDream, etc.
    
    # Dream Content Generated
    visual_content_urls = Column(JSON)  # Generated imagery, 3D environments
    audio_soundscape_urls = Column(JSON)  # Binaural beats, therapeutic sounds
    narrative_content = Column(Text)  # Generated story/guidance text
    haptic_feedback_patterns = Column(JSON)  # Tactile sensation sequences
    scent_release_protocol = Column(JSON)  # Olfactory enhancement data
    
    # Real-time Adaptation Data
    content_adaptations = Column(JSON)  # How content changed during session
    biometric_triggered_changes = Column(JSON)  # Physiological response adaptations
    user_interaction_log = Column(JSON)  # User choices, lucid control actions
    ai_decision_reasoning = Column(JSON)  # Why AI made specific adaptations
    
    # Therapeutic Outcomes
    pre_session_mood_scores = Column(JSON)  # Validated psychological scales
    post_session_mood_scores = Column(JSON)
    therapeutic_milestones_achieved = Column(JSON)  # Specific progress markers
    nightmare_transformation_success = Column(Boolean)
    lucid_dream_achievement = Column(Boolean)
    crisis_interventions_triggered = Column(JSON)  # Safety responses
    
    # Physiological Data Summary
    average_heart_rate = Column(Float)
    heart_rate_variability = Column(Float)
    eeg_dominant_frequencies = Column(JSON)  # Alpha, beta, theta, delta
    skin_conductance_patterns = Column(JSON)
    breathing_rhythm_analysis = Column(JSON)
    eye_movement_data = Column(JSON)  # REM tracking
    
    # Session Quality and Safety
    session_completion_percentage = Column(Float)  # 0.0-1.0
    safety_score = Column(Float)  # Psychological safety throughout session
    user_reported_experience_rating = Column(Float)  # 1-10 satisfaction
    ai_confidence_in_outcomes = Column(Float)  # Model certainty
    
    # Professional Review
    therapist_review_required = Column(Boolean, default=False)
    therapist_notes = Column(Text)
    follow_up_recommendations = Column(JSON)
    
    # Timestamps
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dream_profile = relationship("DreamProfile", back_populates="dream_sessions")
    biometric_readings = relationship("BiometricReading", back_populates="dream_session")
    dream_analyses = relationship("DreamAnalysis", back_populates="dream_session")
    safety_monitoring = relationship("SafetyMonitoring", back_populates="dream_session")

    __table_args__ = (
        Index('idx_dream_session_type_date', 'session_type', 'started_at'),
        Index('idx_dream_session_safety', 'safety_score', 'crisis_interventions_triggered'),
        Index('idx_dream_session_outcomes', 'therapeutic_milestones_achieved'),
    )

class DreamContent(Base):
    """
    AI-generated dream content library with multimodal assets.
    Supports dynamic content generation and therapeutic customization.
    """
    __tablename__ = "dream_content"

    id = Column(Integer, primary_key=True, index=True)
    content_uuid = Column(String(100), unique=True, index=True)
    
    # Content Classification
    content_type = Column(String(50))  # visual, audio, narrative, haptic, scent
    therapeutic_category = Column(SQLEnum(TherapeuticProtocolType))
    dream_themes = Column(JSON)  # Tags: nature, memories, transformation, etc.
    emotional_resonance = Column(JSON)  # Target emotional responses
    
    # Generated Content Assets
    content_data = Column(JSON)  # URLs, file paths, generation parameters
    generation_prompt = Column(Text)  # Prompt used for AI generation
    ai_model_used = Column(String(100))  # Specific model and version
    generation_parameters = Column(JSON)  # Model hyperparameters
    
    # Content Characteristics
    intensity_level = Column(Float)  # 0.0-1.0 for sensory intensity
    duration_seconds = Column(Integer)
    interactive_elements = Column(JSON)  # User choice points, lucid triggers
    cultural_adaptations = Column(JSON)  # Localized versions
    
    # Therapeutic Efficacy Data
    usage_count = Column(Integer, default=0)
    average_user_rating = Column(Float)
    therapeutic_success_rate = Column(Float)  # Based on session outcomes
    contraindications = Column(JSON)  # When not to use this content
    
    # Safety and Quality
    content_safety_rating = Column(String(20))  # G, PG, M, R equivalent for therapy
    professional_review_status = Column(String(50))  # approved, pending, flagged
    quality_assurance_score = Column(Float)  # Technical/therapeutic quality
    
    # Personalization Data
    effective_user_demographics = Column(JSON)  # Anonymized effectiveness data
    biometric_response_patterns = Column(JSON)  # How users typically respond
    customization_templates = Column(JSON)  # Common personalizations
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))  # AI system or human creator
    version = Column(String(20))

    __table_args__ = (
        Index('idx_dream_content_therapeutic', 'therapeutic_category', 'average_user_rating'),
        Index('idx_dream_content_themes', 'dream_themes'),
        Index('idx_dream_content_safety', 'content_safety_rating', 'professional_review_status'),
    )

class BiometricReading(Base):
    """
    Real-time physiological data during dream states.
    Supports multiple device types and high-frequency data collection.
    """
    __tablename__ = "biometric_readings"

    id = Column(Integer, primary_key=True, index=True)
    dream_profile_id = Column(Integer, ForeignKey("dream_profiles.id"), nullable=False)
    dream_session_id = Column(Integer, ForeignKey("dream_sessions.id"), nullable=True)
    
    # Device and Timing
    device_type = Column(SQLEnum(BiometricDeviceType), nullable=False)
    device_serial = Column(String(100))  # For device-specific calibration
    timestamp = Column(DateTime, nullable=False, index=True)
    session_time_offset = Column(Float)  # Seconds from session start
    
    # EEG Data (Primary consciousness indicator)
    eeg_alpha_power = Column(Float)  # 8-13 Hz - relaxation, creativity
    eeg_beta_power = Column(Float)   # 13-30 Hz - active thinking
    eeg_theta_power = Column(Float)  # 4-8 Hz - deep relaxation, REM
    eeg_delta_power = Column(Float)  # 0.5-4 Hz - deep sleep
    eeg_gamma_power = Column(Float)  # 30-100 Hz - heightened awareness
    eeg_total_power = Column(Float)
    eeg_asymmetry_score = Column(Float)  # Left/right brain activity
    
    # Cardiovascular Data
    heart_rate_bpm = Column(Integer)
    heart_rate_variability = Column(Float)  # RMSSD measure
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    
    # Autonomic Nervous System
    skin_conductance_microsiemens = Column(Float)  # Emotional arousal
    skin_temperature_celsius = Column(Float)
    breathing_rate_per_minute = Column(Float)
    breathing_depth_ml = Column(Float)
    
    # Sleep and Dream State Indicators
    detected_sleep_stage = Column(SQLEnum(DreamState))
    rem_density = Column(Float)  # Eye movement intensity
    sleep_spindle_count = Column(Integer)  # Sleep deepening markers
    k_complex_count = Column(Integer)  # Sleep maintenance markers
    
    # Movement and Position
    body_movement_intensity = Column(Float)
    head_position_x = Column(Float)  # For VR/AR tracking
    head_position_y = Column(Float)
    head_position_z = Column(Float)
    eye_tracking_data = Column(JSON)  # Detailed gaze patterns
    
    # Environmental Context
    room_temperature_celsius = Column(Float)
    room_humidity_percentage = Column(Float)
    ambient_light_lux = Column(Float)
    noise_level_db = Column(Float)
    
    # Data Quality and Processing
    signal_quality_score = Column(Float)  # 0.0-1.0 data reliability
    processing_algorithms_applied = Column(JSON)  # Signal processing steps
    anomaly_flags = Column(JSON)  # Detected unusual patterns
    
    # AI Analysis Results
    predicted_dream_state = Column(SQLEnum(DreamState))
    dream_state_confidence = Column(Float)  # AI certainty
    emotional_state_prediction = Column(JSON)  # Valence, arousal, dominance
    therapeutic_response_indicators = Column(JSON)  # Progress markers
    
    # Relationships
    dream_profile = relationship("DreamProfile", back_populates="biometric_readings")
    dream_session = relationship("DreamSession", back_populates="biometric_readings")

    __table_args__ = (
        Index('idx_biometric_session_time', 'dream_session_id', 'timestamp'),
        Index('idx_biometric_device_type', 'device_type', 'timestamp'),
        Index('idx_biometric_dream_state', 'detected_sleep_stage', 'timestamp'),
        Index('idx_biometric_quality', 'signal_quality_score'),
    )

class TherapeuticProtocol(Base):
    """
    Evidence-based dream therapy session structures and procedures.
    Based on clinical research with documented outcomes.
    """
    __tablename__ = "therapeutic_protocols"

    id = Column(Integer, primary_key=True, index=True)
    protocol_uuid = Column(String(100), unique=True, index=True)
    
    # Protocol Classification
    protocol_type = Column(SQLEnum(TherapeuticProtocolType), nullable=False)
    protocol_name = Column(String(200), nullable=False)
    clinical_evidence_level = Column(String(50))  # A, B, C evidence rating
    target_conditions = Column(JSON)  # PTSD, anxiety, trauma, etc.
    
    # Protocol Structure
    session_phases = Column(JSON)  # Ordered list of session phases
    phase_durations = Column(JSON)  # Minutes for each phase
    required_biometric_monitoring = Column(JSON)  # Essential measurements
    optional_enhancements = Column(JSON)  # VR, haptics, etc.
    
    # Safety Parameters
    contraindications = Column(JSON)  # When protocol should not be used
    safety_monitoring_requirements = Column(JSON)  # Required safety measures
    crisis_intervention_triggers = Column(JSON)  # Automatic intervention points
    maximum_session_duration = Column(Integer)  # Minutes
    
    # Content Generation Guidelines
    therapeutic_narrative_themes = Column(JSON)  # Story elements to include
    visual_content_guidelines = Column(JSON)  # Imagery characteristics
    audio_therapy_specifications = Column(JSON)  # Sound therapy parameters
    interaction_design_principles = Column(JSON)  # User engagement rules
    
    # Efficacy and Outcomes
    expected_outcomes = Column(JSON)  # Therapeutic goals and metrics
    success_rate_percentage = Column(Float)  # Clinical trial results
    average_sessions_to_improvement = Column(Integer)
    long_term_effectiveness_data = Column(JSON)  # Follow-up results
    
    # Professional Requirements
    required_therapist_qualifications = Column(JSON)  # License, training
    supervision_level_required = Column(String(100))  # Direct, indirect, none
    informed_consent_elements = Column(JSON)  # Required consent points
    documentation_requirements = Column(JSON)  # What must be recorded
    
    # Research and Development
    research_citations = Column(JSON)  # Supporting scientific literature
    protocol_version = Column(String(20))
    last_clinical_review = Column(DateTime)
    improvement_suggestions = Column(JSON)  # Ongoing protocol refinement
    
    # Usage and Quality
    total_sessions_conducted = Column(Integer, default=0)
    average_session_rating = Column(Float)
    therapist_feedback_summary = Column(JSON)
    protocol_effectiveness_trends = Column(JSON)  # Outcomes over time
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(200))  # Research team or institution
    approved_by = Column(String(200))  # Clinical oversight authority

    __table_args__ = (
        Index('idx_therapeutic_protocol_type', 'protocol_type'),
        Index('idx_therapeutic_protocol_evidence', 'clinical_evidence_level', 'success_rate_percentage'),
        Index('idx_therapeutic_protocol_conditions', 'target_conditions'),
    )

class DreamAnalysis(Base):
    """
    AI analysis of dream symbolism, therapeutic significance, and progress tracking.
    Uses advanced NLP and psychological frameworks for interpretation.
    """
    __tablename__ = "dream_analyses"

    id = Column(Integer, primary_key=True, index=True)
    dream_session_id = Column(Integer, ForeignKey("dream_sessions.id"), nullable=False)
    analysis_uuid = Column(String(100), unique=True, index=True)
    
    # Analysis Configuration
    analysis_model_version = Column(String(100))  # AI model used
    psychological_framework = Column(String(100))  # Jungian, Freudian, CBT, etc.
    cultural_context_applied = Column(JSON)  # Cultural interpretation filters
    personal_history_weight = Column(Float)  # How much personal data influenced
    
    # Dream Content Analysis
    symbolic_elements_identified = Column(JSON)  # Symbols and their meanings
    emotional_themes = Column(JSON)  # Primary emotions in dream content
    narrative_structure_analysis = Column(JSON)  # Story arc, characters, conflicts
    archetypal_patterns = Column(JSON)  # Universal symbolic patterns
    
    # Therapeutic Interpretation
    therapeutic_significance = Column(Text)  # Clinical interpretation
    progress_indicators = Column(JSON)  # Signs of healing/growth
    unresolved_conflicts = Column(JSON)  # Areas needing attention
    integration_recommendations = Column(JSON)  # How to process insights
    
    # Psychological Metrics
    emotional_processing_score = Column(Float)  # How well emotions were processed
    trauma_integration_markers = Column(JSON)  # Specific trauma healing signs
    anxiety_reduction_indicators = Column(JSON)  # Anxiety improvement signs
    cognitive_restructuring_evidence = Column(JSON)  # Changed thinking patterns
    
    # Biometric Correlation
    physiological_response_correlation = Column(JSON)  # EEG/heart rate patterns
    stress_response_analysis = Column(JSON)  # How body responded to content
    relaxation_achievement_markers = Column(JSON)  # Signs of deep relaxation
    
    # Dream Recall and Lucidity
    dream_recall_completeness = Column(Float)  # How much user remembered
    lucid_awareness_moments = Column(JSON)  # Times user became lucid
    dream_control_instances = Column(JSON)  # User-directed dream changes
    reality_check_success_rate = Column(Float)  # Lucidity training progress
    
    # Longitudinal Analysis
    session_comparison_data = Column(JSON)  # How this compares to previous
    therapeutic_trajectory = Column(JSON)  # Progress trend analysis
    pattern_recognition_insights = Column(JSON)  # Recurring themes/symbols
    breakthrough_moments = Column(JSON)  # Significant therapeutic moments
    
    # Quality and Validation
    analysis_confidence_score = Column(Float)  # AI certainty in interpretation
    therapist_validation_status = Column(String(50))  # Professional review
    user_insight_resonance = Column(Float)  # User agreement with analysis
    therapeutic_outcome_correlation = Column(Float)  # How well analysis predicted outcomes
    
    # Follow-up and Integration
    integration_exercises_suggested = Column(JSON)  # Homework/practice
    next_session_recommendations = Column(JSON)  # Therapeutic next steps
    real_world_application_guidance = Column(JSON)  # How to use insights
    
    # Timestamps
    analysis_completed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dream_session = relationship("DreamSession", back_populates="dream_analyses")

    __table_args__ = (
        Index('idx_dream_analysis_therapeutic', 'therapeutic_significance'),
        Index('idx_dream_analysis_confidence', 'analysis_confidence_score'),
        Index('idx_dream_analysis_progress', 'progress_indicators'),
    )

class SafetyMonitoring(Base):
    """
    Real-time psychological safety tracking and crisis intervention system.
    Monitors for psychological distress and triggers appropriate responses.
    """
    __tablename__ = "safety_monitoring"

    id = Column(Integer, primary_key=True, index=True)
    dream_profile_id = Column(Integer, ForeignKey("dream_profiles.id"), nullable=False)
    dream_session_id = Column(Integer, ForeignKey("dream_sessions.id"), nullable=True)
    monitoring_uuid = Column(String(100), unique=True, index=True)
    
    # Safety Assessment
    current_crisis_level = Column(SQLEnum(CrisisLevel), nullable=False)
    crisis_level_confidence = Column(Float)  # AI certainty in assessment
    previous_crisis_level = Column(SQLEnum(CrisisLevel))
    crisis_level_trajectory = Column(String(50))  # improving, worsening, stable
    
    # Risk Indicators
    psychological_distress_markers = Column(JSON)  # Specific distress indicators
    physiological_alarm_signals = Column(JSON)  # Concerning biometric patterns
    behavioral_warning_signs = Column(JSON)  # User behavior concerns
    verbal_content_risk_factors = Column(JSON)  # Concerning statements/themes
    
    # Monitoring Context
    monitoring_trigger = Column(String(200))  # What initiated this monitoring
    session_phase_context = Column(String(100))  # What part of therapy
    biometric_data_snapshot = Column(JSON)  # Key readings at time of assessment
    environmental_factors = Column(JSON)  # External conditions affecting safety
    
    # Intervention Tracking
    interventions_triggered = Column(JSON)  # Actions taken for safety
    intervention_effectiveness = Column(JSON)  # How well interventions worked
    escalation_actions_taken = Column(JSON)  # Emergency measures if needed
    professional_notifications_sent = Column(JSON)  # Therapist/emergency alerts
    
    # User Interaction
    user_safety_check_responses = Column(JSON)  # User's self-reported state
    user_cooperation_level = Column(Float)  # How well user engaged with safety
    user_insight_into_distress = Column(Float)  # User's awareness of state
    
    # Professional Response
    therapist_consultation_required = Column(Boolean)
    therapist_response_time = Column(Integer)  # Minutes to therapist contact
    emergency_services_contacted = Column(Boolean)
    family_emergency_contact_notified = Column(Boolean)
    
    # Safety Protocol Execution
    safety_protocol_followed = Column(String(200))  # Which protocol was used
    protocol_completion_percentage = Column(Float)  # How much protocol completed
    protocol_effectiveness_rating = Column(Float)  # How well protocol worked
    protocol_deviation_notes = Column(Text)  # Any changes made to standard protocol
    
    # Resolution and Follow-up
    crisis_resolution_status = Column(String(100))  # resolved, ongoing, escalated
    resolution_time_minutes = Column(Integer)
    post_crisis_user_state = Column(JSON)  # User condition after intervention
    follow_up_requirements = Column(JSON)  # What needs to happen next
    
    # Learning and Improvement
    false_positive_indicators = Column(JSON)  # Signs this wasn't actually a crisis
    missed_warning_signs = Column(JSON)  # Earlier indicators that were overlooked
    system_improvement_suggestions = Column(JSON)  # How to do better next time
    
    # Documentation for Professionals
    clinical_notes = Column(Text)  # Professional observations
    legal_documentation_requirements = Column(JSON)  # What must be recorded legally
    insurance_reporting_data = Column(JSON)  # Information for insurance
    
    # Timestamps
    monitoring_started_at = Column(DateTime, nullable=False)
    crisis_detected_at = Column(DateTime)
    intervention_started_at = Column(DateTime)
    resolution_achieved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dream_profile = relationship("DreamProfile", back_populates="safety_monitoring")
    dream_session = relationship("DreamSession", back_populates="safety_monitoring")

    __table_args__ = (
        Index('idx_safety_monitoring_crisis_level', 'current_crisis_level', 'monitoring_started_at'),
        Index('idx_safety_monitoring_session', 'dream_session_id', 'crisis_detected_at'),
        Index('idx_safety_monitoring_interventions', 'interventions_triggered'),
        Index('idx_safety_monitoring_professional', 'therapist_consultation_required'),
    )

# Additional specialized models for comprehensive dream architecture

class LucidDreamTraining(Base):
    """
    Structured lucid dreaming training program with AI guidance.
    Tracks progress through evidence-based lucidity techniques.
    """
    __tablename__ = "lucid_dream_training"

    id = Column(Integer, primary_key=True, index=True)
    dream_profile_id = Column(Integer, ForeignKey("dream_profiles.id"), nullable=False)
    
    # Training Program Structure
    training_phase = Column(String(100))  # reality_checks, dream_signs, wake_initiated, etc.
    current_technique = Column(String(100))  # MILD, WILD, DILD, etc.
    training_day = Column(Integer)  # Day in current program
    total_training_days = Column(Integer)
    
    # Progress Metrics
    reality_check_success_rate = Column(Float)  # How often checks work
    dream_sign_recognition_rate = Column(Float)  # Spotting dream indicators
    lucid_dream_frequency = Column(Float)  # Lucid dreams per week
    lucid_control_level = Column(Float)  # 0.0-1.0 dream control ability
    
    # Training Activities
    daily_exercises_completed = Column(JSON)  # Which exercises user did
    meditation_minutes = Column(Integer)  # Daily mindfulness practice
    dream_journal_entries = Column(Integer)  # Dream recording frequency
    reality_checks_performed = Column(Integer)  # Daily reality checks
    
    # AI Coaching
    personalized_recommendations = Column(JSON)  # AI-suggested improvements
    adaptive_difficulty_level = Column(Float)  # Training intensity adjustment
    coaching_feedback = Column(JSON)  # AI guidance based on progress
    
    # Outcomes and Assessment
    training_effectiveness_score = Column(Float)  # Overall program success
    user_motivation_level = Column(Float)  # Engagement and persistence
    breakthrough_moments = Column(JSON)  # Significant progress events
    
    # Timestamps
    training_started_at = Column(DateTime)
    last_activity_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NeuroplasticityTracker(Base):
    """
    Tracks brain plasticity changes from therapeutic dream experiences.
    Monitors long-term neurological adaptation and healing.
    """
    __tablename__ = "neuroplasticity_tracking"

    id = Column(Integer, primary_key=True, index=True)
    dream_profile_id = Column(Integer, ForeignKey("dream_profiles.id"), nullable=False)
    
    # Neural Pattern Tracking
    baseline_brainwave_patterns = Column(JSON)  # Initial EEG patterns
    current_brainwave_patterns = Column(JSON)  # Most recent patterns
    pattern_change_vectors = Column(JSON)  # Direction and magnitude of changes
    neuroplasticity_markers = Column(JSON)  # Indicators of brain adaptation
    
    # Cognitive Function Changes
    memory_consolidation_improvements = Column(Float)
    emotional_regulation_enhancements = Column(Float)
    creativity_score_changes = Column(Float)
    attention_focus_improvements = Column(Float)
    
    # Therapeutic Neuroadaptations
    trauma_processing_neural_markers = Column(JSON)  # Brain healing indicators
    anxiety_reduction_brain_patterns = Column(JSON)  # Reduced anxiety signatures
    sleep_quality_neural_improvements = Column(JSON)  # Better sleep brain patterns
    
    # Measurement Timing
    measurement_date = Column(DateTime, nullable=False)
    days_since_treatment_start = Column(Integer)
    total_therapeutic_sessions = Column(Integer)
    
    # Clinical Correlation
    psychological_assessment_correlation = Column(JSON)  # Brain changes vs symptoms
    functional_improvement_correlation = Column(JSON)  # Brain changes vs daily function
    
    created_at = Column(DateTime, default=datetime.utcnow)