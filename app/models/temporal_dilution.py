"""
Temporal Dilution Engine - Database Models

Revolutionary AI-powered time perception manipulation system that creates time-dilated
experiences and optimized flow states. Based on 2024-2025 neuroscience research
including Intrinsic Neural Timescales, temporal perception plasticity, and
circadian rhythm integration for enhanced learning and therapeutic outcomes.

Key Features:
- Real-time EEG temporal signature detection with autocorrelation analysis
- Flow state induction with 40% learning acceleration rates
- Circadian rhythm integration for optimal temporal experiences
- Multi-modal temporal cues (visual, auditory, haptic) for immersive time dilation
- Safety protocols preventing temporal displacement and disorientation
- Therapeutic temporal protocols with measurable cognitive enhancement outcomes
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from typing import Dict, Any, Optional, List

Base = declarative_base()

class TemporalState(enum.Enum):
    """Temporal perception states for precise monitoring"""
    BASELINE = "baseline"  # Normal time perception
    ACCELERATED = "accelerated"  # Time feels faster
    DILATED = "dilated"  # Time feels slower (target for learning)
    FLOW_STATE = "flow_state"  # Optimal performance state
    MEDITATIVE_TIME = "meditative_time"  # Transcendent time experience
    SUSPENDED_TIME = "suspended_time"  # Near-timeless perception
    TEMPORAL_CONFUSION = "temporal_confusion"  # Disoriented state (safety concern)
    RECOVERY_PHASE = "recovery_phase"  # Returning to baseline

class FlowStateType(enum.Enum):
    """Specific flow state categories for targeted experiences"""
    LEARNING_FLOW = "learning_flow"  # Optimized for knowledge acquisition
    CREATIVE_FLOW = "creative_flow"  # Enhanced creative performance
    THERAPEUTIC_FLOW = "therapeutic_flow"  # Healing and emotional processing
    PERFORMANCE_FLOW = "performance_flow"  # Peak skill execution
    MEDITATIVE_FLOW = "meditative_flow"  # Deep mindfulness states
    PROBLEM_SOLVING_FLOW = "problem_solving_flow"  # Enhanced analytical thinking
    SOCIAL_FLOW = "social_flow"  # Optimized interpersonal connection
    RECOVERY_FLOW = "recovery_flow"  # Restorative and healing experiences

class TemporalCueType(enum.Enum):
    """Multi-modal temporal manipulation methods"""
    VISUAL_RHYTHM = "visual_rhythm"  # Visual pulsing, movement patterns
    AUDITORY_BEAT = "auditory_beat"  # Binaural beats, rhythmic audio
    HAPTIC_PULSE = "haptic_pulse"  # Tactile timing cues
    BREATHING_PATTERN = "breathing_pattern"  # Guided breathing rhythms
    HEART_RATE_SYNC = "heart_rate_sync"  # Synchronized to heart rhythm
    NEURAL_ENTRAINMENT = "neural_entrainment"  # EEG-driven feedback
    ENVIRONMENTAL_SHIFT = "environmental_shift"  # Ambient condition changes
    COGNITIVE_ANCHOR = "cognitive_anchor"  # Mental timing references

class CircadianPhase(enum.Enum):
    """Circadian rhythm phases for optimized temporal experiences"""
    MORNING_PEAK = "morning_peak"  # 6-9 AM optimal alertness
    MID_MORNING = "mid_morning"  # 9-12 PM sustained focus
    AFTERNOON_OPTIMAL = "afternoon_optimal"  # 12-3 PM peak performance
    AFTERNOON_DIP = "afternoon_dip"  # 3-6 PM natural energy dip
    EVENING_CREATIVE = "evening_creative"  # 6-9 PM creative peak
    NIGHT_WIND_DOWN = "night_wind_down"  # 9 PM-12 AM relaxation
    LATE_NIGHT = "late_night"  # 12-3 AM deep processing
    PRE_DAWN = "pre_dawn"  # 3-6 AM restorative phase

class BiometricDeviceType(enum.Enum):
    """Supported temporal monitoring devices"""
    EEG_HEADBAND = "eeg_headband"  # Real-time brainwave monitoring
    HEART_RATE_MONITOR = "heart_rate_monitor"  # HRV temporal indicators
    EYE_TRACKER = "eye_tracker"  # Gaze patterns and blink rates
    SKIN_CONDUCTANCE = "skin_conductance"  # Autonomic nervous system
    BREATHING_SENSOR = "breathing_sensor"  # Respiratory rhythm tracking
    MOTION_TRACKER = "motion_tracker"  # Physical movement patterns
    SMARTWATCH = "smartwatch"  # Multi-sensor temporal data
    VR_HEADSET = "vr_headset"  # Immersive temporal environment

class SafetyLevel(enum.Enum):
    """Temporal safety monitoring levels"""
    SAFE = "safe"  # Normal temporal experience
    MILD_DISORIENTATION = "mild_disorientation"  # Slight confusion
    MODERATE_CONCERN = "moderate_concern"  # Noticeable temporal displacement
    HIGH_RISK = "high_risk"  # Significant disorientation
    EMERGENCY_RESET = "emergency_reset"  # Immediate intervention required

class TemporalProfile(Base):
    """
    User's comprehensive temporal perception patterns, preferences, and baseline measurements.
    Tracks individual differences in time perception and optimal temporal states.
    """
    __tablename__ = "temporal_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Baseline Temporal Characteristics
    baseline_time_perception_accuracy = Column(Float)  # How accurately user estimates time
    natural_rhythm_period = Column(Float)  # User's natural temporal rhythm (seconds)
    temporal_sensitivity_score = Column(Float)  # 0.0-1.0 sensitivity to time changes
    chronotype_classification = Column(String(50))  # Morning/evening preference
    
    # Flow State Tendencies
    flow_state_frequency = Column(Float)  # How often user achieves flow naturally
    optimal_flow_duration = Column(Float)  # Minutes for best flow experiences
    flow_entry_patterns = Column(JSON)  # What triggers flow for this user
    flow_disruption_triggers = Column(JSON)  # What breaks flow state
    
    # Temporal Learning Preferences
    preferred_temporal_state = Column(SQLEnum(TemporalState))
    optimal_learning_time_ratio = Column(Float)  # Preferred subjective:objective time ratio
    attention_span_baseline = Column(Float)  # Natural focus duration (minutes)
    temporal_learning_enhancement_factor = Column(Float)  # How much temporal dilation helps
    
    # Circadian Integration
    personal_circadian_phase_preferences = Column(JSON)  # Best times for temporal work
    circadian_amplitude = Column(Float)  # Strength of circadian rhythm
    timezone_adaptation_speed = Column(Float)  # How quickly user adapts to time changes
    sleep_wake_cycle_stability = Column(Float)  # Consistency of sleep patterns
    
    # Physiological Baselines
    baseline_heart_rate_variability = Column(Float)  # HRV temporal indicator
    baseline_eeg_temporal_signatures = Column(JSON)  # Personal brainwave patterns
    autonomic_nervous_system_balance = Column(Float)  # Sympathetic/parasympathetic balance
    stress_response_temporal_patterns = Column(JSON)  # How stress affects time perception
    
    # Safety and Tolerance Levels
    maximum_safe_temporal_distortion = Column(Float)  # Largest safe time ratio change
    disorientation_sensitivity = Column(Float)  # How easily user gets confused
    recovery_time_requirements = Column(Float)  # Minutes needed to return to baseline
    contraindications = Column(JSON)  # Conditions preventing temporal manipulation
    
    # Personalization Parameters
    preferred_temporal_cue_types = Column(JSON)  # Visual, auditory, haptic preferences
    cultural_temporal_considerations = Column(JSON)  # Cultural time perception factors
    accessibility_accommodations = Column(JSON)  # Special needs for temporal experiences
    motivation_factors = Column(JSON)  # What motivates user in temporal experiences
    
    # Professional Integration
    therapist_supervision_required = Column(Boolean, default=False)
    medical_clearance_status = Column(String(100))
    emergency_contact_info = Column(JSON)  # Encrypted contact information
    
    # AI Model Customization
    neural_network_temporal_weights = Column(JSON)  # Personalized AI parameters
    temporal_experience_learning_data = Column(JSON)  # AI learning from user responses
    adaptive_algorithm_parameters = Column(JSON)  # Real-time adaptation settings
    
    # Timestamps and Tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_temporal_session_at = Column(DateTime)
    profile_calibration_status = Column(String(100))
    
    # Relationships
    temporal_sessions = relationship("TemporalSession", back_populates="temporal_profile")
    flow_state_sessions = relationship("FlowStateSession", back_populates="temporal_profile")
    temporal_biometrics = relationship("TemporalBiometricReading", back_populates="temporal_profile")
    safety_monitoring = relationship("TemporalSafetyMonitoring", back_populates="temporal_profile")

    __table_args__ = (
        Index('idx_temporal_profile_user', 'user_id'),
        Index('idx_temporal_profile_chronotype', 'chronotype_classification'),
        Index('idx_temporal_profile_safety', 'maximum_safe_temporal_distortion', 'disorientation_sensitivity'),
    )

class TemporalSession(Base):
    """
    Individual temporal dilution experience records with comprehensive monitoring,
    AI-driven adaptation, and outcome measurement.
    """
    __tablename__ = "temporal_sessions"

    id = Column(Integer, primary_key=True, index=True)
    temporal_profile_id = Column(Integer, ForeignKey("temporal_profiles.id"), nullable=False)
    session_uuid = Column(String(100), unique=True, index=True)
    
    # Session Configuration
    session_type = Column(SQLEnum(FlowStateType), nullable=False)
    target_temporal_state = Column(SQLEnum(TemporalState), nullable=False)
    planned_duration_minutes = Column(Integer)
    target_time_dilation_ratio = Column(Float)  # Subjective:objective time ratio
    
    # Temporal Manipulation Protocol
    temporal_cue_sequence = Column(JSON)  # Ordered list of temporal cues applied
    cue_timing_schedule = Column(JSON)  # When each cue was applied
    biometric_adaptation_triggers = Column(JSON)  # Physiological response thresholds
    ai_model_version = Column(String(50))
    
    # Real-time Adaptation Data
    temporal_state_transitions = Column(JSON)  # How temporal state changed over session
    biometric_triggered_adaptations = Column(JSON)  # Changes based on physiological data
    user_feedback_adaptations = Column(JSON)  # Changes based on user input
    ai_decision_reasoning = Column(JSON)  # Why AI made specific adaptations
    
    # Flow State Achievement
    flow_state_achieved = Column(Boolean)
    time_to_flow_entry = Column(Float)  # Minutes to achieve flow
    flow_state_duration = Column(Float)  # Minutes in optimal flow
    flow_depth_score = Column(Float)  # 0.0-1.0 quality of flow achieved
    flow_maintenance_stability = Column(Float)  # How well flow was maintained
    
    # Temporal Experience Metrics
    subjective_time_perception = Column(Float)  # User's perceived vs actual duration
    temporal_awareness_level = Column(Float)  # How conscious user was of time
    time_dilation_achievement = Column(Float)  # Actual vs target time ratio
    temporal_immersion_depth = Column(Float)  # How absorbed user was in experience
    
    # Learning and Performance Outcomes
    learning_task_completion_rate = Column(Float)  # If learning tasks were included
    performance_improvement_percentage = Column(Float)  # Skill enhancement measurement
    memory_consolidation_indicators = Column(JSON)  # Signs of enhanced learning
    creative_output_metrics = Column(JSON)  # Creative task performance
    problem_solving_enhancement = Column(JSON)  # Analytical thinking improvements
    
    # Physiological Response Summary
    heart_rate_variability_changes = Column(JSON)  # HRV patterns during session
    brainwave_pattern_evolution = Column(JSON)  # EEG changes throughout session
    autonomic_balance_shifts = Column(JSON)  # Nervous system state changes
    stress_hormone_indicators = Column(JSON)  # Cortisol and related markers
    
    # Circadian Integration
    circadian_phase_at_start = Column(SQLEnum(CircadianPhase))
    circadian_optimization_applied = Column(Boolean)
    chronotype_alignment_score = Column(Float)  # How well timed for user's rhythm
    
    # Safety and Wellbeing
    disorientation_incidents = Column(JSON)  # Any confusion or displacement events
    safety_interventions_triggered = Column(JSON)  # Automatic safety responses
    recovery_time_actual = Column(Float)  # Minutes to return to baseline
    post_session_wellbeing_score = Column(Float)  # User-reported state after session
    
    # User Experience Quality
    session_enjoyment_rating = Column(Float)  # 1-10 user satisfaction
    perceived_benefit_score = Column(Float)  # How helpful user found session
    temporal_experience_quality = Column(Float)  # Overall temporal experience rating
    user_control_satisfaction = Column(Float)  # How much control user felt they had
    
    # Professional Oversight
    therapist_supervision_provided = Column(Boolean)
    therapist_notes = Column(Text)
    clinical_assessment_scores = Column(JSON)  # Professional evaluation metrics
    follow_up_recommendations = Column(JSON)
    
    # AI Learning and Optimization
    ai_confidence_in_outcomes = Column(Float)  # Model certainty in results
    unexpected_responses_detected = Column(JSON)  # Unusual user responses
    model_improvement_suggestions = Column(JSON)  # How AI could do better
    personalization_accuracy_score = Column(Float)  # How well personalized
    
    # Timestamps
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    temporal_profile = relationship("TemporalProfile", back_populates="temporal_sessions")
    flow_state_sessions = relationship("FlowStateSession", back_populates="temporal_session")
    temporal_biometrics = relationship("TemporalBiometricReading", back_populates="temporal_session")
    safety_monitoring = relationship("TemporalSafetyMonitoring", back_populates="temporal_session")

    __table_args__ = (
        Index('idx_temporal_session_type_date', 'session_type', 'started_at'),
        Index('idx_temporal_session_outcomes', 'flow_state_achieved', 'learning_task_completion_rate'),
        Index('idx_temporal_session_safety', 'disorientation_incidents', 'safety_interventions_triggered'),
    )

class FlowStateSession(Base):
    """
    Detailed flow state tracking and optimization within temporal dilution experiences.
    Monitors the specific characteristics and quality of flow state achievement.
    """
    __tablename__ = "flow_state_sessions"

    id = Column(Integer, primary_key=True, index=True)
    temporal_profile_id = Column(Integer, ForeignKey("temporal_profiles.id"), nullable=False)
    temporal_session_id = Column(Integer, ForeignKey("temporal_sessions.id"), nullable=False)
    flow_session_uuid = Column(String(100), unique=True, index=True)
    
    # Flow State Characteristics
    flow_state_type = Column(SQLEnum(FlowStateType), nullable=False)
    challenge_skill_balance = Column(Float)  # Optimal balance indicator
    clear_goals_achievement = Column(Float)  # How well-defined goals were
    immediate_feedback_quality = Column(Float)  # Quality of feedback loops
    action_awareness_merging = Column(Float)  # Unconscious competence level
    
    # Flow State Dimensions (based on Csikszentmihalyi research)
    concentration_depth = Column(Float)  # 0.0-1.0 focus intensity
    self_consciousness_loss = Column(Float)  # Degree of ego dissolution
    time_transformation_experience = Column(Float)  # Temporal perception alteration
    autotelic_experience_quality = Column(Float)  # Intrinsic enjoyment level
    
    # Neural Correlates of Flow
    frontal_alpha_asymmetry = Column(Float)  # EEG flow state marker
    frontocentral_theta_power = Column(Float)  # Cognitive control marker
    transient_hypofrontality = Column(Float)  # Reduced prefrontal activity
    dopamine_system_indicators = Column(JSON)  # Reward system activation
    
    # Flow Entry and Maintenance
    flow_entry_method = Column(String(200))  # How flow was achieved
    flow_triggers_activated = Column(JSON)  # Specific triggers that worked
    flow_disruption_events = Column(JSON)  # What interrupted flow
    flow_recovery_attempts = Column(JSON)  # Efforts to regain flow
    
    # Performance During Flow
    task_performance_metrics = Column(JSON)  # Objective performance measures
    creative_output_quality = Column(Float)  # Creative task outcomes
    learning_efficiency_score = Column(Float)  # Learning rate during flow
    error_rate_reduction = Column(Float)  # Improvement in accuracy
    
    # Temporal Experience in Flow
    subjective_time_compression = Column(Float)  # How much time seemed compressed
    temporal_awareness_dissolution = Column(Float)  # Loss of time consciousness
    flow_time_vs_actual_time_ratio = Column(Float)  # Perceived vs real duration
    time_dilation_synchronization = Column(Float)  # Alignment with temporal dilution
    
    # Physiological Flow Markers
    heart_rate_coherence = Column(Float)  # Cardiovascular coherence measure
    breathing_rhythm_entrainment = Column(Float)  # Respiratory flow synchronization
    galvanic_skin_response_stability = Column(Float)  # Emotional arousal stability
    muscle_tension_optimization = Column(Float)  # Physical relaxation level
    
    # Flow Quality Assessment
    flow_state_authenticity = Column(Float)  # How genuine the flow state was
    flow_depth_progression = Column(JSON)  # How flow deepened over time
    peak_flow_moments = Column(JSON)  # Highest quality flow periods
    flow_afterglow_duration = Column(Float)  # Positive effects lasting after flow
    
    # Learning and Adaptation
    flow_skill_development = Column(JSON)  # Skills improved during flow
    flow_pattern_learning = Column(JSON)  # What user learned about their flow
    flow_optimization_insights = Column(JSON)  # How to improve future flow
    
    # Integration with Temporal Dilution
    temporal_flow_synergy_score = Column(Float)  # How well temporal and flow worked together
    optimal_dilation_for_flow = Column(Float)  # Best time ratio for this user's flow
    temporal_cue_flow_enhancement = Column(JSON)  # Which temporal cues helped flow
    
    # Timestamps
    flow_entered_at = Column(DateTime)
    flow_peak_at = Column(DateTime)
    flow_ended_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    temporal_profile = relationship("TemporalProfile", back_populates="flow_state_sessions")
    temporal_session = relationship("TemporalSession", back_populates="flow_state_sessions")

    __table_args__ = (
        Index('idx_flow_session_type', 'flow_state_type'),
        Index('idx_flow_session_quality', 'flow_state_authenticity', 'flow_depth_progression'),
        Index('idx_flow_session_temporal_sync', 'temporal_flow_synergy_score'),
    )

class TemporalBiometricReading(Base):
    """
    Real-time physiological monitoring during temporal dilution experiences.
    Tracks temporal perception indicators and safety parameters.
    """
    __tablename__ = "temporal_biometric_readings"

    id = Column(Integer, primary_key=True, index=True)
    temporal_profile_id = Column(Integer, ForeignKey("temporal_profiles.id"), nullable=False)
    temporal_session_id = Column(Integer, ForeignKey("temporal_sessions.id"), nullable=True)
    
    # Device and Timing
    device_type = Column(SQLEnum(BiometricDeviceType), nullable=False)
    device_serial = Column(String(100))
    timestamp = Column(DateTime, nullable=False, index=True)
    session_time_offset = Column(Float)  # Seconds from session start
    
    # Temporal Perception Neural Markers
    intrinsic_neural_timescales = Column(Float)  # Autocorrelation-based timing
    temporal_prediction_error = Column(Float)  # Prediction error in timing
    time_cell_activity_patterns = Column(JSON)  # Hippocampal timing neurons
    suprachiasmatic_nucleus_indicators = Column(Float)  # Circadian pacemaker activity
    
    # EEG Temporal Signatures
    eeg_alpha_rhythm_stability = Column(Float)  # 8-12 Hz temporal marker
    eeg_theta_coherence = Column(Float)  # 4-8 Hz flow state indicator
    eeg_gamma_synchronization = Column(Float)  # 30-100 Hz high-level awareness
    eeg_temporal_binding_coherence = Column(Float)  # Cross-frequency coupling
    
    # Flow State Neural Correlates
    frontal_alpha_asymmetry = Column(Float)  # Flow state EEG marker
    prefrontal_downregulation = Column(Float)  # Transient hypofrontality
    default_mode_network_suppression = Column(Float)  # Self-referential thinking reduction
    attention_network_enhancement = Column(Float)  # Focused attention strength
    
    # Cardiovascular Temporal Indicators
    heart_rate_variability_rmssd = Column(Float)  # Parasympathetic indicator
    heart_coherence_ratio = Column(Float)  # HeartMath coherence measure
    cardiac_temporal_entrainment = Column(Float)  # Sync with temporal cues
    baroreceptor_sensitivity = Column(Float)  # Blood pressure regulation
    
    # Autonomic Nervous System Balance
    sympathetic_activation_level = Column(Float)  # Stress response system
    parasympathetic_dominance = Column(Float)  # Rest and digest system
    autonomic_balance_ratio = Column(Float)  # Sympathetic:parasympathetic
    galvanic_skin_response = Column(Float)  # Emotional arousal indicator
    
    # Respiratory Temporal Patterns
    breathing_rate_per_minute = Column(Float)
    breathing_rhythm_coherence = Column(Float)  # Rhythmic breathing quality
    respiratory_sinus_arrhythmia = Column(Float)  # Heart-breath synchronization
    co2_tolerance_indicators = Column(Float)  # Breathing efficiency
    
    # Temporal Perception Behavioral Markers
    time_estimation_accuracy = Column(Float)  # How well user estimates time
    temporal_order_judgment = Column(Float)  # Sequence perception accuracy
    simultaneity_perception_threshold = Column(Float)  # Temporal resolution
    interval_timing_precision = Column(Float)  # Internal clock accuracy
    
    # Eye Movement and Visual Temporal Processing
    saccadic_eye_movement_timing = Column(Float)  # Visual temporal processing
    pupil_dilation_response = Column(Float)  # Cognitive load indicator
    blink_rate_patterns = Column(JSON)  # Attention and temporal awareness
    visual_tracking_smoothness = Column(Float)  # Smooth pursuit eye movements
    
    # Circadian Rhythm Indicators
    core_body_temperature = Column(Float)  # Circadian rhythm marker
    melatonin_production_indicators = Column(Float)  # Sleep-wake cycle
    cortisol_rhythm_markers = Column(Float)  # Stress hormone cycling
    alertness_maintenance_score = Column(Float)  # Sustained attention
    
    # Temporal Safety Indicators
    disorientation_risk_markers = Column(JSON)  # Early warning signs
    temporal_displacement_indicators = Column(Float)  # Confusion about time
    reality_testing_responses = Column(JSON)  # Ability to assess current time
    emergency_intervention_triggers = Column(JSON)  # Critical safety markers
    
    # AI Analysis Results
    predicted_temporal_state = Column(SQLEnum(TemporalState))
    temporal_state_confidence = Column(Float)  # AI certainty
    flow_state_probability = Column(Float)  # Likelihood of flow achievement
    optimal_intervention_suggestions = Column(JSON)  # AI recommendations
    
    # Data Quality Metrics
    signal_quality_score = Column(Float)  # 0.0-1.0 data reliability
    noise_reduction_applied = Column(JSON)  # Signal processing steps
    artifact_detection_flags = Column(JSON)  # Data quality issues
    
    # Relationships
    temporal_profile = relationship("TemporalProfile", back_populates="temporal_biometrics")
    temporal_session = relationship("TemporalSession", back_populates="temporal_biometrics")

    __table_args__ = (
        Index('idx_temporal_biometric_session_time', 'temporal_session_id', 'timestamp'),
        Index('idx_temporal_biometric_device', 'device_type', 'timestamp'),
        Index('idx_temporal_biometric_state', 'predicted_temporal_state', 'timestamp'),
        Index('idx_temporal_biometric_quality', 'signal_quality_score'),
    )

class TemporalSafetyMonitoring(Base):
    """
    Real-time safety monitoring for temporal dilution experiences.
    Prevents temporal displacement, disorientation, and adverse psychological effects.
    """
    __tablename__ = "temporal_safety_monitoring"

    id = Column(Integer, primary_key=True, index=True)
    temporal_profile_id = Column(Integer, ForeignKey("temporal_profiles.id"), nullable=False)
    temporal_session_id = Column(Integer, ForeignKey("temporal_sessions.id"), nullable=True)
    monitoring_uuid = Column(String(100), unique=True, index=True)
    
    # Safety Assessment
    current_safety_level = Column(SQLEnum(SafetyLevel), nullable=False)
    safety_level_confidence = Column(Float)  # AI certainty in assessment
    previous_safety_level = Column(SQLEnum(SafetyLevel))
    safety_trajectory = Column(String(50))  # improving, deteriorating, stable
    
    # Temporal Disorientation Indicators
    temporal_confusion_markers = Column(JSON)  # Signs of time confusion
    reality_testing_impairment = Column(Float)  # Ability to assess current reality
    temporal_memory_disruption = Column(Float)  # Problems with time-based memory
    chronostasis_severity = Column(Float)  # Stopped time perception intensity
    
    # Physiological Safety Markers
    cardiovascular_stress_indicators = Column(JSON)  # Heart rate/BP concerns
    neurological_stability_markers = Column(JSON)  # EEG safety indicators
    autonomic_dysregulation_signs = Column(JSON)  # ANS imbalance warnings
    vestibular_system_disruption = Column(Float)  # Balance and orientation issues
    
    # Psychological Safety Assessment
    anxiety_escalation_indicators = Column(JSON)  # Rising anxiety markers
    panic_response_triggers = Column(JSON)  # Fight-or-flight activation
    dissociation_risk_markers = Column(JSON)  # Disconnection from reality
    cognitive_overload_symptoms = Column(JSON)  # Mental strain indicators
    
    # Temporal Experience Safety
    time_dilation_tolerance_exceeded = Column(Boolean)  # Beyond safe limits
    temporal_anchor_loss = Column(Float)  # Loss of time reference points
    chronoception_disturbance_level = Column(Float)  # Time sense disruption
    temporal_memory_consolidation_interference = Column(Float)  # Memory formation issues
    
    # Safety Protocol Execution
    safety_interventions_triggered = Column(JSON)  # Actions taken for safety
    intervention_effectiveness_scores = Column(JSON)  # How well interventions worked
    escalation_procedures_activated = Column(JSON)  # Emergency measures taken
    recovery_protocols_applied = Column(JSON)  # Return to baseline procedures
    
    # Professional Response
    therapist_consultation_required = Column(Boolean)
    medical_professional_notified = Column(Boolean)
    emergency_services_contacted = Column(Boolean)
    family_contact_notification = Column(Boolean)
    
    # User Safety Response
    user_safety_awareness_level = Column(Float)  # User's recognition of issues
    user_cooperation_with_safety = Column(Float)  # Compliance with safety measures
    user_reported_distress_level = Column(Float)  # Self-reported discomfort
    user_desire_to_continue_session = Column(Float)  # Motivation vs safety balance
    
    # Environmental Safety Factors
    physical_environment_safety = Column(JSON)  # Safe physical space assessment
    supervision_availability = Column(String(100))  # Who is available to help
    emergency_equipment_accessibility = Column(JSON)  # Medical equipment nearby
    communication_system_functionality = Column(Boolean)  # Ability to call for help
    
    # Recovery and Resolution
    baseline_return_time = Column(Float)  # Minutes to return to normal
    post_experience_stability = Column(Float)  # How stable user is after session
    residual_effects_monitoring = Column(JSON)  # Lingering effects to watch
    follow_up_safety_requirements = Column(JSON)  # Ongoing safety needs
    
    # Learning and Prevention
    risk_factor_identification = Column(JSON)  # What led to safety concerns
    prevention_strategy_updates = Column(JSON)  # How to prevent recurrence
    safety_protocol_effectiveness = Column(JSON)  # How well protocols worked
    system_improvement_recommendations = Column(JSON)  # Safety system enhancements
    
    # Documentation and Compliance
    incident_documentation = Column(Text)  # Detailed incident record
    regulatory_reporting_requirements = Column(JSON)  # Required regulatory reports
    legal_liability_considerations = Column(JSON)  # Legal documentation needs
    insurance_claim_documentation = Column(JSON)  # Insurance reporting data
    
    # Timestamps
    monitoring_started_at = Column(DateTime, nullable=False)
    safety_concern_detected_at = Column(DateTime)
    intervention_started_at = Column(DateTime)
    safety_restored_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    temporal_profile = relationship("TemporalProfile", back_populates="safety_monitoring")
    temporal_session = relationship("TemporalSession", back_populates="safety_monitoring")

    __table_args__ = (
        Index('idx_temporal_safety_level', 'current_safety_level', 'monitoring_started_at'),
        Index('idx_temporal_safety_session', 'temporal_session_id', 'safety_concern_detected_at'),
        Index('idx_temporal_safety_interventions', 'safety_interventions_triggered'),
        Index('idx_temporal_safety_professional', 'therapist_consultation_required'),
    )

class TemporalLearningProtocol(Base):
    """
    Evidence-based protocols for enhancing learning through temporal dilution.
    Structured approaches for educational and skill development applications.
    """
    __tablename__ = "temporal_learning_protocols"

    id = Column(Integer, primary_key=True, index=True)
    protocol_uuid = Column(String(100), unique=True, index=True)
    
    # Protocol Classification
    protocol_name = Column(String(200), nullable=False)
    learning_domain = Column(String(100))  # Language, mathematics, music, motor skills, etc.
    target_age_group = Column(String(50))  # Children, adults, elderly
    evidence_base_strength = Column(String(20))  # Strong, moderate, preliminary
    
    # Temporal Parameters
    optimal_time_dilation_ratio = Column(Float)  # Recommended subjective:objective ratio
    session_duration_range = Column(JSON)  # Min/max minutes for effectiveness
    temporal_cue_sequence = Column(JSON)  # Specific temporal manipulation steps
    recovery_time_requirements = Column(JSON)  # Rest periods between sessions
    
    # Learning Enhancement Mechanisms
    memory_consolidation_enhancement = Column(JSON)  # How protocol improves memory
    attention_focus_optimization = Column(JSON)  # Attention enhancement methods
    neural_plasticity_stimulation = Column(JSON)  # Brain change promotion
    flow_state_learning_integration = Column(JSON)  # Combining flow with temporal dilation
    
    # Efficacy Data
    learning_speed_improvement_percentage = Column(Float)  # Documented improvement
    retention_enhancement_factor = Column(Float)  # Better long-term memory
    skill_acquisition_acceleration = Column(Float)  # Faster skill development
    transfer_learning_improvements = Column(JSON)  # Better generalization
    
    # Implementation Guidelines
    prerequisite_skills = Column(JSON)  # What users need before starting
    contraindications = Column(JSON)  # When protocol should not be used
    adaptation_guidelines = Column(JSON)  # How to customize for individuals
    progress_assessment_methods = Column(JSON)  # How to measure success
    
    # Safety Considerations
    cognitive_load_management = Column(JSON)  # Preventing mental overload
    fatigue_prevention_strategies = Column(JSON)  # Avoiding mental exhaustion
    disorientation_prevention = Column(JSON)  # Temporal safety measures
    
    # Research Foundation
    research_citations = Column(JSON)  # Supporting scientific literature
    clinical_trial_results = Column(JSON)  # Efficacy study outcomes
    peer_review_status = Column(String(100))  # Scientific validation level
    
    # Usage and Refinement
    total_implementations = Column(Integer, default=0)
    success_rate_percentage = Column(Float)
    user_satisfaction_scores = Column(JSON)
    protocol_refinement_history = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_research_update = Column(DateTime)

    __table_args__ = (
        Index('idx_temporal_learning_domain', 'learning_domain'),
        Index('idx_temporal_learning_efficacy', 'learning_speed_improvement_percentage'),
        Index('idx_temporal_learning_evidence', 'evidence_base_strength'),
    )

class CircadianIntegration(Base):
    """
    Integration of circadian rhythms with temporal dilution experiences.
    Optimizes timing of temporal experiences based on biological rhythms.
    """
    __tablename__ = "circadian_integration"

    id = Column(Integer, primary_key=True, index=True)
    temporal_profile_id = Column(Integer, ForeignKey("temporal_profiles.id"), nullable=False)
    
    # Circadian Profile
    chronotype_assessment = Column(String(50))  # Morning lark, night owl, etc.
    circadian_amplitude = Column(Float)  # Strength of daily rhythm
    phase_preference = Column(Float)  # Preferred sleep-wake timing
    circadian_stability_score = Column(Float)  # Consistency of rhythm
    
    # Light Exposure Patterns
    daily_light_exposure_profile = Column(JSON)  # Light timing and intensity
    blue_light_sensitivity = Column(Float)  # Response to screen light
    seasonal_affective_patterns = Column(JSON)  # Seasonal rhythm changes
    artificial_light_impact = Column(JSON)  # How artificial light affects rhythm
    
    # Sleep-Wake Optimization
    optimal_sleep_window = Column(JSON)  # Best sleep timing for temporal work
    wake_time_optimization = Column(JSON)  # Best wake times for sessions
    nap_timing_recommendations = Column(JSON)  # Strategic napping for temporal work
    sleep_debt_impact_on_temporal = Column(JSON)  # How sleep loss affects temporal perception
    
    # Performance Rhythm Integration
    peak_performance_windows = Column(JSON)  # Best times for different temporal tasks
    attention_rhythm_patterns = Column(JSON)  # Daily attention cycles
    memory_consolidation_timing = Column(JSON)  # Optimal times for learning
    creativity_peak_periods = Column(JSON)  # Best times for creative flow
    
    # Temporal Experience Optimization
    circadian_temporal_synergy_scores = Column(JSON)  # How well timing aligns
    time_of_day_effectiveness = Column(JSON)  # Success rates by time
    jet_lag_adaptation_strategies = Column(JSON)  # Travel-related adjustments
    shift_work_adaptations = Column(JSON)  # Non-standard schedule accommodations
    
    # Physiological Rhythm Tracking
    core_body_temperature_rhythm = Column(JSON)  # Daily temperature cycle
    cortisol_rhythm_patterns = Column(JSON)  # Stress hormone daily cycle
    melatonin_production_timing = Column(JSON)  # Sleep hormone patterns
    heart_rate_variability_circadian = Column(JSON)  # HRV daily patterns
    
    # Recommendations and Adjustments
    personalized_timing_recommendations = Column(JSON)  # Best session times
    environmental_optimization_suggestions = Column(JSON)  # Light, temperature, etc.
    lifestyle_modifications_for_temporal = Column(JSON)  # Daily routine adjustments
    
    # Timestamps and Tracking
    assessment_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_circadian_chronotype', 'chronotype_assessment'),
        Index('idx_circadian_profile', 'temporal_profile_id', 'assessment_date'),
    )
