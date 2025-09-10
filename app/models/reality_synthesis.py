"""
Reality Synthesis Matrix - Database Models

Revolutionary AI-powered reality manipulation system that creates seamless transitions
between multiple reality layers (physical, AR, VR, mixed) with spatial computing integration.
Based on 2024-2025 research including Neural Radiance Fields, spatial anchoring, and
cross-platform XR protocols for therapeutic and enhancement applications.

Key Features:
- Real-time Neural Radiance Field generation for 3D environment creation
- Portal-based reality switching with spatial anchor management
- Multi-user collaborative reality experiences with personal object awareness
- WebXR protocol implementation for universal cross-platform compatibility
- Safety protocols preventing reality dissociation and spatial disorientation
- Therapeutic reality protocols with measurable psychological outcomes
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from typing import Dict, Any, Optional, List

Base = declarative_base()

class RealityLayer(enum.Enum):
    """Different reality layer types for seamless transitions"""
    PHYSICAL_REALITY = "physical_reality"  # Base physical world
    AUGMENTED_REALITY = "augmented_reality"  # AR overlay on physical
    VIRTUAL_REALITY = "virtual_reality"  # Full VR immersion
    MIXED_REALITY = "mixed_reality"  # Hybrid physical-virtual
    SYNTHETIC_REALITY = "synthetic_reality"  # AI-generated environments
    THERAPEUTIC_REALITY = "therapeutic_reality"  # Clinical virtual environments
    COLLABORATIVE_REALITY = "collaborative_reality"  # Multi-user shared spaces
    TRANSITIONAL_REALITY = "transitional_reality"  # Between-layer experiences

class SpatialComputingPlatform(enum.Enum):
    """Supported spatial computing and XR platforms"""
    APPLE_VISION_PRO = "apple_vision_pro"  # Apple's mixed reality headset
    META_QUEST = "meta_quest"  # Meta's VR headset series
    HOLOLENS = "hololens"  # Microsoft's mixed reality device
    MAGIC_LEAP = "magic_leap"  # Magic Leap AR device
    VARJO_HEADSET = "varjo_headset"  # High-end professional VR/MR
    WEBXR_BROWSER = "webxr_browser"  # Browser-based WebXR
    MOBILE_AR = "mobile_ar"  # Smartphone AR (ARKit/ARCore)
    DESKTOP_VR = "desktop_vr"  # PC-based VR systems
    HOLOGRAPHIC_DISPLAY = "holographic_display"  # Volumetric displays
    CUSTOM_HARDWARE = "custom_hardware"  # Specialized equipment

class RealityRenderingEngine(enum.Enum):
    """3D rendering and reality generation engines"""
    NEURAL_RADIANCE_FIELDS = "neural_radiance_fields"  # NeRF-based generation
    GAUSSIAN_SPLATTING = "gaussian_splatting"  # 3D Gaussian Splatting
    UNREAL_ENGINE = "unreal_engine"  # Epic Games Unreal Engine
    UNITY_ENGINE = "unity_engine"  # Unity 3D engine
    CUSTOM_NEURAL_ENGINE = "custom_neural_engine"  # Proprietary AI engines
    PHOTOGRAMMETRY = "photogrammetry"  # Photo-based 3D reconstruction
    PROCEDURAL_GENERATION = "procedural_generation"  # Algorithm-based generation
    VOLUMETRIC_CAPTURE = "volumetric_capture"  # Live 3D capture systems

class RealityTransitionType(enum.Enum):
    """Types of transitions between reality layers"""
    GRADUAL_FADE = "gradual_fade"  # Smooth opacity transition
    PORTAL_JUMP = "portal_jump"  # Instant doorway transition
    MORPHING_BLEND = "morphing_blend"  # Geometric morphing between realities
    PARTICLE_DISSOLUTION = "particle_dissolution"  # Reality breaks apart into particles
    REALITY_RIPPLE = "reality_ripple"  # Wave-like reality change
    SPATIAL_WARP = "spatial_warp"  # Space-time distortion effect
    LAYER_PEEL = "layer_peel"  # Reality layers peel away
    QUANTUM_SHIFT = "quantum_shift"  # Instant quantum-like transition

class TherapeuticRealityProtocol(enum.Enum):
    """Clinical applications of reality synthesis for therapy"""
    EXPOSURE_THERAPY = "exposure_therapy"  # Gradual phobia treatment
    PTSD_PROCESSING = "ptsd_processing"  # Trauma reprocessing in safe space
    ANXIETY_MANAGEMENT = "anxiety_management"  # Calming environment therapy
    SOCIAL_SKILLS_TRAINING = "social_skills_training"  # Practice social situations
    COGNITIVE_BEHAVIORAL_THERAPY = "cognitive_behavioral_therapy"  # CBT in virtual space
    MINDFULNESS_MEDITATION = "mindfulness_meditation"  # Meditative reality environments
    PAIN_DISTRACTION = "pain_distraction"  # Immersive pain management
    ADDICTION_RECOVERY = "addiction_recovery"  # Craving management environments
    AUTISM_SUPPORT = "autism_support"  # Sensory-friendly therapeutic spaces
    DEMENTIA_CARE = "dementia_care"  # Memory-supportive environments

class SafetyLevel(enum.Enum):
    """Reality synthesis safety monitoring levels"""
    SAFE = "safe"  # Normal reality experience
    MILD_DISORIENTATION = "mild_disorientation"  # Slight spatial confusion
    MODERATE_CONCERN = "moderate_concern"  # Reality dissociation risk
    HIGH_RISK = "high_risk"  # Significant disorientation
    EMERGENCY_RESET = "emergency_reset"  # Immediate return to physical reality

class RealityProfile(Base):
    """
    User's comprehensive reality synthesis preferences, capabilities, and safety parameters.
    Tracks individual responses to different reality layers and optimal configurations.
    """
    __tablename__ = "reality_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Platform and Hardware Capabilities
    supported_platforms = Column(JSON)  # Available XR devices and platforms
    platform_preferences = Column(JSON)  # Preferred platforms for different activities
    hardware_performance_profile = Column(JSON)  # Device capabilities and limitations
    input_method_preferences = Column(JSON)  # Controllers, hand tracking, eye tracking
    
    # Spatial Computing Abilities
    spatial_awareness_score = Column(Float)  # 0.0-1.0 spatial understanding
    depth_perception_accuracy = Column(Float)  # 3D depth judgment ability
    hand_eye_coordination_vr = Column(Float)  # Coordination in virtual space
    motion_sickness_susceptibility = Column(Float)  # Tendency to get motion sick
    
    # Reality Layer Preferences
    preferred_reality_layers = Column(JSON)  # Favorite reality types
    reality_immersion_tolerance = Column(Float)  # How much immersion user can handle
    transition_speed_preference = Column(Float)  # Fast vs gradual reality changes
    reality_complexity_preference = Column(Float)  # Simple vs complex environments
    
    # Visual and Sensory Preferences
    visual_fidelity_requirements = Column(JSON)  # Graphics quality needs
    color_vision_characteristics = Column(JSON)  # Color perception abilities
    motion_blur_sensitivity = Column(Float)  # Sensitivity to motion blur
    field_of_view_preferences = Column(JSON)  # FOV comfort ranges
    
    # Audio and Haptic Preferences
    spatial_audio_preferences = Column(JSON)  # 3D audio settings
    haptic_feedback_sensitivity = Column(Float)  # Tactile feedback preferences
    audio_visual_synchronization = Column(Float)  # Tolerance for A/V sync issues
    
    # Comfort and Safety Parameters
    maximum_session_duration = Column(Float)  # Minutes before fatigue
    break_frequency_requirements = Column(JSON)  # How often breaks are needed
    comfort_zone_boundaries = Column(JSON)  # Safe experience parameters
    disorientation_recovery_time = Column(Float)  # Time to recover from confusion
    
    # Therapeutic Applications
    therapeutic_goals = Column(JSON)  # Clinical objectives for reality therapy
    contraindications = Column(JSON)  # Conditions preventing reality therapy
    therapist_supervision_level = Column(String(100))  # Required professional oversight
    emergency_contact_info = Column(JSON)  # Encrypted emergency contacts
    
    # Accessibility Accommodations
    visual_impairment_accommodations = Column(JSON)  # Visual accessibility needs
    hearing_impairment_accommodations = Column(JSON)  # Auditory accessibility needs
    motor_impairment_accommodations = Column(JSON)  # Physical accessibility needs
    cognitive_load_accommodations = Column(JSON)  # Cognitive accessibility needs
    
    # Social and Collaborative Preferences
    multiplayer_preferences = Column(JSON)  # Social reality experience preferences
    privacy_boundaries = Column(JSON)  # What personal data to share
    communication_preferences = Column(JSON)  # Voice, text, gesture preferences
    avatar_customization_preferences = Column(JSON)  # Virtual representation preferences
    
    # AI and Personalization
    neural_network_reality_weights = Column(JSON)  # Personalized AI parameters
    reality_adaptation_learning_data = Column(JSON)  # AI learning from user responses
    predictive_comfort_modeling = Column(JSON)  # AI-predicted user preferences
    
    # Performance and Optimization
    performance_priority_settings = Column(JSON)  # Quality vs performance preferences
    bandwidth_limitations = Column(JSON)  # Network capacity constraints
    local_processing_capabilities = Column(JSON)  # Edge computing resources
    
    # Timestamps and Tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_reality_session_at = Column(DateTime)
    profile_calibration_status = Column(String(100))
    
    # Relationships
    reality_sessions = relationship("RealitySession", back_populates="reality_profile")
    spatial_environments = relationship("SpatialEnvironment", back_populates="creator_profile")
    reality_safety_monitoring = relationship("RealitySafetyMonitoring", back_populates="reality_profile")
    collaborative_sessions = relationship("CollaborativeRealitySession", back_populates="reality_profile")

    __table_args__ = (
        Index('idx_reality_profile_user', 'user_id'),
        Index('idx_reality_profile_platforms', 'supported_platforms'),
        Index('idx_reality_profile_therapeutic', 'therapeutic_goals'),
    )

class RealitySession(Base):
    """
    Individual reality synthesis experience records with comprehensive tracking,
    multi-layer transitions, and outcome measurement.
    """
    __tablename__ = "reality_sessions"

    id = Column(Integer, primary_key=True, index=True)
    reality_profile_id = Column(Integer, ForeignKey("reality_profiles.id"), nullable=False)
    session_uuid = Column(String(100), unique=True, index=True)
    
    # Session Configuration
    primary_reality_layer = Column(SQLEnum(RealityLayer), nullable=False)
    session_type = Column(SQLEnum(TherapeuticRealityProtocol))
    target_duration_minutes = Column(Integer)
    rendering_engine = Column(SQLEnum(RealityRenderingEngine))
    platform_used = Column(SQLEnum(SpatialComputingPlatform))
    
    # Reality Layer Sequence
    reality_layer_transitions = Column(JSON)  # Ordered list of layer changes
    transition_timestamps = Column(JSON)  # When each transition occurred
    transition_types_used = Column(JSON)  # How each transition was executed
    layer_durations = Column(JSON)  # Time spent in each reality layer
    
    # Spatial Environment Data
    environments_visited = Column(JSON)  # List of 3D environments experienced
    environment_generation_parameters = Column(JSON)  # How environments were created
    spatial_anchor_points = Column(JSON)  # Fixed reference points in space
    coordinate_system_mapping = Column(JSON)  # Mapping between reality layers
    
    # Real-time Adaptation and AI
    ai_model_version = Column(String(50))
    real_time_adaptations = Column(JSON)  # Changes made during session
    user_behavior_adaptations = Column(JSON)  # Adaptations based on user actions
    physiological_adaptations = Column(JSON)  # Changes based on biometric data
    ai_decision_reasoning = Column(JSON)  # Why AI made specific changes
    
    # User Interaction and Performance
    user_actions_log = Column(JSON)  # All user inputs and interactions
    task_completion_metrics = Column(JSON)  # Objective task performance
    spatial_navigation_efficiency = Column(Float)  # How well user navigated 3D space
    hand_tracking_accuracy = Column(Float)  # Precision of hand movements
    eye_tracking_patterns = Column(JSON)  # Gaze patterns and attention
    
    # Immersion and Presence Quality
    presence_questionnaire_scores = Column(JSON)  # Validated presence measures
    immersion_depth_rating = Column(Float)  # How immersed user felt
    reality_confusion_incidents = Column(JSON)  # Times user confused virtual/real
    break_in_presence_events = Column(JSON)  # What broke the illusion
    
    # Multi-sensory Experience
    visual_quality_metrics = Column(JSON)  # Graphics rendering quality
    audio_spatialization_effectiveness = Column(Float)  # 3D audio quality
    haptic_feedback_quality = Column(Float)  # Tactile experience quality
    cross_modal_sensory_integration = Column(JSON)  # How well senses worked together
    
    # Social and Collaborative Elements
    other_users_present = Column(JSON)  # Other users in shared reality
    social_interaction_quality = Column(Float)  # Quality of social experiences
    avatar_embodiment_satisfaction = Column(Float)  # How well avatar represented user
    communication_effectiveness = Column(JSON)  # Quality of communication methods
    
    # Therapeutic Outcomes (if applicable)
    therapeutic_goals_progress = Column(JSON)  # Progress on clinical objectives
    symptom_improvement_indicators = Column(JSON)  # Clinical improvement markers
    therapeutic_alliance_strength = Column(Float)  # Relationship with virtual therapist
    homework_completion_in_vr = Column(JSON)  # Therapeutic exercises completed
    
    # Technical Performance Metrics
    frame_rate_statistics = Column(JSON)  # Rendering performance data
    latency_measurements = Column(JSON)  # System responsiveness metrics
    network_performance_data = Column(JSON)  # Bandwidth and connectivity quality
    hardware_utilization_stats = Column(JSON)  # CPU/GPU/memory usage
    
    # Safety and Comfort Monitoring
    motion_sickness_events = Column(JSON)  # Instances of nausea or discomfort
    eye_strain_indicators = Column(JSON)  # Visual fatigue markers
    physical_comfort_ratings = Column(JSON)  # Comfort throughout session
    safety_interventions_triggered = Column(JSON)  # Automatic safety responses
    
    # User Experience Quality
    overall_satisfaction_rating = Column(Float)  # 1-10 user satisfaction
    perceived_realism_score = Column(Float)  # How realistic experience felt
    ease_of_use_rating = Column(Float)  # User interface usability
    likelihood_to_recommend = Column(Float)  # Net Promoter Score equivalent
    
    # Learning and Skill Development
    skills_practiced_in_vr = Column(JSON)  # Skills worked on during session
    learning_objectives_achieved = Column(JSON)  # Educational goals met
    transfer_to_real_world_potential = Column(Float)  # Expected real-world application
    
    # Professional Oversight
    therapist_present = Column(Boolean)
    therapist_intervention_events = Column(JSON)  # When therapist intervened
    clinical_notes = Column(Text)  # Professional observations
    follow_up_recommendations = Column(JSON)
    
    # Data Export and Integration
    session_recording_available = Column(Boolean)  # If session was recorded
    exportable_data_formats = Column(JSON)  # Available export formats
    integration_with_other_systems = Column(JSON)  # Data shared with other platforms
    
    # Timestamps
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reality_profile = relationship("RealityProfile", back_populates="reality_sessions")
    spatial_environments = relationship("SpatialEnvironment", back_populates="sessions")
    reality_safety_monitoring = relationship("RealitySafetyMonitoring", back_populates="reality_session")
    collaborative_sessions = relationship("CollaborativeRealitySession", back_populates="reality_session")

    __table_args__ = (
        Index('idx_reality_session_layer_date', 'primary_reality_layer', 'started_at'),
        Index('idx_reality_session_platform', 'platform_used', 'started_at'),
        Index('idx_reality_session_therapeutic', 'session_type', 'therapeutic_goals_progress'),
    )

class SpatialEnvironment(Base):
    """
    AI-generated and curated 3D environments for reality synthesis experiences.
    Includes Neural Radiance Field data, spatial anchors, and multi-platform compatibility.
    """
    __tablename__ = "spatial_environments"

    id = Column(Integer, primary_key=True, index=True)
    environment_uuid = Column(String(100), unique=True, index=True)
    creator_profile_id = Column(Integer, ForeignKey("reality_profiles.id"), nullable=True)
    
    # Environment Classification
    environment_name = Column(String(200), nullable=False)
    environment_category = Column(String(100))  # Indoor, outdoor, abstract, therapeutic
    reality_layer_compatibility = Column(JSON)  # Which reality layers support this
    target_use_cases = Column(JSON)  # Education, therapy, entertainment, etc.
    
    # 3D Environment Data
    neural_radiance_field_data = Column(JSON)  # NeRF model parameters
    gaussian_splat_data = Column(JSON)  # 3D Gaussian Splatting data
    mesh_geometry_data = Column(JSON)  # Traditional 3D mesh data
    texture_asset_references = Column(JSON)  # Texture files and parameters
    lighting_configuration = Column(JSON)  # Lighting setup and parameters
    
    # Spatial Computing Integration
    spatial_anchors = Column(JSON)  # Fixed reference points in 3D space
    coordinate_system_definition = Column(JSON)  # World coordinate system
    collision_detection_data = Column(JSON)  # Physics collision boundaries
    occlusion_geometry = Column(JSON)  # What blocks what in 3D space
    
    # Multi-Platform Compatibility
    webxr_compatibility = Column(JSON)  # WebXR implementation data
    mobile_ar_optimization = Column(JSON)  # Mobile AR specific optimizations
    desktop_vr_configuration = Column(JSON)  # PC VR setup parameters
    mixed_reality_anchoring = Column(JSON)  # Real-world alignment data
    
    # AI Generation Parameters
    generation_prompt = Column(Text)  # Text prompt used to generate environment
    ai_model_used = Column(String(100))  # Which AI model created this
    generation_parameters = Column(JSON)  # Model hyperparameters
    post_processing_steps = Column(JSON)  # Modifications after generation
    
    # Interactive Elements
    interactive_objects = Column(JSON)  # Objects users can manipulate
    trigger_zones = Column(JSON)  # Areas that activate events
    dynamic_content_systems = Column(JSON)  # Content that changes over time
    physics_simulation_parameters = Column(JSON)  # Physics engine settings
    
    # Therapeutic Configuration
    therapeutic_applications = Column(JSON)  # Clinical uses for this environment
    therapeutic_effectiveness_data = Column(JSON)  # Outcome data for therapy use
    safety_considerations = Column(JSON)  # Potential risks or concerns
    accessibility_features = Column(JSON)  # Features for users with disabilities
    
    # Performance Optimization
    level_of_detail_settings = Column(JSON)  # LOD for different hardware
    rendering_optimization_data = Column(JSON)  # Performance tuning parameters
    bandwidth_requirements = Column(JSON)  # Network needs for streaming
    local_caching_strategy = Column(JSON)  # Edge computing optimization
    
    # Quality Metrics
    visual_fidelity_score = Column(Float)  # Objective visual quality
    user_rating_average = Column(Float)  # Average user satisfaction
    professional_quality_review = Column(JSON)  # Expert assessment
    technical_performance_metrics = Column(JSON)  # Frame rate, latency, etc.
    
    # Usage and Analytics
    total_sessions_hosted = Column(Integer, default=0)
    average_session_duration = Column(Float)  # Minutes per session
    user_retention_in_environment = Column(Float)  # How long users stay
    popular_interaction_patterns = Column(JSON)  # Common user behaviors
    
    # Content Moderation and Safety
    content_safety_rating = Column(String(20))  # G, PG, M, R equivalent
    moderation_status = Column(String(50))  # Approved, pending, flagged
    reported_issues = Column(JSON)  # User-reported problems
    safety_validation_checklist = Column(JSON)  # Safety review results
    
    # Version Control and Updates
    environment_version = Column(String(20))
    update_history = Column(JSON)  # Record of changes
    deprecated_versions = Column(JSON)  # Old versions and compatibility
    
    # Legal and Compliance
    intellectual_property_rights = Column(JSON)  # Ownership and licensing
    privacy_compliance_status = Column(JSON)  # GDPR, CCPA, etc. compliance
    content_attribution = Column(JSON)  # Credits for assets used
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime)

    # Relationships
    creator_profile = relationship("RealityProfile", back_populates="spatial_environments")
    sessions = relationship("RealitySession", back_populates="spatial_environments")

    __table_args__ = (
        Index('idx_spatial_environment_category', 'environment_category'),
        Index('idx_spatial_environment_quality', 'visual_fidelity_score', 'user_rating_average'),
        Index('idx_spatial_environment_therapeutic', 'therapeutic_applications'),
        Index('idx_spatial_environment_compatibility', 'reality_layer_compatibility'),
    )

class RealitySafetyMonitoring(Base):
    """
    Real-time safety monitoring for reality synthesis experiences.
    Prevents reality dissociation, spatial disorientation, and adverse psychological effects.
    """
    __tablename__ = "reality_safety_monitoring"

    id = Column(Integer, primary_key=True, index=True)
    reality_profile_id = Column(Integer, ForeignKey("reality_profiles.id"), nullable=False)
    reality_session_id = Column(Integer, ForeignKey("reality_sessions.id"), nullable=True)
    monitoring_uuid = Column(String(100), unique=True, index=True)
    
    # Safety Assessment
    current_safety_level = Column(SQLEnum(SafetyLevel), nullable=False)
    safety_level_confidence = Column(Float)  # AI certainty in assessment
    previous_safety_level = Column(SQLEnum(SafetyLevel))
    safety_trajectory = Column(String(50))  # improving, deteriorating, stable
    
    # Reality Dissociation Indicators
    reality_confusion_markers = Column(JSON)  # Signs of virtual/real confusion
    derealization_symptoms = Column(JSON)  # Feelings of unreality
    depersonalization_indicators = Column(JSON)  # Disconnection from self
    spatial_disorientation_level = Column(Float)  # Confusion about location/space
    
    # Physiological Safety Markers
    motion_sickness_severity = Column(Float)  # Nausea and discomfort level
    eye_strain_indicators = Column(JSON)  # Visual fatigue markers
    vestibular_system_disruption = Column(Float)  # Balance system problems
    cardiovascular_stress_markers = Column(JSON)  # Heart rate/BP concerns
    
    # Psychological Safety Assessment
    anxiety_escalation_signs = Column(JSON)  # Rising anxiety markers
    panic_response_triggers = Column(JSON)  # Fight-or-flight activation
    claustrophobia_reactions = Column(JSON)  # Enclosed space anxiety
    agoraphobia_responses = Column(JSON)  # Open space anxiety
    
    # Immersion Safety Monitoring
    presence_overload_indicators = Column(JSON)  # Too much immersion
    reality_layer_confusion = Column(JSON)  # Mixing up different layers
    transition_shock_events = Column(JSON)  # Jarring reality changes
    immersion_addiction_risk = Column(Float)  # Excessive VR use tendency
    
    # Platform-Specific Safety
    hardware_safety_alerts = Column(JSON)  # Device-related safety issues
    tracking_system_failures = Column(JSON)  # Loss of spatial tracking
    boundary_system_violations = Column(JSON)  # Guardian/playspace breaches
    hardware_overheating_warnings = Column(JSON)  # Device temperature issues
    
    # Cognitive Load Assessment
    information_overload_indicators = Column(JSON)  # Too much sensory input
    decision_paralysis_markers = Column(JSON)  # Overwhelmed by choices
    attention_fragmentation_signs = Column(JSON)  # Scattered focus
    cognitive_fatigue_symptoms = Column(JSON)  # Mental exhaustion
    
    # Social Safety (Multi-user Environments)
    harassment_detection_alerts = Column(JSON)  # Inappropriate behavior
    privacy_boundary_violations = Column(JSON)  # Personal space intrusions
    social_anxiety_escalation = Column(JSON)  # Social discomfort in VR
    avatar_embodiment_distress = Column(JSON)  # Discomfort with virtual body
    
    # Safety Protocol Execution
    safety_interventions_triggered = Column(JSON)  # Actions taken for safety
    intervention_effectiveness_scores = Column(JSON)  # How well interventions worked
    escalation_procedures_activated = Column(JSON)  # Emergency measures taken
    reality_reset_events = Column(JSON)  # Times system returned user to baseline
    
    # Professional Response
    therapist_consultation_required = Column(Boolean)
    medical_professional_notified = Column(Boolean)
    emergency_services_contacted = Column(Boolean)
    technical_support_escalation = Column(Boolean)
    
    # User Safety Response
    user_safety_awareness_level = Column(Float)  # User's recognition of issues
    user_cooperation_with_safety = Column(Float)  # Compliance with safety measures
    user_reported_discomfort_level = Column(Float)  # Self-reported issues
    user_desire_to_continue = Column(Float)  # Motivation vs safety balance
    
    # Environmental Safety Factors
    physical_space_safety_check = Column(JSON)  # Real-world safety assessment
    supervision_availability = Column(String(100))  # Who is available to help
    emergency_equipment_accessibility = Column(JSON)  # Medical equipment nearby
    escape_route_accessibility = Column(JSON)  # How to exit VR quickly
    
    # Recovery and Stabilization
    baseline_reality_return_time = Column(Float)  # Minutes to return to normal
    post_session_stability_assessment = Column(JSON)  # User state after VR
    residual_effects_monitoring = Column(JSON)  # Lingering effects to watch
    follow_up_safety_requirements = Column(JSON)  # Ongoing safety needs
    
    # Learning and Prevention
    risk_factor_identification = Column(JSON)  # What led to safety concerns
    prevention_strategy_updates = Column(JSON)  # How to prevent recurrence
    safety_protocol_effectiveness = Column(JSON)  # How well protocols worked
    system_improvement_recommendations = Column(JSON)  # Safety system enhancements
    
    # Documentation and Compliance
    incident_documentation = Column(Text)  # Detailed incident record
    regulatory_reporting_requirements = Column(JSON)  # Required reports
    legal_liability_considerations = Column(JSON)  # Legal documentation needs
    insurance_claim_documentation = Column(JSON)  # Insurance reporting data
    
    # Timestamps
    monitoring_started_at = Column(DateTime, nullable=False)
    safety_concern_detected_at = Column(DateTime)
    intervention_started_at = Column(DateTime)
    safety_restored_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reality_profile = relationship("RealityProfile", back_populates="reality_safety_monitoring")
    reality_session = relationship("RealitySession", back_populates="reality_safety_monitoring")

    __table_args__ = (
        Index('idx_reality_safety_level', 'current_safety_level', 'monitoring_started_at'),
        Index('idx_reality_safety_session', 'reality_session_id', 'safety_concern_detected_at'),
        Index('idx_reality_safety_interventions', 'safety_interventions_triggered'),
        Index('idx_reality_safety_dissociation', 'reality_confusion_markers'),
    )

class CollaborativeRealitySession(Base):
    """
    Multi-user shared reality experiences with personal object awareness,
    collaborative interaction tracking, and social safety monitoring.
    """
    __tablename__ = "collaborative_reality_sessions"

    id = Column(Integer, primary_key=True, index=True)
    reality_profile_id = Column(Integer, ForeignKey("reality_profiles.id"), nullable=False)
    reality_session_id = Column(Integer, ForeignKey("reality_sessions.id"), nullable=False)
    collaborative_session_uuid = Column(String(100), unique=True, index=True)
    
    # Session Configuration
    session_host_user_id = Column(Integer)  # Who initiated the session
    maximum_participants = Column(Integer)
    current_participant_count = Column(Integer)
    session_privacy_level = Column(String(50))  # Public, private, invite-only
    
    # Participant Management
    active_participants = Column(JSON)  # Currently connected users
    participant_roles = Column(JSON)  # Host, participant, observer roles
    participant_permissions = Column(JSON)  # What each user can do
    invitation_system_data = Column(JSON)  # How users join session
    
    # Shared Reality Environment
    synchronized_environment_state = Column(JSON)  # Shared world state
    personal_object_ownership = Column(JSON)  # Which user owns what objects
    shared_object_permissions = Column(JSON)  # Collaborative object access
    spatial_awareness_system = Column(JSON)  # How users see each other
    
    # Avatar and Representation
    avatar_appearance_data = Column(JSON)  # How users appear to each other
    avatar_behavior_synchronization = Column(JSON)  # Movement and gesture sync
    personal_space_boundaries = Column(JSON)  # Virtual personal space bubbles
    avatar_customization_sharing = Column(JSON)  # Shared appearance options
    
    # Communication Systems
    voice_communication_quality = Column(JSON)  # Spatial voice chat metrics
    text_communication_logs = Column(JSON)  # Chat messages and timestamps
    gesture_communication_data = Column(JSON)  # Non-verbal communication
    haptic_communication_events = Column(JSON)  # Touch-based communication
    
    # Collaborative Interaction Tracking
    shared_task_performance = Column(JSON)  # Group task completion metrics
    collaboration_effectiveness_score = Column(Float)  # How well group worked together
    leadership_emergence_patterns = Column(JSON)  # Who took charge when
    conflict_resolution_events = Column(JSON)  # How disagreements were handled
    
    # Social Dynamics Monitoring
    group_cohesion_indicators = Column(JSON)  # How well group bonded
    social_presence_quality = Column(Float)  # How "real" other users felt
    interpersonal_comfort_levels = Column(JSON)  # Comfort with other participants
    social_anxiety_markers = Column(JSON)  # Signs of social discomfort
    
    # Synchronization and Technical Quality
    latency_between_participants = Column(JSON)  # Network delay measurements
    synchronization_quality_metrics = Column(JSON)  # How well experiences synced
    bandwidth_usage_per_participant = Column(JSON)  # Network resource usage
    technical_issues_encountered = Column(JSON)  # Problems during session
    
    # Safety and Moderation
    harassment_incidents = Column(JSON)  # Inappropriate behavior events
    moderation_actions_taken = Column(JSON)  # How incidents were handled
    participant_reporting_events = Column(JSON)  # User reports of problems
    automated_safety_interventions = Column(JSON)  # System safety responses
    
    # Learning and Educational Outcomes
    collaborative_learning_objectives = Column(JSON)  # Educational goals
    peer_teaching_events = Column(JSON)  # Users helping each other learn
    group_problem_solving_effectiveness = Column(Float)  # Collective intelligence
    knowledge_sharing_quality = Column(JSON)  # How well information was shared
    
    # Therapeutic Collaborative Sessions
    group_therapy_session_type = Column(String(100))  # Type of group therapy
    therapeutic_alliance_between_participants = Column(JSON)  # Peer support quality
    shared_therapeutic_goals = Column(JSON)  # Common treatment objectives
    peer_support_effectiveness = Column(JSON)  # How participants helped each other
    
    # Cultural and Accessibility Considerations
    language_barriers_encountered = Column(JSON)  # Communication challenges
    cultural_sensitivity_events = Column(JSON)  # Cross-cultural interactions
    accessibility_accommodations_shared = Column(JSON)  # Collective accessibility needs
    timezone_coordination_challenges = Column(JSON)  # Global participant challenges
    
    # Session Quality and Satisfaction
    overall_session_satisfaction = Column(Float)  # Average participant rating
    likelihood_to_participate_again = Column(Float)  # Retention indicator
    most_valuable_aspects = Column(JSON)  # What participants liked most
    improvement_suggestions = Column(JSON)  # Participant feedback
    
    # Data Privacy and Sharing
    participant_consent_levels = Column(JSON)  # What data sharing was allowed
    data_recording_permissions = Column(JSON)  # What could be recorded
    post_session_data_retention = Column(JSON)  # How long data is kept
    cross_participant_data_sharing = Column(JSON)  # What data participants share
    
    # Timestamps
    session_started_at = Column(DateTime, nullable=False)
    session_ended_at = Column(DateTime)
    last_participant_joined_at = Column(DateTime)
    last_participant_left_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reality_profile = relationship("RealityProfile", back_populates="collaborative_sessions")
    reality_session = relationship("RealitySession", back_populates="collaborative_sessions")

    __table_args__ = (
        Index('idx_collaborative_session_participants', 'current_participant_count'),
        Index('idx_collaborative_session_host', 'session_host_user_id', 'session_started_at'),
        Index('idx_collaborative_session_quality', 'collaboration_effectiveness_score'),
        Index('idx_collaborative_session_safety', 'harassment_incidents', 'moderation_actions_taken'),
    )

class PortalSystem(Base):
    """
    Portal-based reality transition system for seamless movement between reality layers.
    Manages spatial anchors, transition effects, and cross-platform compatibility.
    """
    __tablename__ = "portal_systems"

    id = Column(Integer, primary_key=True, index=True)
    portal_uuid = Column(String(100), unique=True, index=True)
    
    # Portal Configuration
    portal_name = Column(String(200), nullable=False)
    source_reality_layer = Column(SQLEnum(RealityLayer), nullable=False)
    destination_reality_layer = Column(SQLEnum(RealityLayer), nullable=False)
    portal_type = Column(String(100))  # doorway, window, wormhole, fade_zone
    
    # Spatial Positioning
    source_spatial_coordinates = Column(JSON)  # Position in source reality
    destination_spatial_coordinates = Column(JSON)  # Position in destination reality
    spatial_anchor_data = Column(JSON)  # Fixed reference points
    coordinate_system_transformation = Column(JSON)  # Mapping between coordinate systems
    
    # Visual and Audio Design
    portal_visual_effects = Column(JSON)  # How portal appears visually
    transition_animation_data = Column(JSON)  # Animation during transition
    audio_cues_and_effects = Column(JSON)  # Sounds during transition
    haptic_feedback_patterns = Column(JSON)  # Tactile sensations
    
    # Transition Mechanics
    transition_trigger_conditions = Column(JSON)  # What activates the portal
    transition_duration_seconds = Column(Float)  # How long transition takes
    transition_smoothness_parameters = Column(JSON)  # Quality of transition
    loading_time_optimization = Column(JSON)  # Preloading strategies
    
    # Cross-Platform Compatibility
    webxr_portal_implementation = Column(JSON)  # Browser-based portal data
    mobile_ar_portal_optimization = Column(JSON)  # Mobile AR portal handling
    desktop_vr_portal_configuration = Column(JSON)  # PC VR portal setup
    mixed_reality_portal_anchoring = Column(JSON)  # Real-world alignment
    
    # Safety and User Experience
    motion_sickness_mitigation = Column(JSON)  # Comfort during transitions
    disorientation_prevention_measures = Column(JSON)  # Spatial awareness preservation
    emergency_exit_capabilities = Column(JSON)  # Quick escape from portal
    user_control_over_transition = Column(JSON)  # User agency in portal use
    
    # Performance Optimization
    preloading_strategies = Column(JSON)  # Content preparation for smooth transitions
    bandwidth_optimization = Column(JSON)  # Network efficiency measures
    rendering_pipeline_integration = Column(JSON)  # Graphics optimization
    memory_management_during_transition = Column(JSON)  # Resource management
    
    # Usage Analytics
    total_transitions_facilitated = Column(Integer, default=0)
    average_transition_success_rate = Column(Float)
    user_satisfaction_with_portal = Column(Float)
    technical_failure_rate = Column(Float)
    
    # Portal Relationships and Networks
    connected_portals = Column(JSON)  # Other portals this connects to
    portal_network_topology = Column(JSON)  # How portals connect together
    portal_access_permissions = Column(JSON)  # Who can use this portal
    portal_usage_restrictions = Column(JSON)  # Limits on portal use
    
    # Therapeutic and Educational Applications
    therapeutic_portal_applications = Column(JSON)  # Clinical uses
    educational_portal_scenarios = Column(JSON)  # Learning applications
    skill_training_portal_configurations = Column(JSON)  # Training applications
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime)

    __table_args__ = (
        Index('idx_portal_reality_layers', 'source_reality_layer', 'destination_reality_layer'),
        Index('idx_portal_performance', 'average_transition_success_rate'),
        Index('idx_portal_usage', 'total_transitions_facilitated'),
    )
