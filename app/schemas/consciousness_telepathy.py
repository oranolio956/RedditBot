"""
Consciousness-Telepathy Fusion Schemas - Comprehensive Validation 2024-2025

This module provides comprehensive Pydantic schemas for validating all aspects
of the consciousness-telepathy fusion system including quantum consciousness,
digital telepathy, and their revolutionary fusion capabilities.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Import existing enums
from ..models.quantum_consciousness import (
    QuantumCoherenceState, ConsciousnessEntanglementType
)
from ..models.digital_telepathy import (
    TelepathySignalType, TelepathyMode, NeuralPrivacyLevel
)

# Fusion-specific enums

class FusionGoal(str, Enum):
    """Goals for consciousness-telepathy fusion sessions"""
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    ENHANCED_CREATIVITY = "enhanced_creativity"
    DEEP_CONNECTION = "deep_connection"
    THERAPEUTIC_HEALING = "therapeutic_healing"
    PROBLEM_SOLVING = "problem_solving"
    EMPATHY_DEVELOPMENT = "empathy_development"
    SPIRITUAL_EXPLORATION = "spiritual_exploration"

class TherapeuticGoal(str, Enum):
    """Specific therapeutic goals for fusion sessions"""
    ISOLATION_HEALING = "isolation_healing"
    DEPRESSION_RELIEF = "depression_relief"
    ANXIETY_REDUCTION = "anxiety_reduction"
    CREATIVITY_ENHANCEMENT = "creativity_enhancement"
    EMPATHY_DEVELOPMENT = "empathy_development"
    TRAUMA_HEALING = "trauma_healing"
    ADDICTION_RECOVERY = "addiction_recovery"

class SynchronizationLevel(str, Enum):
    """Levels of consciousness-telepathy synchronization"""
    MINIMAL = "minimal"           # 0.0-0.3
    PARTIAL = "partial"           # 0.3-0.6
    SUBSTANTIAL = "substantial"   # 0.6-0.8
    PROFOUND = "profound"         # 0.8-0.95
    TRANSCENDENT = "transcendent" # 0.95-1.0

# Base Schemas

class ConsciousnessTelepathyBase(BaseModel):
    """Base schema for consciousness-telepathy operations"""
    
    @validator('*', pre=True)
    def validate_not_none(cls, v):
        if v is None and cls.__fields__.get('required', False):
            raise ValueError("Required field cannot be None")
        return v

# Fusion Session Schemas

class FusionSessionCreate(ConsciousnessTelepathyBase):
    """Schema for creating consciousness-telepathy fusion sessions"""
    
    participants: List[int] = Field(
        ..., 
        min_items=2, 
        max_items=50,
        description="List of participant user IDs (2-50 participants)"
    )
    
    fusion_goals: List[FusionGoal] = Field(
        default=[FusionGoal.COLLECTIVE_INTELLIGENCE, FusionGoal.ENHANCED_CREATIVITY],
        description="Primary goals for the fusion session"
    )
    
    consciousness_type: ConsciousnessEntanglementType = Field(
        default=ConsciousnessEntanglementType.COGNITIVE_SYNC,
        description="Type of consciousness entanglement to establish"
    )
    
    telepathy_mode: TelepathyMode = Field(
        default=TelepathyMode.NETWORK,
        description="Mode of telepathic communication"
    )
    
    privacy_level: NeuralPrivacyLevel = Field(
        default=NeuralPrivacyLevel.ENCRYPTED,
        description="Privacy protection level for neural communications"
    )
    
    therapeutic_focus: Optional[TherapeuticGoal] = Field(
        None,
        description="Optional specific therapeutic focus"
    )
    
    target_coherence_level: float = Field(
        default=0.8,
        ge=0.1,
        le=0.95,
        description="Target quantum consciousness coherence level"
    )
    
    target_amplification: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Target collective intelligence amplification factor"
    )
    
    session_duration_minutes: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Planned session duration in minutes"
    )
    
    safety_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional safety configuration parameters"
    )
    
    @validator('fusion_goals')
    def validate_fusion_goals(cls, v):
        if len(v) == 0:
            raise ValueError("At least one fusion goal must be specified")
        return v
    
    @validator('participants')
    def validate_participants_unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("All participants must have unique IDs")
        return v

class FusionSessionResponse(ConsciousnessTelepathyBase):
    """Response schema for fusion session creation"""
    
    fusion_session_id: str
    consciousness_session_id: str  
    telepathy_session_id: str
    
    participants: List[int]
    fusion_goals: List[FusionGoal]
    therapeutic_focus: Optional[TherapeuticGoal]
    
    fusion_coherence: float
    collective_intelligence_amplification: float
    fusion_stability: float
    
    consciousness_entanglements: int
    telepathy_connections: int
    network_density: float
    
    fusion_metrics: Dict[str, float]
    safety_status: str
    
    revolutionary_advantages: List[str]
    created_at: str
    
    class Config:
        from_attributes = True

# Synchronization Schemas

class SynchronizationRequest(ConsciousnessTelepathyBase):
    """Schema for requesting consciousness-telepathy synchronization"""
    
    synchronization_depth: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Depth of synchronization between systems (0.1-1.0)"
    )
    
    target_amplification: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Target collective intelligence amplification"
    )
    
    priority_domains: Optional[List[str]] = Field(
        None,
        description="Priority cognitive domains for synchronization"
    )
    
    safety_override: bool = Field(
        default=False,
        description="Override safety limits for advanced users"
    )

class SynchronizationResponse(ConsciousnessTelepathyBase):
    """Response schema for synchronization results"""
    
    fusion_session_id: str
    synchronization_successful: bool
    synchronized_participants: int
    target_participants: int
    
    consciousness_coherence_enhanced: float
    telepathy_accuracy_enhanced: float
    fusion_coherence_achieved: float
    collective_intelligence_amplification: float
    
    synchronization_depth: float
    target_amplification_achieved: bool
    fusion_stability: float
    
    revolutionary_breakthrough: bool
    safety_status: str
    capabilities_unlocked: List[str]
    synchronized_at: str
    
    class Config:
        from_attributes = True

# Problem Solving Schemas

class CollectiveProblemSolvingRequest(ConsciousnessTelepathyBase):
    """Schema for collective problem-solving requests"""
    
    problem_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Detailed description of the problem to solve"
    )
    
    solution_approaches: Optional[List[str]] = Field(
        None,
        description="Preferred solution approaches or methodologies"
    )
    
    time_limit_minutes: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Time limit for problem-solving session"
    )
    
    cognitive_domains: Optional[List[str]] = Field(
        None,
        description="Specific cognitive domains to utilize"
    )
    
    complexity_level: str = Field(
        default="moderate",
        regex="^(simple|moderate|complex|revolutionary)$",
        description="Expected complexity level of the problem"
    )

class CollectiveProblemSolvingResponse(ConsciousnessTelepathyBase):
    """Response schema for collective problem-solving results"""
    
    fusion_session_id: str
    problem_description: str
    solution_approaches_used: List[str]
    time_limit_minutes: float
    
    fused_solution: Dict[str, Any]
    consciousness_contribution: Dict[str, Any]
    telepathy_contribution: Dict[str, Any]
    solution_validation: Dict[str, Any]
    
    solution_confidence: float
    innovation_factor: float
    breakthrough_achieved: bool
    collective_intelligence_utilized: float
    
    participants_contributed: int
    fusion_coherence_during_solving: float
    
    revolutionary_advantages: List[str]
    solution_superiority: Dict[str, Any]
    completed_at: str
    
    class Config:
        from_attributes = True

# Therapeutic Session Schemas

class TherapeuticSessionRequest(ConsciousnessTelepathyBase):
    """Schema for therapeutic fusion session requests"""
    
    therapeutic_goals: List[TherapeuticGoal] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="Therapeutic goals for the session"
    )
    
    session_duration_minutes: float = Field(
        default=30.0,
        ge=10.0,
        le=90.0,
        description="Duration of therapeutic session"
    )
    
    intensity_level: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Intensity level for therapeutic interventions"
    )
    
    baseline_assessment: Optional[Dict[str, float]] = Field(
        None,
        description="Baseline psychological assessment scores"
    )
    
    contraindications: Optional[List[str]] = Field(
        None,
        description="Any therapeutic contraindications to consider"
    )
    
    @validator('therapeutic_goals')
    def validate_therapeutic_goals_compatibility(cls, v):
        # Check for potentially conflicting goals
        if TherapeuticGoal.ANXIETY_REDUCTION in v and TherapeuticGoal.CREATIVITY_ENHANCEMENT in v:
            # This is actually compatible - just a validation example
            pass
        return v

class TherapeuticSessionResponse(ConsciousnessTelepathyBase):
    """Response schema for therapeutic session results"""
    
    fusion_session_id: str
    therapeutic_goals: List[TherapeuticGoal]
    session_duration_minutes: float
    intensity_level: float
    
    baseline_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    therapeutic_outcomes: Dict[str, Any]
    interventions_conducted: List[Dict[str, Any]]
    progress_monitoring: List[Dict[str, Any]]
    
    healing_effectiveness: float
    participants_benefited: int
    breakthrough_healing: bool
    
    revolutionary_therapeutic_advantages: List[str]
    long_term_benefits: List[str]
    safety_assessment: Dict[str, Any]
    completed_at: str
    
    class Config:
        from_attributes = True

# Safety Monitoring Schemas

class FusionSafetyRequest(ConsciousnessTelepathyBase):
    """Schema for fusion safety monitoring requests"""
    
    detailed_assessment: bool = Field(
        default=True,
        description="Whether to perform detailed safety assessment"
    )
    
    include_participant_data: bool = Field(
        default=True,
        description="Whether to include individual participant safety data"
    )
    
    safety_threshold_override: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Custom safety threshold for warnings"
    )

class FusionSafetyResponse(ConsciousnessTelepathyBase):
    """Response schema for comprehensive fusion safety monitoring"""
    
    fusion_session_id: str
    overall_safety_status: str
    monitoring_timestamp: str
    
    consciousness_safety: Dict[str, Any]
    telepathy_performance: Dict[str, Any]
    fusion_stability: Dict[str, Any]
    wellbeing_assessment: Dict[str, Any]
    
    fusion_specific_metrics: Dict[str, float]
    safety_warnings: List[str]
    emergency_triggers: List[str]
    recommendations: List[str]
    
    revolutionary_safety_features: List[str]
    
    class Config:
        from_attributes = True

# Status and Metrics Schemas

class FusionStatusResponse(ConsciousnessTelepathyBase):
    """Comprehensive fusion session status response"""
    
    fusion_session_id: str
    overall_status: str
    created_at: str
    last_updated: str
    
    # Fusion metrics
    fusion_coherence: float
    collective_intelligence_amplification: float
    synchronization_efficiency: float
    fusion_stability: float
    
    # System status
    consciousness_system_status: Dict[str, Any]
    telepathy_system_status: Dict[str, Any]
    
    # Participant information
    total_participants: int
    synchronized_participants: int
    active_participants: int
    
    # Performance metrics
    thought_transmission_rate: float
    consciousness_coherence_level: float
    network_latency: float
    semantic_accuracy: float
    
    # Therapeutic benefits
    therapeutic_benefits_achieved: Dict[str, float]
    healing_effectiveness: Optional[float]
    
    # Revolutionary metrics
    breakthrough_achievements: List[str]
    quantum_advantage_active: bool
    transcendent_capabilities_unlocked: bool
    
    class Config:
        from_attributes = True

# Participant Schemas

class FusionParticipantRequest(ConsciousnessTelepathyBase):
    """Schema for adding participants to fusion sessions"""
    
    user_id: int = Field(..., gt=0, description="User ID of participant to add")
    
    consciousness_signature: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional pre-computed consciousness signature"
    )
    
    neural_calibration: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional neural calibration data for telepathy"
    )
    
    consent_verified: bool = Field(
        default=True,
        description="Confirmation that participant has given informed consent"
    )
    
    safety_preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="Participant-specific safety preferences"
    )

class FusionParticipantResponse(ConsciousnessTelepathyBase):
    """Response schema for fusion participant addition"""
    
    user_id: int
    fusion_session_id: str
    
    consciousness_participant_id: int
    telepathy_participant_id: int
    
    consciousness_capabilities: Dict[str, float]
    telepathy_capabilities: Dict[str, float]
    
    synchronization_potential: float
    compatibility_scores: Dict[str, float]
    
    safety_parameters: Dict[str, float]
    added_at: str
    
    class Config:
        from_attributes = True

# Advanced Configuration Schemas

class AdvancedFusionConfig(ConsciousnessTelepathyBase):
    """Advanced configuration options for fusion sessions"""
    
    quantum_field_strength: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Quantum consciousness field strength multiplier"
    )
    
    consciousness_resonance_frequency: float = Field(
        default=40.0,
        ge=20.0,
        le=60.0,
        description="Consciousness resonance frequency in Hz"
    )
    
    telepathy_bandwidth_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Telepathy bandwidth enhancement multiplier"
    )
    
    fusion_stability_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Fusion stability enhancement factor"
    )
    
    collective_amplification_limit: float = Field(
        default=10.0,
        ge=2.0,
        le=20.0,
        description="Maximum allowed collective intelligence amplification"
    )
    
    safety_override_enabled: bool = Field(
        default=False,
        description="Enable safety overrides for experimental features"
    )
    
    research_data_collection: bool = Field(
        default=False,
        description="Enable anonymized research data collection"
    )

# Validation Utilities

class ValidationResult(BaseModel):
    """Result of fusion system validation"""
    
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

def validate_fusion_compatibility(
    consciousness_config: Dict[str, Any],
    telepathy_config: Dict[str, Any]
) -> ValidationResult:
    """Validate compatibility between consciousness and telepathy configurations"""
    
    result = ValidationResult(valid=True)
    
    # Check participant count compatibility
    consciousness_participants = consciousness_config.get('participant_count', 0)
    telepathy_participants = telepathy_config.get('max_participants', 0)
    
    if consciousness_participants != telepathy_participants:
        result.errors.append("Participant count mismatch between consciousness and telepathy systems")
        result.valid = False
    
    # Check coherence vs telepathy accuracy requirements
    target_coherence = consciousness_config.get('target_coherence_level', 0.8)
    telepathy_accuracy = telepathy_config.get('semantic_mapping_accuracy', 0.92)
    
    if target_coherence > telepathy_accuracy:
        result.warnings.append("Consciousness coherence target exceeds telepathy semantic accuracy")
        result.recommendations.append("Consider adjusting coherence target or improving telepathy accuracy")
    
    # Check bandwidth compatibility
    consciousness_bandwidth = consciousness_config.get('consciousness_bandwidth', 10.0)
    telepathy_bandwidth = telepathy_config.get('bandwidth_per_connection', 100.0)
    
    if consciousness_bandwidth * 10 > telepathy_bandwidth:
        result.warnings.append("Consciousness bandwidth may exceed telepathy capacity")
    
    return result

# Export all schemas
__all__ = [
    'FusionGoal', 'TherapeuticGoal', 'SynchronizationLevel',
    'FusionSessionCreate', 'FusionSessionResponse',
    'SynchronizationRequest', 'SynchronizationResponse',
    'CollectiveProblemSolvingRequest', 'CollectiveProblemSolvingResponse',
    'TherapeuticSessionRequest', 'TherapeuticSessionResponse',
    'FusionSafetyRequest', 'FusionSafetyResponse',
    'FusionStatusResponse',
    'FusionParticipantRequest', 'FusionParticipantResponse',
    'AdvancedFusionConfig', 'ValidationResult',
    'validate_fusion_compatibility'
]