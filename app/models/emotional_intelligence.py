"""
Advanced Emotional Intelligence Models

Revolutionary multi-modal emotion detection and response system with:
- Multi-dimensional emotion modeling (Valence, Arousal, Dominance)
- Temporal emotional trajectory tracking
- Contextual emotion understanding
- Empathy scoring and development
- Crisis detection and intervention
- Emotional matching and therapeutic connections

Scientifically grounded using established psychological models:
- Russell's Circumplex Model of Affect
- Plutchik's Emotion Wheel
- Big Five personality correlations with emotional patterns
- Attachment theory emotional regulation patterns
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import json

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON, ForeignKey, 
    Index, CheckConstraint, DateTime, Enum as SQLEnum, ARRAY, Numeric
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class EmotionDimension(str, Enum):
    """Core emotional dimensions based on Russell's Circumplex Model."""
    VALENCE = "valence"      # Pleasant/Unpleasant (-1 to 1)
    AROUSAL = "arousal"      # High/Low activation (-1 to 1)
    DOMINANCE = "dominance"  # Control/Submission (-1 to 1)


class BasicEmotion(str, Enum):
    """Plutchik's 8 basic emotions with intensity variations."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    TRUST = "trust"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    ANTICIPATION = "anticipation"


class EmotionIntensity(str, Enum):
    """Emotion intensity levels."""
    MINIMAL = "minimal"      # 0.0-0.2
    LOW = "low"              # 0.2-0.4
    MODERATE = "moderate"    # 0.4-0.6
    HIGH = "high"            # 0.6-0.8
    EXTREME = "extreme"      # 0.8-1.0


class DetectionModality(str, Enum):
    """Methods of emotion detection."""
    TEXT_ANALYSIS = "text_analysis"
    VOICE_PROSODY = "voice_prosody"
    FACIAL_EXPRESSION = "facial_expression"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    PHYSIOLOGICAL = "physiological"
    SELF_REPORT = "self_report"
    CONTEXTUAL_INFERENCE = "contextual_inference"


class EmotionRegulationStrategy(str, Enum):
    """Evidence-based emotion regulation techniques."""
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    MINDFULNESS = "mindfulness"
    DEEP_BREATHING = "deep_breathing"
    EXPRESSIVE_WRITING = "expressive_writing"
    SOCIAL_SHARING = "social_sharing"
    PHYSICAL_EXERCISE = "physical_exercise"
    PROGRESSIVE_MUSCLE_RELAXATION = "progressive_muscle_relaxation"
    DISTRACTION = "distraction"
    ACCEPTANCE = "acceptance"


class CrisisLevel(str, Enum):
    """Crisis intervention priority levels."""
    NONE = "none"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_RISK = "high_risk"
    CRISIS = "crisis"
    EMERGENCY = "emergency"


class EmotionalProfile(FullAuditModel):
    """
    Comprehensive emotional profile for each user.
    
    Tracks long-term emotional patterns, personality correlations,
    and individual emotional intelligence development.
    """
    
    __tablename__ = "emotional_profiles"
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="User this emotional profile belongs to"
    )
    
    # Baseline emotional dimensions
    baseline_valence = Column(
        Float,
        nullable=True,
        comment="Average valence score (-1 to 1)"
    )
    
    baseline_arousal = Column(
        Float,
        nullable=True,
        comment="Average arousal score (-1 to 1)"
    )
    
    baseline_dominance = Column(
        Float,
        nullable=True,
        comment="Average dominance score (-1 to 1)"
    )
    
    # Emotional intelligence metrics
    empathy_quotient = Column(
        Float,
        nullable=True,
        comment="Empathy Quotient score (0-80, Baron-Cohen scale)"
    )
    
    emotional_quotient = Column(
        Float,
        nullable=True,
        comment="EQ score (0-200, Mayer-Salovey scale)"
    )
    
    alexithymia_score = Column(
        Float,
        nullable=True,
        comment="TAS-20 Alexithymia score (20-100)"
    )
    
    # Emotional regulation patterns
    primary_regulation_strategies = Column(
        ARRAY(String),
        nullable=True,
        comment="Most used emotion regulation strategies"
    )
    
    regulation_effectiveness = Column(
        JSONB,
        nullable=True,
        comment="Effectiveness scores for each regulation strategy"
    )
    
    # Personality correlations
    big_five_correlations = Column(
        JSONB,
        nullable=True,
        comment="Big Five personality trait correlations with emotions"
    )
    
    attachment_style = Column(
        String(50),
        nullable=True,
        comment="Attachment style (secure, anxious, avoidant, disorganized)"
    )
    
    # Emotional pattern recognition
    dominant_emotions = Column(
        JSONB,
        nullable=True,
        comment="Most frequently experienced emotions with frequencies"
    )
    
    emotion_transitions = Column(
        JSONB,
        nullable=True,
        comment="Common emotional state transition patterns"
    )
    
    trigger_patterns = Column(
        JSONB,
        nullable=True,
        comment="Identified emotional triggers and contexts"
    )
    
    # Social emotional patterns
    emotional_contagion_susceptibility = Column(
        Float,
        nullable=True,
        comment="How easily user catches others' emotions (0-1)"
    )
    
    social_emotion_matching = Column(
        JSONB,
        nullable=True,
        comment="Patterns of emotional matching with different user types"
    )
    
    # Crisis and support patterns
    crisis_indicators = Column(
        JSONB,
        nullable=True,
        comment="Personal crisis warning signs and thresholds"
    )
    
    support_preferences = Column(
        JSONB,
        nullable=True,
        comment="Preferred types of emotional support and interventions"
    )
    
    # Learning and development
    emotional_growth_metrics = Column(
        JSONB,
        nullable=True,
        comment="Tracked improvements in emotional intelligence over time"
    )
    
    coaching_history = Column(
        JSONB,
        nullable=True,
        comment="History of emotional coaching interventions and outcomes"
    )
    
    # Privacy and consent
    sharing_consent_level = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Level of emotional data sharing consent (1-5)"
    )
    
    crisis_intervention_consent = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Consent for crisis intervention outreach"
    )
    
    # Relationships
    user = relationship("User", back_populates="emotional_profile")
    emotion_readings = relationship("EmotionReading", back_populates="emotional_profile")
    empathy_assessments = relationship("EmpathyAssessment", back_populates="emotional_profile")
    
    __table_args__ = (
        CheckConstraint('baseline_valence >= -1 AND baseline_valence <= 1', name='ck_valence_range'),
        CheckConstraint('baseline_arousal >= -1 AND baseline_arousal <= 1', name='ck_arousal_range'),
        CheckConstraint('baseline_dominance >= -1 AND baseline_dominance <= 1', name='ck_dominance_range'),
        CheckConstraint('empathy_quotient >= 0 AND empathy_quotient <= 80', name='ck_empathy_range'),
        CheckConstraint('emotional_quotient >= 0 AND emotional_quotient <= 200', name='ck_eq_range'),
        CheckConstraint('alexithymia_score >= 20 AND alexithymia_score <= 100', name='ck_alexithymia_range'),
        CheckConstraint('sharing_consent_level >= 1 AND sharing_consent_level <= 5', name='ck_consent_range'),
        Index('idx_emotional_profile_user', 'user_id'),
        Index('idx_emotional_profile_empathy', 'empathy_quotient'),
        Index('idx_emotional_profile_eq', 'emotional_quotient'),
    )
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get comprehensive emotional profile summary."""
        return {
            "baseline_dimensions": {
                "valence": self.baseline_valence,
                "arousal": self.baseline_arousal,
                "dominance": self.baseline_dominance
            },
            "intelligence_metrics": {
                "empathy_quotient": self.empathy_quotient,
                "emotional_quotient": self.emotional_quotient,
                "alexithymia_score": self.alexithymia_score
            },
            "dominant_emotions": self.dominant_emotions,
            "primary_regulation_strategies": self.primary_regulation_strategies,
            "attachment_style": self.attachment_style,
            "crisis_risk_level": self.assess_current_crisis_level()
        }
    
    def assess_current_crisis_level(self) -> str:
        """Assess current crisis risk level based on recent patterns."""
        # This would be implemented with sophisticated ML algorithms
        # For now, return a basic assessment
        if not self.crisis_indicators:
            return CrisisLevel.NONE.value
        
        # Simplified risk assessment logic
        recent_indicators = self.crisis_indicators.get("recent_flags", [])
        severity_scores = self.crisis_indicators.get("severity_scores", {})
        
        if len(recent_indicators) >= 3 and max(severity_scores.values()) > 0.8:
            return CrisisLevel.CRISIS.value
        elif len(recent_indicators) >= 2 and max(severity_scores.values()) > 0.6:
            return CrisisLevel.HIGH_RISK.value
        elif len(recent_indicators) >= 1:
            return CrisisLevel.MODERATE_CONCERN.value
        
        return CrisisLevel.NONE.value


class EmotionReading(FullAuditModel):
    """
    Individual emotion detection result from multi-modal analysis.
    
    Stores specific emotion measurements with confidence scores,
    detection methods, and contextual information.
    """
    
    __tablename__ = "emotion_readings"
    
    emotional_profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotional_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Emotional profile this reading belongs to"
    )
    
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Message that triggered this emotion reading"
    )
    
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Conversation context for this reading"
    )
    
    # Core dimensional scores
    valence_score = Column(
        Float,
        nullable=False,
        comment="Valence dimension score (-1 to 1)"
    )
    
    arousal_score = Column(
        Float,
        nullable=False,
        comment="Arousal dimension score (-1 to 1)"
    )
    
    dominance_score = Column(
        Float,
        nullable=False,
        comment="Dominance dimension score (-1 to 1)"
    )
    
    # Basic emotion classification
    primary_emotion = Column(
        SQLEnum(BasicEmotion, name="basic_emotion"),
        nullable=False,
        index=True,
        comment="Primary detected emotion"
    )
    
    secondary_emotion = Column(
        SQLEnum(BasicEmotion, name="basic_emotion_secondary"),
        nullable=True,
        comment="Secondary detected emotion"
    )
    
    emotion_intensity = Column(
        SQLEnum(EmotionIntensity, name="emotion_intensity"),
        nullable=False,
        comment="Intensity level of primary emotion"
    )
    
    # Complex emotion analysis
    plutchik_wheel_position = Column(
        JSONB,
        nullable=True,
        comment="Position on Plutchik's emotion wheel with combinations"
    )
    
    emotion_blend_scores = Column(
        JSONB,
        nullable=False,
        comment="Scores for all 8 basic emotions (0-1)"
    )
    
    # Detection methodology
    detection_modalities = Column(
        ARRAY(String),
        nullable=False,
        comment="Methods used for this emotion detection"
    )
    
    confidence_scores = Column(
        JSONB,
        nullable=False,
        comment="Confidence scores for each detection modality"
    )
    
    # Contextual information
    context_factors = Column(
        JSONB,
        nullable=True,
        comment="Contextual factors affecting emotion (time, social, etc.)"
    )
    
    triggering_content = Column(
        Text,
        nullable=True,
        comment="Content that triggered this emotion (if consented)"
    )
    
    social_context = Column(
        String(100),
        nullable=True,
        comment="Social context (alone, group, public, private)"
    )
    
    # Analysis metadata
    processing_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken for emotion analysis in milliseconds"
    )
    
    model_versions = Column(
        JSONB,
        nullable=True,
        comment="Versions of ML models used for analysis"
    )
    
    analysis_quality_score = Column(
        Float,
        nullable=True,
        comment="Internal quality assessment of this analysis (0-1)"
    )
    
    # Relationships
    emotional_profile = relationship("EmotionalProfile", back_populates="emotion_readings")
    message = relationship("Message", backref="emotion_readings")
    conversation = relationship("Conversation", backref="emotion_readings")
    emotional_interactions = relationship("EmotionalInteraction", 
                                        foreign_keys="EmotionalInteraction.emotion_reading_id",
                                        back_populates="emotion_reading")
    
    __table_args__ = (
        CheckConstraint('valence_score >= -1 AND valence_score <= 1', name='ck_reading_valence_range'),
        CheckConstraint('arousal_score >= -1 AND arousal_score <= 1', name='ck_reading_arousal_range'),
        CheckConstraint('dominance_score >= -1 AND dominance_score <= 1', name='ck_reading_dominance_range'),
        CheckConstraint('analysis_quality_score >= 0 AND analysis_quality_score <= 1', name='ck_analysis_quality_range'),
        CheckConstraint('processing_time_ms >= 0', name='ck_processing_time_positive'),
        Index('idx_emotion_reading_profile_time', 'emotional_profile_id', 'created_at'),
        Index('idx_emotion_reading_primary_emotion', 'primary_emotion'),
        Index('idx_emotion_reading_intensity', 'emotion_intensity'),
        Index('idx_emotion_reading_valence_arousal', 'valence_score', 'arousal_score'),
    )
    
    def get_dimensional_coordinates(self) -> Dict[str, float]:
        """Get emotion as coordinates in 3D emotional space."""
        return {
            "valence": self.valence_score,
            "arousal": self.arousal_score,
            "dominance": self.dominance_score
        }
    
    def calculate_emotional_distance(self, other_reading: 'EmotionReading') -> float:
        """Calculate emotional distance from another reading."""
        import math
        
        valence_diff = (self.valence_score - other_reading.valence_score) ** 2
        arousal_diff = (self.arousal_score - other_reading.arousal_score) ** 2
        dominance_diff = (self.dominance_score - other_reading.dominance_score) ** 2
        
        return math.sqrt(valence_diff + arousal_diff + dominance_diff)
    
    def is_crisis_indicator(self) -> bool:
        """Check if this reading indicates potential crisis."""
        # Crisis patterns: High arousal + low valence + low dominance
        if (self.arousal_score > 0.5 and 
            self.valence_score < -0.5 and 
            self.dominance_score < -0.3):
            return True
        
        # Extreme negative emotions
        if (self.primary_emotion in [BasicEmotion.SADNESS, BasicEmotion.FEAR, BasicEmotion.ANGER] and
            self.emotion_intensity in [EmotionIntensity.HIGH, EmotionIntensity.EXTREME]):
            return True
        
        return False


class EmotionalInteraction(FullAuditModel):
    """
    Records of emotional exchanges between users.
    
    Tracks how emotions spread, influence, and evolve through social interactions.
    Used for emotional contagion analysis and social emotional intelligence.
    """
    
    __tablename__ = "emotional_interactions"
    
    # Source and target users
    source_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who initiated the emotional interaction"
    )
    
    target_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who received the emotional interaction"
    )
    
    # Emotional states
    emotion_reading_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotion_readings.id", ondelete="CASCADE"),
        nullable=False,
        comment="Source emotion reading that initiated interaction"
    )
    
    response_emotion_reading_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotion_readings.id", ondelete="SET NULL"),
        nullable=True,
        comment="Target user's emotional response reading"
    )
    
    # Interaction context
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Conversation where interaction occurred"
    )
    
    interaction_type = Column(
        String(50),
        nullable=False,
        comment="Type of emotional interaction (empathy, comfort, challenge, etc.)"
    )
    
    # Emotional influence metrics
    emotional_contagion_score = Column(
        Float,
        nullable=True,
        comment="How much target user's emotion was influenced (0-1)"
    )
    
    empathy_demonstrated = Column(
        Float,
        nullable=True,
        comment="Level of empathy demonstrated by source user (0-1)"
    )
    
    emotional_support_provided = Column(
        Float,
        nullable=True,
        comment="Quality of emotional support provided (0-1)"
    )
    
    # Outcome measurement
    target_emotion_change = Column(
        JSONB,
        nullable=True,
        comment="Measured change in target's emotional state"
    )
    
    interaction_effectiveness = Column(
        Float,
        nullable=True,
        comment="Overall effectiveness of emotional interaction (0-1)"
    )
    
    # Relationship quality impact
    relationship_strengthening = Column(
        Float,
        nullable=True,
        comment="How much this interaction strengthened the relationship (0-1)"
    )
    
    mutual_understanding_gain = Column(
        Float,
        nullable=True,
        comment="Increase in mutual emotional understanding (0-1)"
    )
    
    # Relationships
    source_user = relationship("User", foreign_keys=[source_user_id], backref="initiated_emotional_interactions")
    target_user = relationship("User", foreign_keys=[target_user_id], backref="received_emotional_interactions")
    emotion_reading = relationship("EmotionReading", foreign_keys=[emotion_reading_id], back_populates="emotional_interactions")
    response_emotion_reading = relationship("EmotionReading", foreign_keys=[response_emotion_reading_id])
    conversation = relationship("Conversation", backref="emotional_interactions")
    
    __table_args__ = (
        CheckConstraint('source_user_id != target_user_id', name='ck_different_users'),
        CheckConstraint('emotional_contagion_score >= 0 AND emotional_contagion_score <= 1', name='ck_contagion_range'),
        CheckConstraint('empathy_demonstrated >= 0 AND empathy_demonstrated <= 1', name='ck_empathy_range'),
        CheckConstraint('emotional_support_provided >= 0 AND emotional_support_provided <= 1', name='ck_support_range'),
        CheckConstraint('interaction_effectiveness >= 0 AND interaction_effectiveness <= 1', name='ck_effectiveness_range'),
        Index('idx_emotional_interaction_users', 'source_user_id', 'target_user_id'),
        Index('idx_emotional_interaction_conversation', 'conversation_id'),
        Index('idx_emotional_interaction_type', 'interaction_type'),
        Index('idx_emotional_interaction_effectiveness', 'interaction_effectiveness'),
    )


class EmotionTrajectory(BaseModel):
    """
    Tracks emotional state changes over time for pattern recognition.
    
    Captures temporal dynamics of emotional states to identify:
    - Emotional cycles and patterns
    - Triggering events and recovery times
    - Emotional regulation effectiveness
    - Long-term emotional development trends
    """
    
    __tablename__ = "emotion_trajectories"
    
    emotional_profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotional_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Emotional profile this trajectory belongs to"
    )
    
    # Time window for this trajectory segment
    start_time = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of this trajectory segment"
    )
    
    end_time = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="End of this trajectory segment"
    )
    
    # Trajectory characteristics
    trajectory_type = Column(
        String(50),
        nullable=False,
        comment="Type of trajectory (daily, weekly, episode, crisis, recovery)"
    )
    
    # Emotional path data
    emotion_sequence = Column(
        JSONB,
        nullable=False,
        comment="Sequence of emotional states with timestamps"
    )
    
    dimensional_path = Column(
        JSONB,
        nullable=False,
        comment="Path through valence-arousal-dominance space over time"
    )
    
    # Pattern analysis
    dominant_pattern = Column(
        String(100),
        nullable=True,
        comment="Identified dominant emotional pattern"
    )
    
    pattern_stability = Column(
        Float,
        nullable=True,
        comment="Stability score of emotional patterns (0-1)"
    )
    
    volatility_score = Column(
        Float,
        nullable=True,
        comment="Emotional volatility during this period (0-1)"
    )
    
    # Trigger and context analysis
    identified_triggers = Column(
        JSONB,
        nullable=True,
        comment="Events or contexts that triggered emotional changes"
    )
    
    trigger_response_patterns = Column(
        JSONB,
        nullable=True,
        comment="Patterns of emotional responses to different triggers"
    )
    
    # Regulation and recovery
    regulation_attempts = Column(
        JSONB,
        nullable=True,
        comment="Emotion regulation strategies attempted during this period"
    )
    
    regulation_effectiveness = Column(
        JSONB,
        nullable=True,
        comment="Effectiveness of different regulation strategies"
    )
    
    recovery_time_minutes = Column(
        Integer,
        nullable=True,
        comment="Average time to recover from negative emotions"
    )
    
    # Social and environmental factors
    social_influences = Column(
        JSONB,
        nullable=True,
        comment="Social factors that influenced emotions during this period"
    )
    
    environmental_factors = Column(
        JSONB,
        nullable=True,
        comment="Environmental contexts affecting emotions"
    )
    
    # Analysis metadata
    analysis_confidence = Column(
        Float,
        nullable=True,
        comment="Confidence in trajectory analysis (0-1)"
    )
    
    data_completeness = Column(
        Float,
        nullable=True,
        comment="Completeness of data for this trajectory (0-1)"
    )
    
    # Relationships
    emotional_profile = relationship("EmotionalProfile", backref="emotion_trajectories")
    
    __table_args__ = (
        CheckConstraint('start_time < end_time', name='ck_trajectory_time_order'),
        CheckConstraint('pattern_stability >= 0 AND pattern_stability <= 1', name='ck_stability_range'),
        CheckConstraint('volatility_score >= 0 AND volatility_score <= 1', name='ck_volatility_range'),
        CheckConstraint('analysis_confidence >= 0 AND analysis_confidence <= 1', name='ck_confidence_range'),
        CheckConstraint('data_completeness >= 0 AND data_completeness <= 1', name='ck_completeness_range'),
        CheckConstraint('recovery_time_minutes >= 0', name='ck_recovery_time_positive'),
        Index('idx_emotion_trajectory_profile_time', 'emotional_profile_id', 'start_time'),
        Index('idx_emotion_trajectory_type', 'trajectory_type'),
        Index('idx_emotion_trajectory_pattern', 'dominant_pattern'),
    )


class EmpathyAssessment(FullAuditModel):
    """
    Tracks empathy development and emotional intelligence growth.
    
    Records assessments, coaching interventions, and progress in
    emotional intelligence and empathic abilities.
    """
    
    __tablename__ = "empathy_assessments"
    
    emotional_profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emotional_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Emotional profile being assessed"
    )
    
    # Assessment type and methodology
    assessment_type = Column(
        String(50),
        nullable=False,
        comment="Type of empathy assessment (formal, behavioral, AI-observed)"
    )
    
    assessment_methodology = Column(
        String(100),
        nullable=False,
        comment="Specific methodology used (EQ-i 2.0, Baron-Cohen EQ, behavioral analysis)"
    )
    
    # Core empathy dimensions
    cognitive_empathy_score = Column(
        Float,
        nullable=True,
        comment="Ability to understand others' emotions (0-100)"
    )
    
    affective_empathy_score = Column(
        Float,
        nullable=True,
        comment="Ability to share others' emotions (0-100)"
    )
    
    compassionate_empathy_score = Column(
        Float,
        nullable=True,
        comment="Ability to take action to help others (0-100)"
    )
    
    # Specific skill areas
    emotion_recognition_accuracy = Column(
        Float,
        nullable=True,
        comment="Accuracy in recognizing others' emotions (0-1)"
    )
    
    perspective_taking_ability = Column(
        Float,
        nullable=True,
        comment="Ability to take others' perspectives (0-100)"
    )
    
    emotional_responsiveness = Column(
        Float,
        nullable=True,
        comment="Appropriateness of emotional responses (0-100)"
    )
    
    # Social emotional intelligence
    social_awareness_score = Column(
        Float,
        nullable=True,
        comment="Awareness of social emotional dynamics (0-100)"
    )
    
    relationship_management_score = Column(
        Float,
        nullable=True,
        comment="Ability to manage emotional aspects of relationships (0-100)"
    )
    
    # Behavioral evidence
    observed_empathic_behaviors = Column(
        JSONB,
        nullable=True,
        comment="Specific empathic behaviors observed with contexts"
    )
    
    empathy_failures = Column(
        JSONB,
        nullable=True,
        comment="Situations where empathy was lacking with analysis"
    )
    
    # Development and coaching
    targeted_skill_areas = Column(
        ARRAY(String),
        nullable=True,
        comment="Areas identified for empathy development"
    )
    
    coaching_interventions = Column(
        JSONB,
        nullable=True,
        comment="Coaching activities and interventions provided"
    )
    
    progress_since_last_assessment = Column(
        JSONB,
        nullable=True,
        comment="Measured progress in different empathy dimensions"
    )
    
    # Assessment context
    assessment_context = Column(
        String(200),
        nullable=True,
        comment="Context and circumstances of this assessment"
    )
    
    peer_feedback_included = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether peer feedback was included in assessment"
    )
    
    # Quality and validation
    assessment_reliability = Column(
        Float,
        nullable=True,
        comment="Reliability score of this assessment (0-1)"
    )
    
    assessor_type = Column(
        String(50),
        nullable=False,
        comment="Type of assessor (AI, self, peer, professional)"
    )
    
    # Relationships
    emotional_profile = relationship("EmotionalProfile", back_populates="empathy_assessments")
    
    __table_args__ = (
        CheckConstraint('cognitive_empathy_score >= 0 AND cognitive_empathy_score <= 100', name='ck_cognitive_empathy_range'),
        CheckConstraint('affective_empathy_score >= 0 AND affective_empathy_score <= 100', name='ck_affective_empathy_range'),
        CheckConstraint('compassionate_empathy_score >= 0 AND compassionate_empathy_score <= 100', name='ck_compassionate_empathy_range'),
        CheckConstraint('emotion_recognition_accuracy >= 0 AND emotion_recognition_accuracy <= 1', name='ck_recognition_accuracy_range'),
        CheckConstraint('assessment_reliability >= 0 AND assessment_reliability <= 1', name='ck_reliability_range'),
        Index('idx_empathy_assessment_profile_time', 'emotional_profile_id', 'created_at'),
        Index('idx_empathy_assessment_type', 'assessment_type'),
        Index('idx_empathy_assessment_scores', 'cognitive_empathy_score', 'affective_empathy_score'),
    )


# User model relationships (added to avoid circular imports)
# These would be added to the User model in app/models/user.py:
"""
# Add to User model:
emotional_profile = relationship("EmotionalProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
"""