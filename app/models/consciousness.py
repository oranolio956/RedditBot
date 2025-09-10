"""
Consciousness Profile Models

Database models for storing cognitive twins, personality evolution,
and consciousness mirroring data.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
import json

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, 
    ForeignKey, JSON, Index, DateTime, ARRAY
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
import numpy as np

from app.database.base import Base, FullAuditModel


class CognitiveProfile(FullAuditModel):
    """
    Stores the complete cognitive profile of a user's digital twin.
    This is the consciousness mirror that thinks like the user.
    """
    
    __tablename__ = "cognitive_profiles"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="cognitive_profile")
    
    # Big Five Personality Scores (0-1 scale)
    openness = Column(Float, default=0.5, nullable=False)
    conscientiousness = Column(Float, default=0.5, nullable=False)
    extraversion = Column(Float, default=0.5, nullable=False)
    agreeableness = Column(Float, default=0.5, nullable=False)
    neuroticism = Column(Float, default=0.5, nullable=False)
    
    # Cognitive Metrics
    mirror_accuracy = Column(Float, default=0.0, nullable=False)  # How well we mirror (0-1)
    thought_velocity = Column(Float, default=1.0, nullable=False)  # Speed of thinking
    creativity_index = Column(Float, default=0.5, nullable=False)  # Divergent thinking
    prediction_confidence = Column(Float, default=0.0, nullable=False)  # Response prediction accuracy
    
    # Linguistic Fingerprint (stored as JSONB for complex queries)
    linguistic_fingerprint = Column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Word choice patterns, formality, complexity metrics"
    )
    
    # Decision Patterns
    decision_patterns = Column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Historical decision-making patterns and preferences"
    )
    
    # Cognitive Biases
    cognitive_biases = Column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Identified cognitive biases and their strengths"
    )
    
    # Response Templates (common phrases/patterns)
    response_templates = Column(
        JSONB,
        default=list,
        nullable=False,
        comment="Common response patterns used by the user"
    )
    
    # Emotional Baseline (8 basic emotions)
    emotional_baseline = Column(
        JSONB,
        default=lambda: [0.5] * 8,
        nullable=False,
        comment="Default emotional state [joy, sadness, anger, fear, surprise, disgust, trust, anticipation]"
    )
    
    # Temporal Evolution History
    personality_history = Column(
        JSONB,
        default=list,
        nullable=False,
        comment="Historical personality vectors over time"
    )
    
    # Last synchronization with user
    last_sync = Column(DateTime, default=datetime.utcnow, nullable=False)
    sync_message_count = Column(Integer, default=0, nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_cognitive_user_id', 'user_id'),
        Index('idx_cognitive_accuracy', 'mirror_accuracy'),
        Index('idx_cognitive_sync', 'last_sync'),
    )
    
    def get_personality_vector(self) -> np.ndarray:
        """Get Big Five personality as numpy array."""
        return np.array([
            self.openness,
            self.conscientiousness,
            self.extraversion,
            self.agreeableness,
            self.neuroticism
        ])
    
    def set_personality_vector(self, vector: np.ndarray):
        """Set Big Five personality from numpy array."""
        if len(vector) != 5:
            raise ValueError("Personality vector must have 5 dimensions")
        
        self.openness = float(vector[0])
        self.conscientiousness = float(vector[1])
        self.extraversion = float(vector[2])
        self.agreeableness = float(vector[3])
        self.neuroticism = float(vector[4])
    
    def add_personality_snapshot(self):
        """Add current personality to history."""
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'vector': self.get_personality_vector().tolist(),
            'accuracy': self.mirror_accuracy
        }
        
        if not self.personality_history:
            self.personality_history = []
        
        self.personality_history.append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.personality_history) > 1000:
            self.personality_history = self.personality_history[-1000:]


class KeystrokePattern(FullAuditModel):
    """
    Stores keystroke dynamics patterns for personality assessment.
    Used to infer cognitive and emotional states from typing behavior.
    """
    
    __tablename__ = "keystroke_patterns"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="keystroke_patterns")
    
    # Session identifier
    session_id = Column(String(64), nullable=False)
    
    # Keystroke Metrics
    avg_dwell_time = Column(Float, comment="Average key hold duration (ms)")
    avg_flight_time = Column(Float, comment="Average time between keys (ms)")
    typing_speed = Column(Float, comment="Words per minute")
    rhythm_variance = Column(Float, comment="Typing rhythm consistency")
    deletion_rate = Column(Float, comment="Correction frequency")
    pause_frequency = Column(Float, comment="Thinking pause frequency")
    
    # Emotional Indicators
    emotional_pressure = Column(Float, comment="Pressure/speed correlation")
    stress_indicator = Column(Float, comment="Stress level from patterns")
    confidence_score = Column(Float, comment="Typing confidence")
    
    # Raw pattern data (for ML training)
    raw_patterns = Column(
        JSONB,
        comment="Raw keystroke timing data"
    )
    
    # Associated message
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    message = relationship("Message", back_populates="keystroke_pattern")
    
    # Indexes
    __table_args__ = (
        Index('idx_keystroke_user_session', 'user_id', 'session_id'),
        Index('idx_keystroke_created', 'created_at'),
    )


class ConsciousnessSession(FullAuditModel):
    """
    Represents a conversation session with the consciousness mirror.
    Tracks interactions between user and their cognitive twin.
    """
    
    __tablename__ = "consciousness_sessions"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="consciousness_sessions")
    
    # Session type
    session_type = Column(
        String(50),
        nullable=False,
        comment="twin_chat, future_self, past_self, decision_helper"
    )
    
    # Twin configuration
    twin_config = Column(
        JSONB,
        default=dict,
        comment="Configuration for this twin session (e.g., years_ahead for future self)"
    )
    
    # Session metrics
    message_count = Column(Integer, default=0)
    prediction_accuracy = Column(Float, comment="Average prediction accuracy this session")
    user_satisfaction = Column(Float, comment="User rating of twin accuracy")
    
    # Conversation log
    conversation_log = Column(
        JSONB,
        default=list,
        comment="Full conversation history with predictions and actuals"
    )
    
    # Session state
    is_active = Column(Boolean, default=True)
    ended_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_consciousness_session_user', 'user_id'),
        Index('idx_consciousness_session_active', 'is_active'),
        Index('idx_consciousness_session_type', 'session_type'),
    )
    
    def add_exchange(self, user_message: str, twin_response: str, confidence: float):
        """Add a conversation exchange to the log."""
        exchange = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user_message,
            'twin': twin_response,
            'confidence': confidence
        }
        
        if not self.conversation_log:
            self.conversation_log = []
        
        self.conversation_log.append(exchange)
        self.message_count += 1


class PersonalityEvolution(FullAuditModel):
    """
    Tracks the evolution of personality over time.
    Used for predicting future personality states and understanding change.
    """
    
    __tablename__ = "personality_evolution"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="personality_evolution")
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Personality change metrics
    openness_delta = Column(Float, default=0.0)
    conscientiousness_delta = Column(Float, default=0.0)
    extraversion_delta = Column(Float, default=0.0)
    agreeableness_delta = Column(Float, default=0.0)
    neuroticism_delta = Column(Float, default=0.0)
    
    # Change drivers
    major_events = Column(
        JSONB,
        default=list,
        comment="Significant events that may have influenced personality"
    )
    
    # Prediction vs Reality
    predicted_vector = Column(
        JSONB,
        comment="What we predicted personality would be"
    )
    actual_vector = Column(
        JSONB,
        comment="What personality actually became"
    )
    prediction_error = Column(Float, comment="Prediction error magnitude")
    
    # Environmental factors
    environmental_factors = Column(
        JSONB,
        default=dict,
        comment="External factors affecting personality (stress, relationships, etc.)"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_evolution_user_period', 'user_id', 'period_start'),
        Index('idx_evolution_prediction_error', 'prediction_error'),
    )


class DecisionHistory(FullAuditModel):
    """
    Stores decision-making history for pattern recognition.
    Used to predict future decisions based on past behavior.
    """
    
    __tablename__ = "decision_history"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="decision_history")
    
    # Decision context
    context_hash = Column(String(64), nullable=False, comment="Hash of decision context")
    context_data = Column(JSONB, nullable=False, comment="Full context information")
    
    # Decision made
    choice = Column(String(500), nullable=False, comment="The choice that was made")
    alternatives = Column(JSONB, comment="Other options that were available")
    
    # Outcome
    outcome_score = Column(Float, comment="How good the outcome was (-1 to 1)")
    outcome_description = Column(Text, comment="What happened as a result")
    
    # Confidence and prediction
    predicted_choice = Column(String(500), comment="What we predicted they would choose")
    prediction_confidence = Column(Float, comment="How confident we were")
    prediction_correct = Column(Boolean, comment="Whether prediction was correct")
    
    # Decision factors
    decision_factors = Column(
        JSONB,
        default=dict,
        comment="Factors that influenced the decision"
    )
    
    # Time taken
    decision_time = Column(Float, comment="Seconds taken to decide")
    
    # Indexes
    __table_args__ = (
        Index('idx_decision_user_context', 'user_id', 'context_hash'),
        Index('idx_decision_outcome', 'outcome_score'),
        Index('idx_decision_prediction', 'prediction_correct'),
    )


class MirrorCalibration(FullAuditModel):
    """
    Calibration data for improving mirror accuracy.
    Stores feedback and adjustments to improve cognitive twin.
    """
    
    __tablename__ = "mirror_calibrations"
    
    # User relationship
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="mirror_calibrations")
    
    # Calibration type
    calibration_type = Column(
        String(50),
        nullable=False,
        comment="personality, decision, linguistic, emotional"
    )
    
    # Before/After metrics
    accuracy_before = Column(Float, nullable=False)
    accuracy_after = Column(Float, nullable=False)
    improvement = Column(Float, nullable=False)
    
    # Calibration data
    calibration_data = Column(
        JSONB,
        nullable=False,
        comment="Specific adjustments made"
    )
    
    # User feedback
    user_feedback = Column(Text, comment="User's feedback on accuracy")
    user_rating = Column(Float, comment="User's rating (1-5)")
    
    # Method used
    calibration_method = Column(
        String(100),
        comment="Algorithm or method used for calibration"
    )
    
    # Success metrics
    messages_analyzed = Column(Integer, default=0)
    patterns_identified = Column(Integer, default=0)
    
    # Indexes
    __table_args__ = (
        Index('idx_calibration_user_type', 'user_id', 'calibration_type'),
        Index('idx_calibration_improvement', 'improvement'),
    )