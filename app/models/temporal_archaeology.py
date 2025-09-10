"""
Temporal Archaeology Database Models

Stores linguistic fingerprints, reconstructed messages, and behavioral patterns
for conversation archaeology and recovery.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text,
    ForeignKey, JSON, Index, DateTime, ARRAY, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

from app.database.base import Base, FullAuditModel


class LinguisticFingerprint(FullAuditModel):
    """
    Stores unique linguistic patterns that identify a user's writing style.
    Used for message reconstruction and behavioral authentication.
    """
    
    __tablename__ = "linguistic_fingerprints"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="linguistic_fingerprints")
    
    # N-gram distributions
    ngram_distribution = Column(
        JSONB,
        nullable=False,
        comment="Unigram, bigram, trigram frequency distributions"
    )
    
    # Lexical diversity metrics
    lexical_diversity = Column(
        JSONB,
        nullable=False,
        comment="TTR, Yule's K, Simpson's D, hapax legomena counts"
    )
    
    # Stylistic features
    stylistic_features = Column(
        JSONB,
        nullable=False,
        comment="Punctuation, capitalization, sentence structure patterns"
    )
    
    # Temporal signatures
    temporal_signatures = Column(
        JSONB,
        nullable=False,
        comment="Response times, active hours, burst patterns"
    )
    
    # Emotional patterns
    emotional_patterns = Column(
        JSONB,
        nullable=False,
        comment="Emotional expression tendencies and markers"
    )
    
    # Vocabulary metrics
    vocabulary_size = Column(Integer, default=0, nullable=False)
    avg_message_length = Column(Float, default=0.0, nullable=False)
    unique_phrases = Column(
        ARRAY(String),
        default=list,
        comment="Frequently used unique phrases"
    )
    
    # Fingerprint quality
    confidence_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Confidence in fingerprint accuracy (0-1)"
    )
    message_sample_size = Column(Integer, default=0, nullable=False)
    time_span_days = Column(Integer, default=0, nullable=False)
    
    # Character-level patterns
    character_patterns = Column(
        JSONB,
        comment="Double letters, uppercase ratio, special chars"
    )
    
    # Update tracking
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    update_count = Column(Integer, default=0, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_fingerprint_user_id', 'user_id'),
        Index('idx_fingerprint_confidence', 'confidence_score'),
        Index('idx_fingerprint_updated', 'last_updated'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
        CheckConstraint('vocabulary_size >= 0', name='check_vocabulary_positive'),
    )


class TemporalPattern(FullAuditModel):
    """
    Discovered patterns in user's temporal communication behavior.
    Used for predicting message timing and conversation flow.
    """
    
    __tablename__ = "temporal_patterns"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="temporal_patterns")
    
    # Pattern identification
    pattern_type = Column(
        String(50),
        nullable=False,
        comment="peak_hours, weekly_rhythm, response_timing, etc."
    )
    pattern_signature = Column(
        JSONB,
        nullable=False,
        comment="Detailed pattern data and parameters"
    )
    
    # Pattern metrics
    frequency = Column(Integer, default=0, nullable=False, comment="Occurrence count")
    confidence = Column(Float, default=0.0, nullable=False, comment="Pattern reliability")
    predictive_power = Column(Float, comment="Success rate in predictions")
    
    # Temporal scope
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_from = Column(DateTime)
    valid_until = Column(DateTime)
    
    # Evolution tracking
    previous_version_id = Column(PG_UUID(as_uuid=True), ForeignKey("temporal_patterns.id"))
    evolution_delta = Column(JSONB, comment="Changes from previous version")
    
    # Usage statistics
    usage_count = Column(Integer, default=0, nullable=False)
    last_used = Column(DateTime)
    success_rate = Column(Float, comment="Accuracy when applied")
    
    # Indexes
    __table_args__ = (
        Index('idx_pattern_user_type', 'user_id', 'pattern_type'),
        Index('idx_pattern_confidence', 'confidence'),
        Index('idx_pattern_discovered', 'discovered_at'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_pattern_confidence'),
    )


class ReconstructedMessage(FullAuditModel):
    """
    Messages reconstructed from behavioral patterns and context.
    Represents probable content of lost or deleted conversations.
    """
    
    __tablename__ = "reconstructed_messages"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="reconstructed_messages")
    
    # Reconstructed content
    content = Column(Text, nullable=False, comment="Reconstructed message text")
    estimated_timestamp = Column(DateTime, nullable=False)
    
    # Reconstruction metadata
    confidence_score = Column(
        Float,
        nullable=False,
        comment="Confidence in reconstruction accuracy (0-1)"
    )
    reconstruction_method = Column(
        String(100),
        nullable=False,
        comment="Method used for reconstruction"
    )
    
    # Context information
    context_before = Column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        comment="Message IDs before gap"
    )
    context_after = Column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        comment="Message IDs after gap"
    )
    
    # Quality metrics
    linguistic_match_score = Column(
        Float,
        comment="How well it matches user's style (0-1)"
    )
    contextual_coherence = Column(
        Float,
        comment="How well it fits conversation flow (0-1)"
    )
    
    # Evidence and validation
    evidence_markers = Column(
        ARRAY(String),
        default=list,
        comment="Supporting evidence for reconstruction"
    )
    validation_status = Column(
        String(50),
        default="unvalidated",
        comment="unvalidated, confirmed, rejected, partial"
    )
    
    # Associated patterns
    pattern_ids = Column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        comment="Temporal patterns used in reconstruction"
    )
    
    # Ghost conversation link
    ghost_conversation_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ghost_conversations.id")
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_reconstructed_user_time', 'user_id', 'estimated_timestamp'),
        Index('idx_reconstructed_confidence', 'confidence_score'),
        Index('idx_reconstructed_validation', 'validation_status'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_reconstruction_confidence'),
    )


class GhostConversation(FullAuditModel):
    """
    Complete reconstructed conversations from temporal gaps.
    Represents entire conversation threads recovered through archaeology.
    """
    
    __tablename__ = "ghost_conversations"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="ghost_conversations")
    
    # Time range
    time_range_start = Column(DateTime, nullable=False)
    time_range_end = Column(DateTime, nullable=False)
    gap_duration_seconds = Column(Integer, nullable=False)
    
    # Reconstructed content
    reconstructed_messages = Column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        comment="IDs of reconstructed messages"
    )
    message_count = Column(Integer, default=0, nullable=False)
    
    # Quality metrics
    confidence_score = Column(
        Float,
        nullable=False,
        comment="Overall confidence in reconstruction (0-1)"
    )
    completeness_score = Column(
        Float,
        comment="Estimated completeness of recovery (0-1)"
    )
    
    # Reconstruction details
    reconstruction_method = Column(
        String(100),
        nullable=False,
        comment="Primary method used"
    )
    reconstruction_duration = Column(
        Float,
        comment="Time taken to reconstruct (seconds)"
    )
    
    # Fingerprint used
    fingerprint_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("linguistic_fingerprints.id")
    )
    fingerprint = relationship("LinguisticFingerprint")
    
    # Evidence and validation
    evidence_summary = Column(
        JSONB,
        comment="Summary of reconstruction evidence"
    )
    topics_identified = Column(
        ARRAY(String),
        default=list,
        comment="Topics discussed in ghost conversation"
    )
    
    # User feedback
    user_validation = Column(
        String(50),
        comment="User's assessment: accurate, partial, incorrect"
    )
    user_feedback = Column(Text, comment="User's feedback on reconstruction")
    
    # Relationships
    reconstructed_messages_rel = relationship(
        "ReconstructedMessage",
        back_populates="ghost_conversation",
        foreign_keys="ReconstructedMessage.ghost_conversation_id"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_ghost_user_time', 'user_id', 'time_range_start'),
        Index('idx_ghost_confidence', 'confidence_score'),
        Index('idx_ghost_validation', 'user_validation'),
        CheckConstraint('time_range_end > time_range_start', name='check_time_range_valid'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_ghost_confidence'),
    )


class ConversationFragment(FullAuditModel):
    """
    Partial conversation pieces used in reconstruction.
    Stores fragments of context used to rebuild lost messages.
    """
    
    __tablename__ = "conversation_fragments"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="conversation_fragments")
    
    # Fragment content
    fragment_type = Column(
        String(50),
        nullable=False,
        comment="opening, closing, transition, response, etc."
    )
    content_pattern = Column(
        Text,
        nullable=False,
        comment="Pattern or template of fragment"
    )
    
    # Usage statistics
    occurrence_count = Column(Integer, default=1, nullable=False)
    contexts = Column(
        JSONB,
        default=list,
        comment="Contexts where fragment appears"
    )
    
    # Temporal data
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Quality metrics
    reliability_score = Column(
        Float,
        default=0.5,
        comment="How reliably this fragment appears (0-1)"
    )
    
    # Associated patterns
    associated_patterns = Column(
        JSONB,
        default=dict,
        comment="Related temporal and linguistic patterns"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_fragment_user_type', 'user_id', 'fragment_type'),
        Index('idx_fragment_occurrence', 'occurrence_count'),
        Index('idx_fragment_reliability', 'reliability_score'),
    )


class ArchaeologySession(FullAuditModel):
    """
    Tracks archaeology sessions for exploring lost conversations.
    Manages the discovery and reconstruction process.
    """
    
    __tablename__ = "archaeology_sessions"
    
    # User relationship
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="archaeology_sessions")
    
    # Session metadata
    session_type = Column(
        String(50),
        default="manual",
        comment="manual, automated, scheduled"
    )
    session_status = Column(
        String(50),
        default="active",
        comment="active, completed, failed, cancelled"
    )
    
    # Discovery results
    patterns_discovered = Column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        comment="IDs of discovered patterns"
    )
    gaps_identified = Column(Integer, default=0, nullable=False)
    messages_reconstructed = Column(Integer, default=0, nullable=False)
    
    # Performance metrics
    total_messages_analyzed = Column(Integer, default=0, nullable=False)
    processing_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Time range analyzed
    time_range_analyzed = Column(
        JSONB,
        comment="Start and end times of analysis period"
    )
    
    # Quality metrics
    overall_confidence = Column(
        Float,
        comment="Average confidence across reconstructions (0-1)"
    )
    discoveries_validated = Column(Integer, default=0)
    discoveries_rejected = Column(Integer, default=0)
    
    # Session configuration
    configuration = Column(
        JSONB,
        default=dict,
        comment="Session parameters and thresholds"
    )
    
    # Results summary
    results_summary = Column(
        JSONB,
        comment="Summary of session discoveries and reconstructions"
    )
    
    # Completion data
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_archaeology_user_status', 'user_id', 'session_status'),
        Index('idx_archaeology_created', 'created_at'),
        Index('idx_archaeology_confidence', 'overall_confidence'),
    )