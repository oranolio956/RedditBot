"""
Risk Assessment Models

Defines models for conversation risk assessment, safety monitoring,
and content moderation to ensure safe AI interactions.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, ForeignKey,
    Index, CheckConstraint, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    CONTENT_SAFETY = "content_safety"
    PRIVACY_VIOLATION = "privacy_violation"
    HARASSMENT = "harassment"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    MANIPULATION = "manipulation"
    PERSONAL_INFORMATION = "personal_information"
    EMOTIONAL_DISTRESS = "emotional_distress"
    TECHNICAL_EXPLOITATION = "technical_exploitation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class AssessmentStatus(str, Enum):
    """Assessment status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class RiskFactor(BaseModel):
    """
    Risk factor definitions and scoring templates.
    
    Defines individual risk factors that can be detected and assessed
    in conversations and user interactions.
    """
    
    __tablename__ = "risk_factors"
    
    # Factor identification
    name = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique risk factor name"
    )
    
    category = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Risk category this factor belongs to"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of this risk factor"
    )
    
    # Scoring configuration
    base_risk_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Base risk score for this factor (0-1)"
    )
    
    max_risk_score = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Maximum risk score this factor can contribute"
    )
    
    weight = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Weight of this factor in overall risk calculation"
    )
    
    # Detection configuration
    detection_patterns = Column(
        JSONB,
        nullable=True,
        comment="Patterns and rules for detecting this risk factor"
    )
    
    ml_model_config = Column(
        JSONB,
        nullable=True,
        comment="ML model configuration for automated detection"
    )
    
    threshold_config = Column(
        JSONB,
        nullable=True,
        comment="Threshold configurations for different risk levels"
    )
    
    # Response and mitigation
    mitigation_strategies = Column(
        JSONB,
        nullable=True,
        comment="Strategies for mitigating this risk factor"
    )
    
    escalation_rules = Column(
        JSONB,
        nullable=True,
        comment="Rules for when to escalate this risk factor"
    )
    
    # Status and metadata
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this risk factor is actively monitored"
    )
    
    detection_accuracy = Column(
        Float,
        nullable=True,
        comment="Measured accuracy of detection for this factor"
    )
    
    false_positive_rate = Column(
        Float,
        nullable=True,
        comment="False positive rate for this factor"
    )
    
    # Database constraints
    __table_args__ = (
        Index('idx_risk_factor_category', 'category'),
        Index('idx_risk_factor_active', 'is_active'),
        CheckConstraint('base_risk_score >= 0 AND base_risk_score <= 1', name='ck_base_risk_range'),
        CheckConstraint('max_risk_score >= 0 AND max_risk_score <= 1', name='ck_max_risk_range'),
        CheckConstraint('weight >= 0', name='ck_weight_positive'),
        CheckConstraint('detection_accuracy >= 0 AND detection_accuracy <= 1', name='ck_accuracy_range'),
        CheckConstraint('false_positive_rate >= 0 AND false_positive_rate <= 1', name='ck_fpr_range'),
    )
    
    def calculate_risk_score(self, detection_confidence: float, context_factors: Dict[str, Any]) -> float:
        """
        Calculate risk score based on detection confidence and context.
        
        Args:
            detection_confidence: Confidence in detection (0-1)
            context_factors: Additional context factors
            
        Returns:
            Calculated risk score (0-1)
        """
        # Base calculation
        risk_score = self.base_risk_score + (self.max_risk_score - self.base_risk_score) * detection_confidence
        
        # Apply context modifiers if configured
        if self.threshold_config and "context_modifiers" in self.threshold_config:
            modifiers = self.threshold_config["context_modifiers"]
            
            for factor, modifier in modifiers.items():
                if factor in context_factors:
                    risk_score *= modifier.get("multiplier", 1.0)
                    risk_score += modifier.get("additive", 0.0)
        
        # Apply weight
        risk_score *= self.weight
        
        # Ensure within valid range
        return max(0.0, min(1.0, risk_score))
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """Get risk level based on score."""
        if not self.threshold_config:
            # Default thresholds
            if risk_score >= 0.9:
                return RiskLevel.CRITICAL
            elif risk_score >= 0.7:
                return RiskLevel.VERY_HIGH
            elif risk_score >= 0.5:
                return RiskLevel.HIGH
            elif risk_score >= 0.3:
                return RiskLevel.MEDIUM
            elif risk_score >= 0.1:
                return RiskLevel.LOW
            else:
                return RiskLevel.VERY_LOW
        
        # Use configured thresholds
        thresholds = self.threshold_config.get("levels", {})
        if risk_score >= thresholds.get("critical", 0.9):
            return RiskLevel.CRITICAL
        elif risk_score >= thresholds.get("very_high", 0.7):
            return RiskLevel.VERY_HIGH
        elif risk_score >= thresholds.get("high", 0.5):
            return RiskLevel.HIGH
        elif risk_score >= thresholds.get("medium", 0.3):
            return RiskLevel.MEDIUM
        elif risk_score >= thresholds.get("low", 0.1):
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def should_escalate(self, risk_score: float, context: Dict[str, Any]) -> bool:
        """Determine if this risk factor should trigger escalation."""
        if not self.escalation_rules:
            # Default escalation for high-risk scores
            return risk_score >= 0.7
        
        rules = self.escalation_rules
        
        # Check score-based escalation
        if "score_threshold" in rules and risk_score >= rules["score_threshold"]:
            return True
        
        # Check context-based escalation
        if "context_conditions" in rules:
            for condition in rules["context_conditions"]:
                if self._evaluate_condition(condition, context):
                    return True
        
        return False
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single escalation condition."""
        field = condition.get("field")
        operator = condition.get("operator", "eq")
        value = condition.get("value")
        
        if field not in context:
            return False
        
        context_value = context[field]
        
        if operator == "eq":
            return context_value == value
        elif operator == "gt":
            return context_value > value
        elif operator == "gte":
            return context_value >= value
        elif operator == "lt":
            return context_value < value
        elif operator == "lte":
            return context_value <= value
        elif operator == "in":
            return context_value in value
        elif operator == "contains":
            return value in str(context_value)
        
        return False


class RiskAssessment(FullAuditModel):
    """
    Risk assessment records for individual content or interactions.
    
    Stores detailed risk analysis results including detected factors,
    scores, and recommended actions.
    """
    
    __tablename__ = "risk_assessments"
    
    # Assessment metadata
    assessment_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique assessment identifier"
    )
    
    assessment_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of assessment (message, conversation, user, etc.)"
    )
    
    target_type = Column(
        String(50),
        nullable=False,
        comment="Type of target being assessed"
    )
    
    target_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="ID of the target being assessed"
    )
    
    # Assessment results
    overall_risk_score = Column(
        Float,
        nullable=False,
        index=True,
        comment="Overall risk score (0-1)"
    )
    
    overall_risk_level = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Overall risk level classification"
    )
    
    # Detailed risk breakdown
    risk_factors_detected = Column(
        JSONB,
        nullable=True,
        comment="List of detected risk factors with scores"
    )
    
    category_scores = Column(
        JSONB,
        nullable=True,
        comment="Risk scores by category"
    )
    
    confidence_score = Column(
        Float,
        nullable=True,
        comment="Confidence in the assessment (0-1)"
    )
    
    # Assessment context
    assessment_context = Column(
        JSONB,
        nullable=True,
        comment="Context data used in assessment"
    )
    
    content_analyzed = Column(
        Text,
        nullable=True,
        comment="Content that was analyzed"
    )
    
    assessment_metadata = Column(
        JSONB,
        nullable=True,
        comment="Additional assessment metadata"
    )
    
    # Processing information
    processing_time_ms = Column(
        Integer,
        nullable=True,
        comment="Time taken to complete assessment in milliseconds"
    )
    
    models_used = Column(
        ARRAY(String),
        nullable=True,
        comment="ML models used in assessment"
    )
    
    assessment_version = Column(
        String(20),
        nullable=True,
        comment="Version of assessment system used"
    )
    
    # Status and workflow
    status = Column(
        String(20),
        default=AssessmentStatus.COMPLETED,
        nullable=False,
        index=True,
        comment="Assessment status"
    )
    
    requires_escalation = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this assessment requires escalation"
    )
    
    escalated_at = Column(
        "escalated_at",
        nullable=True,
        comment="When escalation was triggered"
    )
    
    escalation_reason = Column(
        Text,
        nullable=True,
        comment="Reason for escalation"
    )
    
    # Actions taken
    actions_recommended = Column(
        JSONB,
        nullable=True,
        comment="Recommended actions based on assessment"
    )
    
    actions_taken = Column(
        JSONB,
        nullable=True,
        comment="Actions actually taken"
    )
    
    # Human review
    reviewed_by_human = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this assessment was reviewed by a human"
    )
    
    human_review_at = Column(
        "human_review_at",
        nullable=True,
        comment="When human review occurred"
    )
    
    human_review_notes = Column(
        Text,
        nullable=True,
        comment="Notes from human review"
    )
    
    review_outcome = Column(
        String(50),
        nullable=True,
        comment="Outcome of human review"
    )
    
    # Database constraints
    __table_args__ = (
        Index('idx_assessment_type_target', 'assessment_type', 'target_type'),
        Index('idx_assessment_risk_level', 'overall_risk_level'),
        Index('idx_assessment_escalation', 'requires_escalation', 'escalated_at'),
        Index('idx_assessment_review', 'reviewed_by_human'),
        CheckConstraint('overall_risk_score >= 0 AND overall_risk_score <= 1', name='ck_risk_score_range'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='ck_confidence_range'),
        CheckConstraint('processing_time_ms >= 0', name='ck_processing_time_positive'),
    )
    
    def add_risk_factor(self, factor_name: str, score: float, confidence: float, evidence: Dict[str, Any]) -> None:
        """Add a detected risk factor to the assessment."""
        if not self.risk_factors_detected:
            self.risk_factors_detected = []
        
        factor_data = {
            "factor_name": factor_name,
            "risk_score": score,
            "confidence": confidence,
            "evidence": evidence,
            "detected_at": datetime.utcnow().isoformat()
        }
        
        self.risk_factors_detected.append(factor_data)
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'risk_factors_detected')
    
    def calculate_overall_score(self) -> None:
        """Calculate overall risk score from individual factors."""
        if not self.risk_factors_detected:
            self.overall_risk_score = 0.0
            self.overall_risk_level = RiskLevel.VERY_LOW
            return
        
        # Weighted average of factor scores
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in self.risk_factors_detected:
            score = factor.get("risk_score", 0.0)
            confidence = factor.get("confidence", 1.0)
            weight = factor.get("weight", 1.0)
            
            weighted_score = score * confidence * weight
            total_weighted_score += weighted_score
            total_weight += weight
        
        if total_weight > 0:
            self.overall_risk_score = total_weighted_score / total_weight
        else:
            self.overall_risk_score = 0.0
        
        # Determine risk level
        if self.overall_risk_score >= 0.9:
            self.overall_risk_level = RiskLevel.CRITICAL
        elif self.overall_risk_score >= 0.7:
            self.overall_risk_level = RiskLevel.VERY_HIGH
        elif self.overall_risk_score >= 0.5:
            self.overall_risk_level = RiskLevel.HIGH
        elif self.overall_risk_score >= 0.3:
            self.overall_risk_level = RiskLevel.MEDIUM
        elif self.overall_risk_score >= 0.1:
            self.overall_risk_level = RiskLevel.LOW
        else:
            self.overall_risk_level = RiskLevel.VERY_LOW
    
    def should_block_content(self) -> bool:
        """Determine if content should be blocked based on risk level."""
        return self.overall_risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]
    
    def should_moderate_response(self) -> bool:
        """Determine if AI response should be moderated."""
        return self.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]
    
    def get_recommended_actions(self) -> List[Dict[str, Any]]:
        """Get recommended actions based on risk assessment."""
        if self.actions_recommended:
            return self.actions_recommended
        
        actions = []
        
        if self.overall_risk_level == RiskLevel.CRITICAL:
            actions.extend([
                {"action": "block_content", "priority": "immediate"},
                {"action": "escalate_to_human", "priority": "immediate"},
                {"action": "log_incident", "priority": "high"}
            ])
        elif self.overall_risk_level == RiskLevel.VERY_HIGH:
            actions.extend([
                {"action": "moderate_response", "priority": "high"},
                {"action": "escalate_to_human", "priority": "high"},
                {"action": "increase_monitoring", "priority": "medium"}
            ])
        elif self.overall_risk_level == RiskLevel.HIGH:
            actions.extend([
                {"action": "moderate_response", "priority": "medium"},
                {"action": "flag_for_review", "priority": "medium"},
                {"action": "increase_monitoring", "priority": "low"}
            ])
        elif self.overall_risk_level == RiskLevel.MEDIUM:
            actions.extend([
                {"action": "log_event", "priority": "low"},
                {"action": "track_patterns", "priority": "low"}
            ])
        
        return actions
    
    def get_assessment_summary(self) -> Dict[str, Any]:
        """Get summary of risk assessment."""
        return {
            "assessment_id": self.assessment_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "overall_risk_score": self.overall_risk_score,
            "overall_risk_level": self.overall_risk_level,
            "confidence_score": self.confidence_score,
            "factors_detected": len(self.risk_factors_detected) if self.risk_factors_detected else 0,
            "requires_escalation": self.requires_escalation,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ConversationRisk(FullAuditModel):
    """
    Conversation-level risk tracking and monitoring.
    
    Aggregates risk assessments for entire conversations and tracks
    risk patterns over time.
    """
    
    __tablename__ = "conversation_risks"
    
    # Conversation reference
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Conversation being monitored"
    )
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User in the conversation"
    )
    
    # Aggregated risk metrics
    current_risk_level = Column(
        String(20),
        default=RiskLevel.VERY_LOW,
        nullable=False,
        index=True,
        comment="Current risk level for conversation"
    )
    
    peak_risk_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Highest risk score observed"
    )
    
    average_risk_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Average risk score across conversation"
    )
    
    # Risk pattern tracking
    risk_trend = Column(
        String(20),
        nullable=True,
        comment="Risk trend (increasing, decreasing, stable)"
    )
    
    risk_history = Column(
        JSONB,
        nullable=True,
        comment="Historical risk scores and timestamps"
    )
    
    escalation_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of escalations for this conversation"
    )
    
    # Monitoring status
    monitoring_level = Column(
        String(20),
        default="normal",
        nullable=False,
        comment="Monitoring intensity level"
    )
    
    last_assessment_at = Column(
        "last_assessment_at",
        nullable=True,
        index=True,
        comment="When last risk assessment was performed"
    )
    
    next_assessment_at = Column(
        "next_assessment_at",
        nullable=True,
        index=True,
        comment="When next assessment should be performed"
    )
    
    # Mitigation and actions
    mitigations_applied = Column(
        JSONB,
        nullable=True,
        comment="Risk mitigation strategies applied"
    )
    
    restrictions_active = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether conversation has active restrictions"
    )
    
    # Relationships
    conversation = relationship("Conversation", back_populates="risk_assessments")
    user = relationship("User", back_populates="conversation_risks")
    
    # Database constraints
    __table_args__ = (
        Index('idx_conversation_risk_level', 'current_risk_level'),
        Index('idx_conversation_risk_monitoring', 'monitoring_level', 'next_assessment_at'),
        Index('idx_conversation_risk_user', 'user_id', 'current_risk_level'),
        CheckConstraint('peak_risk_score >= 0 AND peak_risk_score <= 1', name='ck_peak_risk_range'),
        CheckConstraint('average_risk_score >= 0 AND average_risk_score <= 1', name='ck_avg_risk_range'),
        CheckConstraint('escalation_count >= 0', name='ck_escalation_count_positive'),
    )
    
    def update_risk_assessment(self, new_assessment: RiskAssessment) -> None:
        """Update conversation risk based on new assessment."""
        # Update current risk level
        if new_assessment.overall_risk_score > self.peak_risk_score:
            self.peak_risk_score = new_assessment.overall_risk_score
            self.current_risk_level = new_assessment.overall_risk_level
        
        # Update risk history
        if not self.risk_history:
            self.risk_history = {"assessments": []}
        
        assessment_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "risk_score": new_assessment.overall_risk_score,
            "risk_level": new_assessment.overall_risk_level,
            "assessment_id": new_assessment.assessment_id
        }
        
        self.risk_history["assessments"].append(assessment_data)
        
        # Keep only recent history (last 50 assessments)
        if len(self.risk_history["assessments"]) > 50:
            self.risk_history["assessments"] = self.risk_history["assessments"][-50:]
        
        # Recalculate average
        recent_scores = [a["risk_score"] for a in self.risk_history["assessments"][-10:]]
        self.average_risk_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        
        # Update assessment timing
        self.last_assessment_at = datetime.utcnow()
        self.next_assessment_at = self._calculate_next_assessment_time()
        
        # Check for escalation
        if new_assessment.requires_escalation:
            self.escalation_count += 1
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'risk_history')
    
    def _calculate_next_assessment_time(self) -> datetime:
        """Calculate when next assessment should be performed."""
        base_interval = timedelta(minutes=30)  # Default 30-minute interval
        
        # Adjust interval based on risk level
        if self.current_risk_level in [RiskLevel.CRITICAL, RiskLevel.VERY_HIGH]:
            interval = timedelta(minutes=5)  # Very frequent monitoring
        elif self.current_risk_level == RiskLevel.HIGH:
            interval = timedelta(minutes=15)
        elif self.current_risk_level == RiskLevel.MEDIUM:
            interval = timedelta(minutes=30)
        else:
            interval = timedelta(hours=2)  # Less frequent for low risk
        
        return datetime.utcnow() + interval
    
    def should_restrict_conversation(self) -> bool:
        """Determine if conversation should have restrictions."""
        return (
            self.current_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL] or
            self.escalation_count >= 3 or
            self.average_risk_score > 0.6
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for conversation."""
        return {
            "conversation_id": str(self.conversation_id),
            "current_risk_level": self.current_risk_level,
            "peak_risk_score": self.peak_risk_score,
            "average_risk_score": self.average_risk_score,
            "escalation_count": self.escalation_count,
            "monitoring_level": self.monitoring_level,
            "restrictions_active": self.restrictions_active,
            "last_assessment": self.last_assessment_at.isoformat() if self.last_assessment_at else None,
        }


# Update User and Conversation models to include risk relationships
# User model relationships are defined in the User model itself
# to avoid circular import issues
# Conversation model relationships are defined in the Conversation model itself
# to avoid circular import issues