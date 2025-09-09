"""
Personality Models

Defines models for AI personality profiling, adaptation, and user interaction modeling.
Supports dynamic personality adaptation based on user behavior and preferences.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, ForeignKey,
    Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.sql import func

from app.database.base import FullAuditModel, BaseModel


class PersonalityDimension(str, Enum):
    """Personality dimension enumeration (Big Five + additional)."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"  
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    HUMOR = "humor"
    EMPATHY = "empathy"
    FORMALITY = "formality"
    DIRECTNESS = "directness"
    ENTHUSIASM = "enthusiasm"


class AdaptationStrategy(str, Enum):
    """Personality adaptation strategy enumeration."""
    MIRROR = "mirror"          # Mirror user's personality
    COMPLEMENT = "complement"   # Complement user's personality
    BALANCE = "balance"        # Balance between mirror and complement
    STATIC = "static"          # No adaptation
    CUSTOM = "custom"          # Custom adaptation rules


class PersonalityTrait(BaseModel):
    """
    Personality trait definitions and scoring templates.
    
    Defines individual personality traits that can be measured
    and adapted by the AI system.
    """
    
    __tablename__ = "personality_traits"
    
    # Trait identification
    name = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique trait name"
    )
    
    dimension = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Personality dimension this trait belongs to"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of the trait"
    )
    
    # Scoring configuration
    min_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Minimum possible score for this trait"
    )
    
    max_score = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Maximum possible score for this trait"
    )
    
    default_score = Column(
        Float,
        default=0.5,
        nullable=False,
        comment="Default/neutral score for this trait"
    )
    
    # Measurement configuration
    measurement_indicators = Column(
        JSONB,
        nullable=True,
        comment="Indicators used to measure this trait from user behavior"
    )
    
    adaptation_rules = Column(
        JSONB,
        nullable=True,
        comment="Rules for adapting bot behavior based on this trait"
    )
    
    # Trait metadata
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether this trait is actively measured"
    )
    
    weight = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Weight/importance of this trait in overall personality"
    )
    
    # Database constraints
    __table_args__ = (
        Index('idx_trait_dimension', 'dimension'),
        Index('idx_trait_active', 'is_active'),
        CheckConstraint('min_score <= max_score', name='ck_trait_score_range'),
        CheckConstraint('default_score >= min_score AND default_score <= max_score', name='ck_trait_default_range'),
        CheckConstraint('weight >= 0', name='ck_trait_weight_positive'),
    )
    
    def normalize_score(self, raw_score: float) -> float:
        """Normalize a raw score to the trait's range."""
        if raw_score < self.min_score:
            return self.min_score
        elif raw_score > self.max_score:
            return self.max_score
        return raw_score
    
    def get_trait_config(self) -> Dict[str, Any]:
        """Get trait configuration for ML pipeline."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "score_range": [self.min_score, self.max_score],
            "default_score": self.default_score,
            "weight": self.weight,
            "indicators": self.measurement_indicators or {},
            "adaptation_rules": self.adaptation_rules or {},
        }


class PersonalityProfile(FullAuditModel):
    """
    Personality profile templates and configurations.
    
    Defines different personality profiles that the AI can adopt,
    with specific trait combinations and behavioral patterns.
    """
    
    __tablename__ = "personality_profiles"
    
    # Profile identification
    name = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique profile name"
    )
    
    display_name = Column(
        String(100),
        nullable=False,
        comment="Human-readable profile name"
    )
    
    description = Column(
        Text,
        nullable=False,
        comment="Detailed description of this personality profile"
    )
    
    category = Column(
        String(50),
        nullable=True,
        index=True,
        comment="Profile category (e.g., professional, casual, supportive)"
    )
    
    # Profile configuration
    trait_scores = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Default trait scores for this profile"
    )
    
    behavioral_patterns = Column(
        JSONB,
        nullable=True,
        comment="Specific behavioral patterns and responses"
    )
    
    communication_style = Column(
        JSONB,
        nullable=True,
        comment="Communication style preferences and rules"
    )
    
    # Adaptation settings
    adaptation_strategy = Column(
        String(20),
        default=AdaptationStrategy.BALANCE,
        nullable=False,
        comment="How this profile adapts to users"
    )
    
    adaptation_sensitivity = Column(
        Float,
        default=0.5,
        nullable=False,
        comment="How quickly this profile adapts (0-1)"
    )
    
    adaptation_limits = Column(
        JSONB,
        nullable=True,
        comment="Limits on how much traits can be adapted"
    )
    
    # Usage and performance
    usage_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times this profile has been used"
    )
    
    average_satisfaction_score = Column(
        Float,
        nullable=True,
        comment="Average user satisfaction with this profile"
    )
    
    performance_metrics = Column(
        JSONB,
        nullable=True,
        comment="Detailed performance metrics for this profile"
    )
    
    # Status and availability
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this profile is available for use"
    )
    
    is_default = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is a default profile"
    )
    
    # Relationships
    user_mappings = relationship("UserPersonalityMapping", back_populates="profile")
    
    # Database constraints
    __table_args__ = (
        Index('idx_profile_category', 'category'),
        Index('idx_profile_active_default', 'is_active', 'is_default'),
        CheckConstraint('adaptation_sensitivity >= 0 AND adaptation_sensitivity <= 1', name='ck_adaptation_sensitivity_range'),
        CheckConstraint('usage_count >= 0', name='ck_usage_count_positive'),
        CheckConstraint('average_satisfaction_score >= 0 AND average_satisfaction_score <= 1', name='ck_satisfaction_range'),
    )
    
    def get_trait_score(self, trait_name: str, default: float = 0.5) -> float:
        """Get score for a specific trait."""
        return self.trait_scores.get(trait_name, default) if self.trait_scores else default
    
    def set_trait_score(self, trait_name: str, score: float) -> None:
        """Set score for a specific trait."""
        if not self.trait_scores:
            self.trait_scores = {}
        
        self.trait_scores[trait_name] = score
        # Mark field as modified for SQLAlchemy
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'trait_scores')
    
    def calculate_personality_distance(self, other_scores: Dict[str, float]) -> float:
        """Calculate personality distance from another set of trait scores."""
        if not self.trait_scores:
            return 1.0
        
        total_distance = 0.0
        trait_count = 0
        
        for trait, our_score in self.trait_scores.items():
            their_score = other_scores.get(trait, 0.5)
            total_distance += abs(our_score - their_score) ** 2
            trait_count += 1
        
        if trait_count == 0:
            return 1.0
        
        # Return normalized Euclidean distance
        return (total_distance / trait_count) ** 0.5
    
    def adapt_to_user(self, user_traits: Dict[str, float], strategy: Optional[str] = None) -> Dict[str, float]:
        """
        Adapt personality profile to user traits based on adaptation strategy.
        
        Args:
            user_traits: User's personality trait scores
            strategy: Override adaptation strategy
            
        Returns:
            Adapted trait scores
        """
        if not self.trait_scores:
            return {}
        
        adaptation_strategy = strategy or self.adaptation_strategy
        adapted_scores = self.trait_scores.copy()
        
        for trait, base_score in self.trait_scores.items():
            user_score = user_traits.get(trait, 0.5)
            
            if adaptation_strategy == AdaptationStrategy.MIRROR:
                # Move towards user's score
                adapted_scores[trait] = base_score + (user_score - base_score) * self.adaptation_sensitivity
            
            elif adaptation_strategy == AdaptationStrategy.COMPLEMENT:
                # Move away from user's score
                complement_score = 1.0 - user_score if user_score > 0.5 else 1.0 - user_score
                adapted_scores[trait] = base_score + (complement_score - base_score) * self.adaptation_sensitivity
            
            elif adaptation_strategy == AdaptationStrategy.BALANCE:
                # Balance between mirroring and complementing
                mirror_score = base_score + (user_score - base_score) * self.adaptation_sensitivity * 0.7
                complement_score = 1.0 - user_score if user_score > 0.5 else 1.0 - user_score
                balanced_complement = base_score + (complement_score - base_score) * self.adaptation_sensitivity * 0.3
                adapted_scores[trait] = mirror_score * 0.7 + balanced_complement * 0.3
            
            # Apply adaptation limits if defined
            if self.adaptation_limits and trait in self.adaptation_limits:
                limits = self.adaptation_limits[trait]
                min_limit = limits.get('min', 0.0)
                max_limit = limits.get('max', 1.0)
                adapted_scores[trait] = max(min_limit, min(max_limit, adapted_scores[trait]))
            
            # Ensure scores stay within valid range [0, 1]
            adapted_scores[trait] = max(0.0, min(1.0, adapted_scores[trait]))
        
        return adapted_scores
    
    def update_performance_metrics(self, satisfaction_score: float, interaction_data: Dict[str, Any]) -> None:
        """Update profile performance metrics."""
        self.usage_count += 1
        
        # Update average satisfaction score
        if self.average_satisfaction_score is None:
            self.average_satisfaction_score = satisfaction_score
        else:
            # Exponential moving average
            alpha = 0.1  # Learning rate
            self.average_satisfaction_score = (
                alpha * satisfaction_score + (1 - alpha) * self.average_satisfaction_score
            )
        
        # Update detailed performance metrics
        if not self.performance_metrics:
            self.performance_metrics = {}
        
        metrics = self.performance_metrics
        metrics['last_updated'] = datetime.utcnow().isoformat()
        metrics['total_interactions'] = self.usage_count
        metrics['recent_satisfaction'] = satisfaction_score
        
        # Update interaction type counters
        interaction_type = interaction_data.get('type', 'unknown')
        if 'interaction_types' not in metrics:
            metrics['interaction_types'] = {}
        metrics['interaction_types'][interaction_type] = metrics['interaction_types'].get(interaction_type, 0) + 1
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'performance_metrics')


class UserPersonalityMapping(FullAuditModel):
    """
    User-specific personality mappings and adaptations.
    
    Tracks how personality profiles are adapted for individual users,
    storing learned preferences and behavioral patterns.
    """
    
    __tablename__ = "user_personality_mappings"
    
    # Relationships
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User this mapping applies to"
    )
    
    profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("personality_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Base personality profile"
    )
    
    # User-specific trait measurements
    measured_user_traits = Column(
        JSONB,
        nullable=True,
        comment="Measured personality traits of the user"
    )
    
    adapted_profile_traits = Column(
        JSONB,
        nullable=True,
        comment="Adapted personality traits for this user"
    )
    
    # Learning and adaptation data
    interaction_history_summary = Column(
        JSONB,
        nullable=True,
        comment="Summary of interaction patterns with this user"
    )
    
    adaptation_confidence = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Confidence in personality measurements (0-1)"
    )
    
    learning_iterations = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of learning iterations performed"
    )
    
    # Performance tracking
    satisfaction_scores = Column(
        JSONB,
        nullable=True,
        comment="History of user satisfaction scores"
    )
    
    engagement_metrics = Column(
        JSONB,
        nullable=True,
        comment="User engagement metrics with this personality"
    )
    
    effectiveness_score = Column(
        Float,
        nullable=True,
        comment="Overall effectiveness of this personality mapping"
    )
    
    # Status and usage
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this mapping is currently active"
    )
    
    is_primary = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is the primary personality for this user"
    )
    
    last_used_at = Column(
        "last_used_at",
        nullable=True,
        index=True,
        comment="When this personality was last used"
    )
    
    usage_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times this mapping has been used"
    )
    
    # Relationships
    user = relationship("User", back_populates="personality_mappings")
    profile = relationship("PersonalityProfile", back_populates="user_mappings")
    
    # Database constraints
    __table_args__ = (
        Index('idx_user_personality_active', 'user_id', 'is_active'),
        Index('idx_user_personality_primary', 'user_id', 'is_primary'),
        Index('idx_personality_mapping_usage', 'last_used_at'),
        UniqueConstraint('user_id', 'profile_id', name='uq_user_profile_mapping'),
        CheckConstraint('adaptation_confidence >= 0 AND adaptation_confidence <= 1', name='ck_confidence_range'),
        CheckConstraint('effectiveness_score >= 0 AND effectiveness_score <= 1', name='ck_effectiveness_range'),
        CheckConstraint('learning_iterations >= 0', name='ck_learning_iterations_positive'),
        CheckConstraint('usage_count >= 0', name='ck_usage_count_positive'),
    )
    
    def update_user_traits(self, new_measurements: Dict[str, float], confidence_boost: float = 0.1) -> None:
        """Update measured user traits with new data."""
        if not self.measured_user_traits:
            self.measured_user_traits = {}
        
        # Use exponential moving average to update traits
        alpha = min(0.3, confidence_boost + 0.1)  # Learning rate based on confidence
        
        for trait, new_value in new_measurements.items():
            current_value = self.measured_user_traits.get(trait, 0.5)
            self.measured_user_traits[trait] = alpha * new_value + (1 - alpha) * current_value
        
        # Update confidence
        self.adaptation_confidence = min(1.0, self.adaptation_confidence + confidence_boost)
        self.learning_iterations += 1
        
        # Mark field as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'measured_user_traits')
    
    def get_current_personality(self) -> Dict[str, float]:
        """Get the current adapted personality for this user."""
        if not self.adapted_profile_traits:
            # Generate adapted traits if they don't exist
            if self.measured_user_traits and self.profile:
                self.adapted_profile_traits = self.profile.adapt_to_user(self.measured_user_traits)
                from sqlalchemy.orm import attributes
                attributes.flag_modified(self, 'adapted_profile_traits')
            else:
                return self.profile.trait_scores if self.profile else {}
        
        return self.adapted_profile_traits
    
    def record_interaction(self, satisfaction_score: float, engagement_data: Dict[str, Any]) -> None:
        """Record interaction results for learning."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
        
        # Update satisfaction history
        if not self.satisfaction_scores:
            self.satisfaction_scores = {"scores": [], "average": satisfaction_score}
        
        scores_list = self.satisfaction_scores["scores"]
        scores_list.append({
            "score": satisfaction_score,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent scores (last 50 interactions)
        if len(scores_list) > 50:
            scores_list = scores_list[-50:]
        
        # Update average
        recent_scores = [s["score"] for s in scores_list[-10:]]  # Last 10 scores
        self.satisfaction_scores["average"] = sum(recent_scores) / len(recent_scores)
        self.satisfaction_scores["scores"] = scores_list
        
        # Update engagement metrics
        if not self.engagement_metrics:
            self.engagement_metrics = {}
        
        for metric, value in engagement_data.items():
            if metric not in self.engagement_metrics:
                self.engagement_metrics[metric] = {"values": [], "average": value}
            
            self.engagement_metrics[metric]["values"].append(value)
            # Keep only recent values
            if len(self.engagement_metrics[metric]["values"]) > 20:
                self.engagement_metrics[metric]["values"] = self.engagement_metrics[metric]["values"][-20:]
            
            # Update average
            self.engagement_metrics[metric]["average"] = (
                sum(self.engagement_metrics[metric]["values"]) / 
                len(self.engagement_metrics[metric]["values"])
            )
        
        # Calculate overall effectiveness
        satisfaction_weight = 0.6
        engagement_weight = 0.4
        
        avg_satisfaction = self.satisfaction_scores["average"]
        avg_engagement = sum(
            m["average"] for m in self.engagement_metrics.values()
        ) / len(self.engagement_metrics) if self.engagement_metrics else 0.5
        
        self.effectiveness_score = (
            satisfaction_weight * avg_satisfaction + 
            engagement_weight * avg_engagement
        )
        
        # Mark fields as modified
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'satisfaction_scores')
        attributes.flag_modified(self, 'engagement_metrics')
    
    def should_retrain(self) -> bool:
        """Determine if this mapping should be retrained."""
        if self.learning_iterations < 5:
            return True
        
        if self.effectiveness_score and self.effectiveness_score < 0.6:
            return True
        
        # Retrain periodically based on usage
        if self.usage_count > 0 and self.usage_count % 100 == 0:
            return True
        
        return False
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of this personality mapping."""
        return {
            "mapping_id": str(self.id),
            "profile_name": self.profile.name if self.profile else "Unknown",
            "adaptation_confidence": self.adaptation_confidence,
            "effectiveness_score": self.effectiveness_score,
            "usage_count": self.usage_count,
            "learning_iterations": self.learning_iterations,
            "is_primary": self.is_primary,
            "last_used": self.last_used_at.isoformat() if self.last_used_at else None,
        }


# Update User model to include personality relationships
User.personality_mappings = relationship("UserPersonalityMapping", back_populates="user", cascade="all, delete-orphan")