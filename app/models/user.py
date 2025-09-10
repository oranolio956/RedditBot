"""
User Model

Defines the User model for storing Telegram user information
and bot interaction history.
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Integer, BigInteger, Boolean, Text, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from app.database.base import FullAuditModel


class User(FullAuditModel):
    """
    User model for storing Telegram user information.
    
    Stores both Telegram-specific data and application-specific
    user preferences and interaction history.
    """
    
    __tablename__ = "users"
    
    # Telegram-specific fields
    telegram_id = Column(
        BigInteger,
        unique=True,
        nullable=False,
        index=True,
        comment="Telegram user ID"
    )
    
    username = Column(
        String(32),
        nullable=True,
        index=True,
        comment="Telegram username (without @)"
    )
    
    first_name = Column(
        String(64),
        nullable=True,
        comment="User's first name from Telegram"
    )
    
    last_name = Column(
        String(64),
        nullable=True,
        comment="User's last name from Telegram"
    )
    
    language_code = Column(
        String(10),
        nullable=True,
        default="en",
        comment="User's language code (ISO 639-1)"
    )
    
    # Bot interaction fields
    is_bot = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this user is a bot"
    )
    
    is_premium = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether user has Telegram Premium"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether user is active in the bot"
    )
    
    is_blocked = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether user is blocked by the bot"
    )
    
    # Interaction tracking
    first_interaction = Column(
        String(50),
        nullable=True,
        comment="First command/message type from user"
    )
    
    last_activity = Column(
        String(50),
        nullable=True,
        comment="Last activity type"
    )
    
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of messages from this user"
    )
    
    command_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of commands from this user"
    )
    
    # User preferences stored as JSONB for flexibility
    preferences = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="User preferences and settings"
    )
    
    # ML-related fields
    personality_profile = Column(
        JSONB,
        nullable=True,
        comment="ML-generated personality profile"
    )
    
    interaction_history = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Recent interaction patterns for ML"
    )
    
    # Relationships
    conversation_sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    group_memberships = relationship("GroupMember", back_populates="user", cascade="all, delete-orphan")
    
    # Database indexes for performance
    # Revolutionary Features Relationships
    cognitive_profile = relationship("CognitiveProfile", back_populates="user", uselist=False)
    keystroke_patterns = relationship("KeystrokePattern", back_populates="user")
    consciousness_sessions = relationship("ConsciousnessSession", back_populates="user")
    decision_history = relationship("DecisionHistory", back_populates="user")
    personality_evolution = relationship("PersonalityEvolution", back_populates="user")
    mirror_calibrations = relationship("MirrorCalibration", back_populates="user")
    
    # Memory Palace relationships
    memory_palaces = relationship("MemoryPalace", back_populates="user")
    
    # Temporal Archaeology relationships
    conversation_fragments = relationship("ConversationFragment", back_populates="user")
    reconstructed_messages = relationship("ReconstructedMessage", back_populates="user")
    temporal_patterns = relationship("TemporalPattern", back_populates="user")
    linguistic_fingerprint = relationship("LinguisticFingerprint", back_populates="user", uselist=False)
    ghost_conversations = relationship("GhostConversation", back_populates="user")
    archaeology_sessions = relationship("ArchaeologySession", back_populates="user")
    
    # Emotional Intelligence relationships
    emotional_profile = relationship("EmotionalProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_telegram_id', 'telegram_id'),
        Index('idx_user_username', 'username'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_blocked', 'is_blocked'),
        Index('idx_user_activity', 'updated_at'),
        Index('idx_user_language', 'language_code'),
    )
    
    def get_display_name(self) -> str:
        """
        Get user's display name for UI purposes.
        
        Returns the best available name in order of preference:
        1. Full name (first + last)
        2. First name only
        3. Username
        4. Telegram ID as fallback
        """
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.username:
            return f"@{self.username}"
        else:
            return f"User {self.telegram_id}"
    
    def get_mention(self) -> str:
        """
        Get user mention string for Telegram.
        
        Returns either username mention or first name for
        use in Telegram message replies.
        """
        if self.username:
            return f"@{self.username}"
        elif self.first_name:
            return self.first_name
        else:
            return f"User {self.telegram_id}"
    
    def update_activity(self, activity_type: str) -> None:
        """
        Update user's last activity tracking.
        
        Args:
            activity_type: Type of activity (e.g., 'message', 'command', 'callback')
        """
        self.last_activity = activity_type
        self.updated_at = datetime.utcnow()
        
        if activity_type == "command":
            self.command_count += 1
        else:
            self.message_count += 1
    
    def get_preference(self, key: str, default=None):
        """
        Get user preference value.
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        if not self.preferences:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value) -> None:
        """
        Set user preference value.
        
        Args:
            key: Preference key
            value: Preference value
        """
        if not self.preferences:
            self.preferences = {}
        
        self.preferences[key] = value
        # Mark the field as modified for SQLAlchemy
        from sqlalchemy.orm import attributes
        attributes.flag_modified(self, 'preferences')
    
    def is_new_user(self) -> bool:
        """Check if this is a new user (first interaction today)."""
        if not self.created_at:
            return True
        
        time_diff = datetime.utcnow() - self.created_at
        return time_diff.total_seconds() < 86400  # 24 hours
    
    def get_interaction_stats(self) -> dict:
        """
        Get user interaction statistics.
        
        Returns:
            Dictionary with interaction statistics
        """
        return {
            "total_messages": self.message_count,
            "total_commands": self.command_count,
            "last_activity": self.last_activity,
            "days_active": (datetime.utcnow() - self.created_at).days if self.created_at else 0,
            "is_new_user": self.is_new_user(),
        }
    
    def __str__(self) -> str:
        """String representation of user."""
        return f"User({self.get_display_name()}, telegram_id={self.telegram_id})"