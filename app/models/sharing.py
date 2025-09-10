"""
Viral Sharing and Referral Models

Database models for viral content sharing, referral tracking,
and growth mechanics to drive organic user acquisition.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, Text, JSON, 
    Float, ForeignKey, Index, UniqueConstraint, DateTime
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.database.base import FullAuditModel


class ShareableContentType(str, Enum):
    """Types of shareable content that can go viral."""
    CONVERSATION_EXCERPT = "conversation_excerpt"
    AI_RESPONSE = "ai_response"
    PERSONALITY_INSIGHT = "personality_insight"
    FUNNY_MOMENT = "funny_moment"
    EDUCATIONAL_SUMMARY = "educational_summary"
    ACHIEVEMENT_CARD = "achievement_card"
    MILESTONE_CELEBRATION = "milestone_celebration"
    WISDOM_QUOTE = "wisdom_quote"


class SocialPlatform(str, Enum):
    """Social platforms where content can be shared."""
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    LINKEDIN = "linkedin"


class ShareableContent(FullAuditModel):
    """
    Viral-optimized shareable content generated from bot interactions.
    
    Stores AI-curated moments that have high viral potential,
    with anonymization and social optimization built-in.
    """
    
    __tablename__ = "shareable_content"
    
    # Content metadata
    content_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of shareable content"
    )
    
    title = Column(
        String(200),
        nullable=False,
        comment="Viral-optimized title for social sharing"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="SEO and social-optimized description"
    )
    
    # Content data
    content_data = Column(
        JSONB,
        nullable=False,
        comment="Structured content data for different formats"
    )
    
    # Visual elements
    image_url = Column(
        String(500),
        nullable=True,
        comment="Generated image/card URL for visual sharing"
    )
    
    video_url = Column(
        String(500),
        nullable=True,
        comment="Generated video URL for TikTok/Instagram"
    )
    
    # Viral optimization
    viral_score = Column(
        Float,
        default=0.0,
        nullable=False,
        index=True,
        comment="AI-calculated viral potential score (0-100)"
    )
    
    hashtags = Column(
        JSON,
        nullable=True,
        comment="Trending hashtags for maximum reach"
    )
    
    optimal_platforms = Column(
        JSON,
        nullable=True,
        comment="Best social platforms for this content type"
    )
    
    # Anonymization and privacy
    is_anonymized = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether content has been anonymized"
    )
    
    anonymization_level = Column(
        String(20),
        default="high",
        nullable=False,
        comment="Level of anonymization applied"
    )
    
    # Source tracking (anonymized)
    source_conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        comment="Source conversation (for analytics only)"
    )
    
    source_user_anonymous_id = Column(
        String(50),
        nullable=True,
        comment="Anonymized user identifier for analytics"
    )
    
    # Engagement tracking
    view_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total views across all platforms"
    )
    
    share_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total shares across all platforms"
    )
    
    like_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total likes/reactions across platforms"
    )
    
    comment_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total comments across platforms"
    )
    
    # Content lifecycle
    is_published = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether content is live and shareable"
    )
    
    published_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When content was first published"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When content expires (for time-sensitive content)"
    )
    
    # AI enhancement
    ai_enhancement_data = Column(
        JSONB,
        nullable=True,
        comment="AI-generated enhancements and optimizations"
    )
    
    # Relationships
    shares = relationship("ContentShare", back_populates="content", cascade="all, delete-orphan")
    source_conversation = relationship("Conversation", backref="generated_content")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_content_viral_score', 'viral_score'),
        Index('idx_content_type_published', 'content_type', 'is_published'),
        Index('idx_content_published_viral', 'is_published', 'viral_score'),
        Index('idx_content_expires', 'expires_at'),
    )
    
    def is_trending(self) -> bool:
        """Check if content is currently trending."""
        if not self.published_at:
            return False
        
        # Consider trending if published within last 24 hours and high engagement
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        is_recent = self.published_at > recent_threshold
        
        # High engagement criteria
        total_engagement = self.view_count + (self.share_count * 10) + (self.like_count * 2)
        is_engaging = total_engagement > 100
        
        return is_recent and is_engaging and self.viral_score > 70
    
    def get_viral_metrics(self) -> Dict[str, Any]:
        """Get comprehensive viral performance metrics."""
        engagement_rate = 0
        if self.view_count > 0:
            engagement_rate = (self.like_count + self.comment_count + self.share_count) / self.view_count
        
        return {
            "viral_score": self.viral_score,
            "total_views": self.view_count,
            "total_shares": self.share_count,
            "total_engagement": self.like_count + self.comment_count,
            "engagement_rate": engagement_rate,
            "is_trending": self.is_trending(),
            "days_live": (datetime.utcnow() - self.published_at).days if self.published_at else 0,
            "shares_per_day": self.share_count / max((datetime.utcnow() - self.published_at).days, 1) if self.published_at else 0
        }


class ContentShare(FullAuditModel):
    """
    Individual shares of viral content across social platforms.
    
    Tracks every share to measure viral coefficient and
    identify most effective content and platforms.
    """
    
    __tablename__ = "content_shares"
    
    # Content reference
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("shareable_content.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to shared content"
    )
    
    # Sharer information (anonymized)
    sharer_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who shared the content"
    )
    
    sharer_anonymous_id = Column(
        String(50),
        nullable=True,
        comment="Anonymized sharer identifier"
    )
    
    # Platform and format
    platform = Column(
        String(20),
        nullable=False,
        index=True,
        comment="Social platform where content was shared"
    )
    
    share_format = Column(
        String(50),
        nullable=True,
        comment="Format used for sharing (image, video, text, etc.)"
    )
    
    # Share metadata
    share_url = Column(
        String(500),
        nullable=True,
        comment="URL of the shared content on the platform"
    )
    
    share_text = Column(
        Text,
        nullable=True,
        comment="Custom text added by sharer"
    )
    
    # Performance tracking
    impressions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times the share was viewed"
    )
    
    clicks = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of clicks on the shared content"
    )
    
    secondary_shares = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of re-shares from this share"
    )
    
    # Attribution tracking
    referral_source = Column(
        String(100),
        nullable=True,
        comment="Source that led to this share"
    )
    
    campaign_id = Column(
        String(100),
        nullable=True,
        comment="Marketing campaign attribution"
    )
    
    # Relationships
    content = relationship("ShareableContent", back_populates="shares")
    sharer = relationship("User", backref="content_shares")
    
    # Indexes
    __table_args__ = (
        Index('idx_share_platform_created', 'platform', 'created_at'),
        Index('idx_share_content_platform', 'content_id', 'platform'),
        Index('idx_share_performance', 'impressions', 'clicks', 'secondary_shares'),
    )


class ReferralProgram(FullAuditModel):
    """
    Referral program configuration and tracking.
    
    Manages different referral campaigns, rewards,
    and growth mechanics for user acquisition.
    """
    
    __tablename__ = "referral_programs"
    
    # Program details
    name = Column(
        String(100),
        nullable=False,
        unique=True,
        comment="Program name"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="Program description"
    )
    
    # Reward configuration
    referrer_reward = Column(
        JSON,
        nullable=False,
        comment="Reward for the person making referral"
    )
    
    referee_reward = Column(
        JSON,
        nullable=False,
        comment="Reward for the person being referred"
    )
    
    # Program settings
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether program is currently active"
    )
    
    max_referrals_per_user = Column(
        Integer,
        nullable=True,
        comment="Maximum referrals per user (null = unlimited)"
    )
    
    valid_from = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Program start date"
    )
    
    valid_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Program end date"
    )
    
    # Requirements
    minimum_activity_level = Column(
        String(20),
        default="basic",
        nullable=False,
        comment="Minimum activity level to participate"
    )
    
    # Tracking
    total_referrals = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total referrals made through this program"
    )
    
    successful_conversions = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Referrals that converted to active users"
    )
    
    # Relationships
    referrals = relationship("UserReferral", back_populates="program", cascade="all, delete-orphan")
    
    def get_conversion_rate(self) -> float:
        """Calculate referral to conversion rate."""
        if self.total_referrals == 0:
            return 0.0
        return self.successful_conversions / self.total_referrals


class UserReferral(FullAuditModel):
    """
    Individual user referrals and their conversion tracking.
    
    Tracks each referral from initial share through conversion
    to measure referral program effectiveness.
    """
    
    __tablename__ = "user_referrals"
    
    # Program reference
    program_id = Column(
        UUID(as_uuid=True),
        ForeignKey("referral_programs.id", ondelete="CASCADE"),
        nullable=False,
        comment="Referral program used"
    )
    
    # Referrer (person making referral)
    referrer_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User making the referral"
    )
    
    # Referee (person being referred)
    referee_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who was referred (once they join)"
    )
    
    referee_contact = Column(
        String(200),
        nullable=True,
        comment="Contact info of referred person (before they join)"
    )
    
    # Referral details
    referral_code = Column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique referral code"
    )
    
    referral_method = Column(
        String(50),
        nullable=False,
        comment="Method used for referral (share, direct, etc.)"
    )
    
    shared_content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("shareable_content.id", ondelete="SET NULL"),
        nullable=True,
        comment="Content that was shared for referral"
    )
    
    # Conversion tracking
    first_click_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When referral link was first clicked"
    )
    
    signup_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When referred user signed up"
    )
    
    first_activity_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When referred user first used the bot"
    )
    
    conversion_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When referral was considered successful"
    )
    
    # Status tracking
    status = Column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
        comment="Referral status (pending, clicked, signed_up, converted)"
    )
    
    # Reward tracking
    referrer_reward_given = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether referrer reward was distributed"
    )
    
    referee_reward_given = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether referee reward was distributed"
    )
    
    # Analytics
    click_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times referral link was clicked"
    )
    
    attribution_data = Column(
        JSONB,
        nullable=True,
        comment="Additional attribution and tracking data"
    )
    
    # Relationships
    program = relationship("ReferralProgram", back_populates="referrals")
    referrer = relationship("User", foreign_keys=[referrer_user_id], backref="sent_referrals")
    referee = relationship("User", foreign_keys=[referee_user_id], backref="received_referrals")
    shared_content = relationship("ShareableContent", backref="referrals")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_referral_code', 'referral_code'),
        Index('idx_referral_status_created', 'status', 'created_at'),
        Index('idx_referral_program_status', 'program_id', 'status'),
        Index('idx_referral_conversion_tracking', 'first_click_at', 'signup_at', 'conversion_at'),
        UniqueConstraint('referral_code', name='unique_referral_code'),
    )
    
    def get_conversion_time(self) -> Optional[timedelta]:
        """Get time from first click to conversion."""
        if not self.first_click_at or not self.conversion_at:
            return None
        return self.conversion_at - self.first_click_at
    
    def is_converted(self) -> bool:
        """Check if referral has converted successfully."""
        return self.status == "converted" and self.conversion_at is not None


class ViralMetrics(FullAuditModel):
    """
    Aggregate viral and growth metrics for analytics.
    
    Stores daily/weekly/monthly rollups of viral performance
    to track growth trends and identify successful patterns.
    """
    
    __tablename__ = "viral_metrics"
    
    # Time period
    date = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Date for these metrics"
    )
    
    period_type = Column(
        String(10),
        nullable=False,
        index=True,
        comment="Type of period (daily, weekly, monthly)"
    )
    
    # Content performance
    total_content_created = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total shareable content created"
    )
    
    total_content_shared = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total pieces of content shared"
    )
    
    total_views = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total content views across platforms"
    )
    
    total_shares = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total shares across all content"
    )
    
    total_engagement = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total likes, comments, reactions"
    )
    
    # Referral performance
    total_referrals_sent = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total referrals sent"
    )
    
    total_referrals_converted = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total referrals that converted"
    )
    
    new_users_from_referrals = Column(
        Integer,
        default=0,
        nullable=False,
        comment="New users acquired through referrals"
    )
    
    # Growth metrics
    viral_coefficient = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Viral coefficient (new users per existing user)"
    )
    
    average_shares_per_user = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Average content shares per active user"
    )
    
    top_performing_content_type = Column(
        String(50),
        nullable=True,
        comment="Best performing content type for this period"
    )
    
    top_performing_platform = Column(
        String(20),
        nullable=True,
        comment="Best performing social platform"
    )
    
    # Additional analytics
    metrics_data = Column(
        JSONB,
        nullable=True,
        comment="Additional detailed metrics and breakdowns"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_date_period', 'date', 'period_type'),
        Index('idx_metrics_viral_coefficient', 'viral_coefficient'),
        UniqueConstraint('date', 'period_type', name='unique_daily_metrics'),
    )
    
    def get_growth_rate(self) -> float:
        """Calculate growth rate for this period."""
        if self.total_referrals_sent == 0:
            return 0.0
        return self.total_referrals_converted / self.total_referrals_sent