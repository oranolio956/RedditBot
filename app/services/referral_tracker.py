"""
Referral Tracker Service

Growth mechanics and gamification system for viral user acquisition.
Implements referral rewards, leaderboards, social proof, and premium unlocks
to drive organic growth through user advocacy.
"""

import uuid
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
import structlog

from app.models.sharing import (
    ReferralProgram, UserReferral, ShareableContent, ContentShare,
    ViralMetrics, SocialPlatform
)
from app.models.user import User
from app.models.conversation import Conversation

logger = structlog.get_logger(__name__)


class RewardType(str, Enum):
    """Types of referral rewards."""
    CREDITS = "credits"
    PREMIUM_TIME = "premium_time"
    FEATURE_UNLOCK = "feature_unlock"
    BADGE = "badge"
    LEADERBOARD_POINTS = "leaderboard_points"


class ActivityLevel(str, Enum):
    """User activity levels for referral eligibility."""
    BASIC = "basic"
    ACTIVE = "active"
    ENGAGED = "engaged"
    POWER_USER = "power_user"


@dataclass
class ReferralReward:
    """Referral reward configuration."""
    type: RewardType
    amount: int
    description: str
    conditions: Dict[str, Any]


@dataclass
class LeaderboardEntry:
    """Leaderboard entry with user stats."""
    user_id: str
    display_name: str
    referral_count: int
    conversion_count: int
    total_points: int
    badges: List[str]
    rank: int


class ReferralTracker:
    """
    Comprehensive referral tracking and gamification system.
    
    Manages referral programs, tracks conversions, distributes rewards,
    maintains leaderboards, and provides social proof to drive growth.
    """
    
    def __init__(self, db: Session):
        self.db = db
        
        # Referral reward configurations
        self.reward_configs = self._setup_reward_configs()
        
        # Achievement badges
        self.achievement_badges = self._setup_achievement_badges()
        
        # Leaderboard cache
        self._leaderboard_cache = {}
        self._leaderboard_updated_at = None
    
    def _setup_reward_configs(self) -> Dict[str, ReferralReward]:
        """Setup referral reward configurations."""
        return {
            "standard_referral": ReferralReward(
                type=RewardType.CREDITS,
                amount=25,
                description="$25 in bot credits for successful referral",
                conditions={"min_activity_days": 7}
            ),
            "premium_referral": ReferralReward(
                type=RewardType.PREMIUM_TIME,
                amount=30,  # 30 days
                description="1 month premium features for referral",
                conditions={"referred_user_premium": True}
            ),
            "milestone_referral": ReferralReward(
                type=RewardType.FEATURE_UNLOCK,
                amount=1,
                description="Unlock advanced personality analysis",
                conditions={"referral_milestone": 5}
            ),
            "viral_content_bonus": ReferralReward(
                type=RewardType.LEADERBOARD_POINTS,
                amount=100,
                description="Bonus points for viral content creation",
                conditions={"content_shares": 50}
            )
        }
    
    def _setup_achievement_badges(self) -> Dict[str, Dict[str, Any]]:
        """Setup achievement badge system."""
        return {
            "first_referral": {
                "name": "Connector",
                "description": "Made your first successful referral",
                "icon": "🤝",
                "requirement": {"referrals": 1}
            },
            "super_referrer": {
                "name": "Super Referrer",
                "description": "Successfully referred 10 people",
                "icon": "⭐",
                "requirement": {"referrals": 10}
            },
            "viral_creator": {
                "name": "Viral Creator",
                "description": "Created content shared 100+ times",
                "icon": "🚀",
                "requirement": {"content_shares": 100}
            },
            "community_builder": {
                "name": "Community Builder",
                "description": "Referred 25 active users",
                "icon": "🏗️",
                "requirement": {"active_referrals": 25}
            },
            "influence_master": {
                "name": "Influence Master",
                "description": "Generated 1000+ viral content views",
                "icon": "👑",
                "requirement": {"content_views": 1000}
            }
        }
    
    async def create_referral_program(
        self,
        name: str,
        description: str,
        referrer_reward: Dict[str, Any],
        referee_reward: Dict[str, Any],
        max_referrals: Optional[int] = None,
        valid_until: Optional[datetime] = None
    ) -> ReferralProgram:
        """Create a new referral program."""
        
        program = ReferralProgram(
            name=name,
            description=description,
            referrer_reward=referrer_reward,
            referee_reward=referee_reward,
            max_referrals_per_user=max_referrals,
            valid_until=valid_until,
            is_active=True,
            minimum_activity_level=ActivityLevel.BASIC.value
        )
        
        self.db.add(program)
        self.db.commit()
        
        logger.info(
            "referral_program_created",
            program_id=program.id,
            name=name,
            max_referrals=max_referrals
        )
        
        return program
    
    async def generate_referral_code(
        self,
        user_id: str,
        program_id: Optional[str] = None,
        shared_content_id: Optional[str] = None
    ) -> UserReferral:
        """Generate a unique referral code for a user."""
        
        # Get or create default program
        if not program_id:
            program = await self._get_default_program()
            program_id = program.id
        
        # Generate unique referral code
        code = self._generate_unique_code()
        
        referral = UserReferral(
            program_id=program_id,
            referrer_user_id=user_id,
            referral_code=code,
            referral_method="direct_share",
            shared_content_id=shared_content_id,
            status="pending"
        )
        
        self.db.add(referral)
        self.db.commit()
        
        logger.info(
            "referral_code_generated",
            user_id=user_id,
            referral_code=code,
            program_id=program_id
        )
        
        return referral
    
    def _generate_unique_code(self, length: int = 8) -> str:
        """Generate a unique referral code."""
        while True:
            code = ''.join(random.choices(
                string.ascii_uppercase + string.digits, 
                k=length
            ))
            
            # Ensure uniqueness
            existing = self.db.query(UserReferral).filter(
                UserReferral.referral_code == code
            ).first()
            
            if not existing:
                return code
    
    async def track_referral_click(
        self,
        referral_code: str,
        visitor_data: Dict[str, Any]
    ) -> bool:
        """Track when someone clicks a referral link."""
        
        referral = self.db.query(UserReferral).filter(
            UserReferral.referral_code == referral_code
        ).first()
        
        if not referral:
            logger.warning("referral_code_not_found", code=referral_code)
            return False
        
        # Update click tracking
        referral.click_count += 1
        
        if not referral.first_click_at:
            referral.first_click_at = datetime.utcnow()
            referral.status = "clicked"
        
        # Store attribution data
        if not referral.attribution_data:
            referral.attribution_data = {}
        
        referral.attribution_data.update({
            "clicks": referral.click_count,
            "latest_click": datetime.utcnow().isoformat(),
            "visitor_info": visitor_data
        })
        
        self.db.commit()
        
        logger.info(
            "referral_click_tracked",
            referral_code=referral_code,
            total_clicks=referral.click_count
        )
        
        return True
    
    async def process_referral_signup(
        self,
        referral_code: str,
        new_user: User
    ) -> Optional[UserReferral]:
        """Process when a referred user signs up."""
        
        referral = self.db.query(UserReferral).filter(
            UserReferral.referral_code == referral_code
        ).first()
        
        if not referral:
            return None
        
        # Update referral with new user info
        referral.referee_user_id = new_user.id
        referral.signup_at = datetime.utcnow()
        referral.status = "signed_up"
        
        # Update program statistics
        program = referral.program
        program.total_referrals += 1
        
        self.db.commit()
        
        logger.info(
            "referral_signup_processed",
            referral_code=referral_code,
            new_user_id=new_user.id,
            referrer_user_id=referral.referrer_user_id
        )
        
        # Give referee their welcome reward
        await self._distribute_referee_reward(referral, new_user)
        
        return referral
    
    async def process_referral_conversion(
        self,
        user_id: str,
        activity_threshold: int = 5
    ) -> Optional[UserReferral]:
        """Process referral conversion when user becomes active."""
        
        # Find pending referral for this user
        referral = self.db.query(UserReferral).filter(
            and_(
                UserReferral.referee_user_id == user_id,
                UserReferral.status == "signed_up"
            )
        ).first()
        
        if not referral:
            return None
        
        # Check if user meets activity threshold
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or user.message_count < activity_threshold:
            return None
        
        # Mark as converted
        referral.conversion_at = datetime.utcnow()
        referral.first_activity_at = datetime.utcnow()
        referral.status = "converted"
        
        # Update program conversion stats
        program = referral.program
        program.successful_conversions += 1
        
        self.db.commit()
        
        logger.info(
            "referral_conversion_processed",
            referral_code=referral.referral_code,
            user_id=user_id,
            activity_count=user.message_count
        )
        
        # Distribute referrer rewards
        await self._distribute_referrer_reward(referral)
        
        # Update leaderboards and achievements
        await self._update_user_achievements(referral.referrer_user_id)
        
        return referral
    
    async def _distribute_referee_reward(
        self,
        referral: UserReferral,
        referee_user: User
    ):
        """Distribute welcome reward to the referred user."""
        
        reward = referral.program.referee_reward
        
        if reward.get("type") == RewardType.CREDITS.value:
            await self._add_user_credits(
                referee_user.id,
                reward.get("amount", 10),
                f"Welcome bonus from referral {referral.referral_code}"
            )
        
        elif reward.get("type") == RewardType.PREMIUM_TIME.value:
            await self._add_premium_time(
                referee_user.id,
                reward.get("amount", 7),  # days
                "Referral welcome bonus"
            )
        
        referral.referee_reward_given = True
        self.db.commit()
        
        logger.info(
            "referee_reward_distributed",
            user_id=referee_user.id,
            reward_type=reward.get("type"),
            amount=reward.get("amount")
        )
    
    async def _distribute_referrer_reward(self, referral: UserReferral):
        """Distribute reward to the user who made the referral."""
        
        reward = referral.program.referrer_reward
        
        if reward.get("type") == RewardType.CREDITS.value:
            await self._add_user_credits(
                referral.referrer_user_id,
                reward.get("amount", 25),
                f"Referral bonus for {referral.referral_code}"
            )
        
        elif reward.get("type") == RewardType.PREMIUM_TIME.value:
            await self._add_premium_time(
                referral.referrer_user_id,
                reward.get("amount", 30),
                "Successful referral bonus"
            )
        
        referral.referrer_reward_given = True
        self.db.commit()
        
        logger.info(
            "referrer_reward_distributed",
            user_id=referral.referrer_user_id,
            reward_type=reward.get("type"),
            amount=reward.get("amount")
        )
    
    async def _add_user_credits(
        self,
        user_id: str,
        amount: int,
        description: str
    ):
        """Add credits to user account."""
        
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return
        
        # Get or create user preferences
        if not user.preferences:
            user.preferences = {}
        
        current_credits = user.preferences.get("credits", 0)
        user.preferences["credits"] = current_credits + amount
        
        # Log credit transaction
        credit_history = user.preferences.get("credit_history", [])
        credit_history.append({
            "amount": amount,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "balance_after": current_credits + amount
        })
        user.preferences["credit_history"] = credit_history[-50:]  # Keep last 50
        
        # Mark as modified for SQLAlchemy
        from sqlalchemy.orm import attributes
        attributes.flag_modified(user, 'preferences')
        
        self.db.commit()
    
    async def _add_premium_time(
        self,
        user_id: str,
        days: int,
        description: str
    ):
        """Add premium time to user account."""
        
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return
        
        if not user.preferences:
            user.preferences = {}
        
        # Calculate premium expiry
        current_premium = user.preferences.get("premium_expires_at")
        base_date = datetime.utcnow()
        
        if current_premium:
            try:
                current_expiry = datetime.fromisoformat(current_premium)
                if current_expiry > base_date:
                    base_date = current_expiry
            except:
                pass
        
        new_expiry = base_date + timedelta(days=days)
        user.preferences["premium_expires_at"] = new_expiry.isoformat()
        
        # Log premium transaction
        premium_history = user.preferences.get("premium_history", [])
        premium_history.append({
            "days_added": days,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "expires_at": new_expiry.isoformat()
        })
        user.preferences["premium_history"] = premium_history[-20:]  # Keep last 20
        
        from sqlalchemy.orm import attributes
        attributes.flag_modified(user, 'preferences')
        
        self.db.commit()
    
    async def get_user_referral_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive referral statistics for a user."""
        
        # Get all referrals sent by user
        sent_referrals = self.db.query(UserReferral).filter(
            UserReferral.referrer_user_id == user_id
        ).all()
        
        # Get user's own referral (how they joined)
        received_referral = self.db.query(UserReferral).filter(
            UserReferral.referee_user_id == user_id
        ).first()
        
        # Calculate stats
        total_sent = len(sent_referrals)
        converted_referrals = [r for r in sent_referrals if r.status == "converted"]
        pending_referrals = [r for r in sent_referrals if r.status in ["pending", "clicked", "signed_up"]]
        
        # Get content sharing stats
        user = self.db.query(User).filter(User.id == user_id).first()
        content_shares = self.db.query(ContentShare).filter(
            ContentShare.sharer_user_id == user_id
        ).count() if user else 0
        
        # Calculate total points
        total_points = self._calculate_user_points(user_id, sent_referrals)
        
        # Get achievements
        achievements = await self._get_user_achievements(user_id, sent_referrals)
        
        return {
            "referrals_sent": total_sent,
            "referrals_converted": len(converted_referrals),
            "referrals_pending": len(pending_referrals),
            "conversion_rate": len(converted_referrals) / max(total_sent, 1),
            "content_shares": content_shares,
            "total_points": total_points,
            "achievements": achievements,
            "rank": await self._get_user_rank(user_id),
            "referred_by": received_referral.referrer.get_display_name() if received_referral and received_referral.referrer else None,
            "credits": user.preferences.get("credits", 0) if user and user.preferences else 0,
            "premium_expires_at": user.preferences.get("premium_expires_at") if user and user.preferences else None
        }
    
    def _calculate_user_points(
        self,
        user_id: str,
        referrals: List[UserReferral]
    ) -> int:
        """Calculate total points for a user."""
        
        points = 0
        
        # Points for conversions
        converted = [r for r in referrals if r.status == "converted"]
        points += len(converted) * 100
        
        # Bonus points for milestones
        if len(converted) >= 5:
            points += 500  # Milestone bonus
        
        if len(converted) >= 10:
            points += 1000  # Super referrer bonus
        
        # Points for content sharing
        content_shares = self.db.query(ContentShare).filter(
            ContentShare.sharer_user_id == user_id
        ).count()
        points += content_shares * 10
        
        return points
    
    async def _get_user_achievements(
        self,
        user_id: str,
        referrals: List[UserReferral]
    ) -> List[Dict[str, Any]]:
        """Get user's unlocked achievements."""
        
        achievements = []
        converted_count = len([r for r in referrals if r.status == "converted"])
        
        # Check each achievement
        for badge_id, badge_data in self.achievement_badges.items():
            requirement = badge_data["requirement"]
            unlocked = False
            
            if "referrals" in requirement:
                unlocked = converted_count >= requirement["referrals"]
            
            elif "content_shares" in requirement:
                shares = self.db.query(ContentShare).filter(
                    ContentShare.sharer_user_id == user_id
                ).count()
                unlocked = shares >= requirement["content_shares"]
            
            elif "active_referrals" in requirement:
                # Count active referred users (have recent activity)
                active_count = 0
                for referral in referrals:
                    if referral.referee and referral.referee.last_activity:
                        last_activity = referral.referee.updated_at
                        if (datetime.utcnow() - last_activity).days <= 30:
                            active_count += 1
                
                unlocked = active_count >= requirement["active_referrals"]
            
            elif "content_views" in requirement:
                # Count total views for user's shared content
                total_views = self.db.query(func.sum(ShareableContent.view_count))\
                    .join(ContentShare, ContentShare.content_id == ShareableContent.id)\
                    .filter(ContentShare.sharer_user_id == user_id)\
                    .scalar() or 0
                
                unlocked = total_views >= requirement["content_views"]
            
            if unlocked:
                achievements.append({
                    "id": badge_id,
                    "name": badge_data["name"],
                    "description": badge_data["description"],
                    "icon": badge_data["icon"],
                    "unlocked_at": datetime.utcnow().isoformat()
                })
        
        return achievements
    
    async def _get_user_rank(self, user_id: str) -> int:
        """Get user's rank on the referral leaderboard."""
        
        # Get cached leaderboard
        leaderboard = await self.get_referral_leaderboard()
        
        for i, entry in enumerate(leaderboard):
            if entry.user_id == user_id:
                return i + 1
        
        return len(leaderboard) + 1  # Not on leaderboard
    
    async def get_referral_leaderboard(
        self,
        limit: int = 100,
        period: str = "all_time"
    ) -> List[LeaderboardEntry]:
        """Get referral leaderboard with rankings."""
        
        # Check cache
        cache_key = f"{period}_{limit}"
        if (self._leaderboard_cache.get(cache_key) and
            self._leaderboard_updated_at and
            datetime.utcnow() - self._leaderboard_updated_at < timedelta(minutes=15)):
            return self._leaderboard_cache[cache_key]
        
        # Query referral stats
        query = self.db.query(
            UserReferral.referrer_user_id,
            func.count(UserReferral.id).label('total_referrals'),
            func.count(UserReferral.id).filter(UserReferral.status == 'converted').label('conversions')
        ).group_by(UserReferral.referrer_user_id)
        
        # Apply period filter
        if period == "monthly":
            month_ago = datetime.utcnow() - timedelta(days=30)
            query = query.filter(UserReferral.created_at >= month_ago)
        elif period == "weekly":
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.filter(UserReferral.created_at >= week_ago)
        
        referral_stats = query.all()
        
        # Build leaderboard entries
        entries = []
        for stats in referral_stats:
            user = self.db.query(User).filter(User.id == stats.referrer_user_id).first()
            if not user:
                continue
            
            # Calculate total points
            points = self._calculate_user_points(stats.referrer_user_id, [])
            
            # Get user achievements
            achievements = await self._get_user_achievements(stats.referrer_user_id, [])
            
            entry = LeaderboardEntry(
                user_id=stats.referrer_user_id,
                display_name=user.get_display_name(),
                referral_count=stats.total_referrals,
                conversion_count=stats.conversions,
                total_points=points,
                badges=[a["icon"] for a in achievements],
                rank=0  # Will be set after sorting
            )
            entries.append(entry)
        
        # Sort by total points and conversions
        entries.sort(key=lambda x: (x.total_points, x.conversion_count), reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(entries[:limit]):
            entry.rank = i + 1
        
        # Cache results
        self._leaderboard_cache[cache_key] = entries[:limit]
        self._leaderboard_updated_at = datetime.utcnow()
        
        return entries[:limit]
    
    async def get_social_proof_stats(self) -> Dict[str, Any]:
        """Get social proof statistics for marketing."""
        
        # Total users and referrals
        total_users = self.db.query(User).count()
        total_referrals = self.db.query(UserReferral).count()
        successful_conversions = self.db.query(UserReferral).filter(
            UserReferral.status == "converted"
        ).count()
        
        # Recent activity (last 30 days)
        month_ago = datetime.utcnow() - timedelta(days=30)
        recent_referrals = self.db.query(UserReferral).filter(
            UserReferral.created_at >= month_ago
        ).count()
        
        recent_conversions = self.db.query(UserReferral).filter(
            and_(
                UserReferral.conversion_at >= month_ago,
                UserReferral.status == "converted"
            )
        ).count()
        
        # Content sharing stats
        total_content_shares = self.db.query(ContentShare).count()
        viral_content_count = self.db.query(ShareableContent).filter(
            ShareableContent.viral_score > 70
        ).count()
        
        # Calculate viral coefficient
        viral_coefficient = successful_conversions / max(total_users - successful_conversions, 1)
        
        return {
            "total_users": total_users,
            "total_referrals": total_referrals,
            "successful_conversions": successful_conversions,
            "conversion_rate": successful_conversions / max(total_referrals, 1),
            "recent_referrals": recent_referrals,
            "recent_conversions": recent_conversions,
            "viral_coefficient": viral_coefficient,
            "total_content_shares": total_content_shares,
            "viral_content_count": viral_content_count,
            "growth_metrics": {
                "monthly_growth": recent_conversions / max(total_users - recent_conversions, 1),
                "referral_effectiveness": recent_conversions / max(recent_referrals, 1),
                "content_engagement": viral_content_count / max(total_content_shares, 1) if total_content_shares > 0 else 0
            }
        }
    
    async def _get_default_program(self) -> ReferralProgram:
        """Get or create the default referral program."""
        
        program = self.db.query(ReferralProgram).filter(
            ReferralProgram.name == "default"
        ).first()
        
        if not program:
            program = await self.create_referral_program(
                name="default",
                description="Standard referral program",
                referrer_reward={
                    "type": RewardType.CREDITS.value,
                    "amount": 25,
                    "description": "$25 credit for successful referral"
                },
                referee_reward={
                    "type": RewardType.CREDITS.value,
                    "amount": 10,
                    "description": "$10 welcome bonus"
                }
            )
        
        return program
    
    async def _update_user_achievements(self, user_id: str):
        """Update user achievements after referral conversion."""
        
        # This would trigger achievement notifications
        # and update user's achievement status
        achievements = await self._get_user_achievements(user_id, [])
        
        # Store achievements in user preferences
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            if not user.preferences:
                user.preferences = {}
            
            user.preferences["achievements"] = [a["id"] for a in achievements]
            
            from sqlalchemy.orm import attributes
            attributes.flag_modified(user, 'preferences')
            
            self.db.commit()
        
        logger.info(
            "user_achievements_updated",
            user_id=user_id,
            achievement_count=len(achievements)
        )
    
    async def create_shareable_referral_content(
        self,
        user_id: str,
        content_type: str = "referral_invite"
    ) -> Dict[str, Any]:
        """Create shareable content for referral invitations."""
        
        # Generate referral code if user doesn't have one
        referral = self.db.query(UserReferral).filter(
            and_(
                UserReferral.referrer_user_id == user_id,
                UserReferral.status == "pending"
            )
        ).first()
        
        if not referral:
            referral = await self.generate_referral_code(user_id)
        
        user = self.db.query(User).filter(User.id == user_id).first()
        user_stats = await self.get_user_referral_stats(user_id)
        
        # Create personalized sharing content
        share_content = {
            "referral_code": referral.referral_code,
            "share_url": f"https://yourbot.com/join?ref={referral.referral_code}",
            "personalized_message": f"Hey! I've been using this amazing AI bot that's helped me with personal growth and conversations. Join me and we both get $25 in credits! 🚀",
            "social_media_posts": {
                "twitter": f"Just discovered this incredible AI companion that's genuinely helpful for personal growth and conversations. Join me and we both get $25! 🧠✨ https://yourbot.com/join?ref={referral.referral_code}",
                "instagram": f"Found my new favorite AI companion! 🤖 It's like having a wise friend who's always there. Join me and we both get $25 in credits to explore! Link in bio 🚀",
                "facebook": f"I've been blown away by this AI bot that helps with personal conversations and growth. It's not just another chatbot - it genuinely understands and helps. If you're curious about AI or want a smart conversation partner, check it out! We both get $25 when you join: https://yourbot.com/join?ref={referral.referral_code}"
            },
            "user_stats": {
                "current_referrals": user_stats["referrals_converted"],
                "achievements": user_stats["achievements"][:3],  # Top 3 achievements
                "rank": user_stats["rank"] if user_stats["rank"] <= 100 else None
            }
        }
        
        return share_content