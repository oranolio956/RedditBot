"""
Viral Integration Service

Integrates viral sharing mechanics seamlessly with existing bot functionality.
Automatically detects shareable moments, triggers content generation,
and enables effortless sharing within conversations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy.orm import Session
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import structlog

from app.models.sharing import ShareableContent, UserReferral, SocialPlatform
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.viral_engine import ViralEngine
from app.services.referral_tracker import ReferralTracker
from app.services.llm_service import LLMService

logger = structlog.get_logger(__name__)


@dataclass
class ShareNotification:
    """Notification for shareable content."""
    content: ShareableContent
    suggested_platforms: List[SocialPlatform]
    share_message: str
    viral_potential: str


class ViralIntegration:
    """
    Seamless integration of viral mechanics with bot conversations.
    
    Automatically detects shareable moments, suggests content creation,
    and provides easy sharing options without disrupting user experience.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
        self.viral_engine = ViralEngine(db, self.llm_service)
        self.referral_tracker = ReferralTracker(db)
        
        # Viral trigger thresholds
        self.auto_generation_threshold = 80.0  # Auto-generate content above this score
        self.suggestion_threshold = 65.0       # Suggest sharing above this score
        self.notification_cooldown = 300       # 5 minutes between notifications
        
        # User notification preferences
        self._last_notifications = {}
    
    async def process_conversation_message(
        self,
        conversation: Conversation,
        message: Message,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[ShareNotification]:
        """
        Process new conversation messages for viral opportunities.
        
        Called after each bot response to check if the conversation
        has reached a shareable moment.
        """
        
        try:
            user_id = str(conversation.user_id)
            
            # Check if user wants viral suggestions
            if not self._should_suggest_viral_content(user_id):
                return None
            
            # Check cooldown to avoid spam
            if not self._is_notification_allowed(user_id):
                return None
            
            # Analyze conversation for viral potential
            generated_content = await self.viral_engine.analyze_conversation_for_viral_content(
                conversation=conversation,
                real_time=True
            )
            
            if not generated_content:
                return None
            
            # Find highest scoring content
            best_content = max(generated_content, key=lambda c: c.viral_score)
            
            # Auto-generate if score is very high
            if best_content.viral_score >= self.auto_generation_threshold:
                await self._auto_generate_shareable_content(
                    best_content, conversation.user, update, context
                )
                return None
            
            # Suggest sharing if score is good
            elif best_content.viral_score >= self.suggestion_threshold:
                notification = await self._create_share_notification(
                    best_content, conversation.user
                )
                
                await self._send_share_suggestion(
                    notification, update, context
                )
                
                self._update_notification_timestamp(user_id)
                return notification
            
        except Exception as e:
            logger.error("viral_integration_error", error=str(e))
        
        return None
    
    def _should_suggest_viral_content(self, user_id: str) -> bool:
        """Check if user has enabled viral content suggestions."""
        
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.preferences:
            return True  # Default to enabled
        
        return user.preferences.get("viral_suggestions_enabled", True)
    
    def _is_notification_allowed(self, user_id: str) -> bool:
        """Check if enough time has passed since last notification."""
        
        last_notification = self._last_notifications.get(user_id)
        if not last_notification:
            return True
        
        time_since_last = datetime.utcnow() - last_notification
        return time_since_last.total_seconds() > self.notification_cooldown
    
    def _update_notification_timestamp(self, user_id: str):
        """Update last notification timestamp for user."""
        self._last_notifications[user_id] = datetime.utcnow()
    
    async def _create_share_notification(
        self,
        content: ShareableContent,
        user: User
    ) -> ShareNotification:
        """Create a share notification for the user."""
        
        # Determine best platforms for this content
        suggested_platforms = []
        for platform_name in content.optimal_platforms or []:
            try:
                platform = SocialPlatform(platform_name)
                suggested_platforms.append(platform)
            except ValueError:
                continue
        
        # Default platforms if none specified
        if not suggested_platforms:
            suggested_platforms = [
                SocialPlatform.TWITTER,
                SocialPlatform.INSTAGRAM,
                SocialPlatform.TIKTOK
            ]
        
        # Generate personalized share message
        share_message = await self._generate_share_message(content, user)
        
        # Determine viral potential description
        if content.viral_score >= 85:
            viral_potential = "🚀 EXTREMELY HIGH - This could go viral!"
        elif content.viral_score >= 75:
            viral_potential = "⭐ HIGH - Great sharing potential"
        elif content.viral_score >= 65:
            viral_potential = "✨ GOOD - Worth sharing"
        else:
            viral_potential = "💡 MODERATE - Might resonate with some"
        
        return ShareNotification(
            content=content,
            suggested_platforms=suggested_platforms,
            share_message=share_message,
            viral_potential=viral_potential
        )
    
    async def _generate_share_message(
        self,
        content: ShareableContent,
        user: User
    ) -> str:
        """Generate personalized share message for the user."""
        
        # Get user's referral stats for personalization
        stats = await self.referral_tracker.get_user_referral_stats(user.id)
        
        # Personalize based on user's sharing history
        if stats["referrals_sent"] == 0:
            motivation = "Share your first piece of viral content and earn $25 in credits!"
        elif stats["referrals_sent"] < 5:
            motivation = f"You've shared {stats['referrals_sent']} times - keep building your network!"
        else:
            motivation = f"You're a sharing superstar with {stats['referrals_sent']} referrals! 🌟"
        
        share_message = f"""
✨ **Shareable Moment Detected!**

Your conversation just created something special that others would love to see.

**Content**: {content.title}
**Viral Potential**: {content.viral_score:.0f}/100

{motivation}

Would you like to share this moment with your network?
        """.strip()
        
        return share_message
    
    async def _send_share_suggestion(
        self,
        notification: ShareNotification,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Send sharing suggestion to user via Telegram."""
        
        # Create sharing options keyboard
        keyboard = []
        
        # Platform-specific sharing buttons
        platform_buttons = []
        for platform in notification.suggested_platforms[:3]:  # Limit to 3 platforms
            platform_emoji = {
                SocialPlatform.TWITTER: "🐦",
                SocialPlatform.INSTAGRAM: "📸",
                SocialPlatform.TIKTOK: "🎵",
                SocialPlatform.LINKEDIN: "💼",
                SocialPlatform.FACEBOOK: "👥"
            }.get(platform, "📱")
            
            platform_buttons.append(
                InlineKeyboardButton(
                    f"{platform_emoji} {platform.value.title()}",
                    callback_data=f"share_platform:{notification.content.id}:{platform.value}"
                )
            )
        
        if platform_buttons:
            keyboard.append(platform_buttons)
        
        # Additional options
        keyboard.append([
            InlineKeyboardButton(
                "🎨 Customize Content",
                callback_data=f"customize_share:{notification.content.id}"
            ),
            InlineKeyboardButton(
                "👀 Preview First",
                callback_data=f"preview_share:{notification.content.id}"
            )
        ])
        
        keyboard.append([
            InlineKeyboardButton(
                "📊 View Analytics",
                callback_data=f"share_analytics"
            ),
            InlineKeyboardButton(
                "❌ Not Now",
                callback_data="dismiss_share_suggestion"
            )
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                notification.share_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error("send_share_suggestion_error", error=str(e))
    
    async def _auto_generate_shareable_content(
        self,
        content: ShareableContent,
        user: User,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Auto-generate and offer shareable content for high-scoring moments."""
        
        # Mark content as published since it's high quality
        content.is_published = True
        content.published_at = datetime.utcnow()
        self.db.commit()
        
        # Generate referral code if user doesn't have one
        existing_referral = self.db.query(UserReferral).filter(
            UserReferral.referrer_user_id == user.id,
            UserReferral.status == "pending"
        ).first()
        
        if not existing_referral:
            referral = await self.referral_tracker.generate_referral_code(
                user_id=user.id,
                shared_content_id=content.id
            )
        else:
            referral = existing_referral
        
        # Create ready-to-share content
        share_url = f"https://yourbot.com/share/{content.id}?ref={referral.referral_code}"
        
        auto_message = f"""
🎉 **Viral Content Created!**

Your conversation just generated something amazing with a {content.viral_score:.0f}/100 viral score!

**"{content.title}"**

I've automatically created shareable content for you. Here's your personalized sharing link:
{share_url}

Share this and earn $25 when someone joins through your link! 💰
        """.strip()
        
        # Create sharing keyboard
        keyboard = [
            [
                InlineKeyboardButton("📱 Share Now", url=f"tg://msg?text={share_url}"),
                InlineKeyboardButton("🎨 Customize", callback_data=f"customize_share:{content.id}")
            ],
            [
                InlineKeyboardButton("📊 View Content", callback_data=f"view_content:{content.id}"),
                InlineKeyboardButton("🏆 Leaderboard", callback_data="view_leaderboard")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                auto_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error("auto_share_message_error", error=str(e))
    
    async def handle_share_callback(
        self,
        query_data: str,
        user: User,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle sharing-related callback queries."""
        
        parts = query_data.split(":")
        action = parts[0]
        
        try:
            if action == "share_platform":
                content_id, platform = parts[1], parts[2]
                await self._handle_platform_share(content_id, platform, user, update, context)
            
            elif action == "customize_share":
                content_id = parts[1]
                await self._handle_customize_share(content_id, user, update, context)
            
            elif action == "preview_share":
                content_id = parts[1]
                await self._handle_preview_share(content_id, user, update, context)
            
            elif action == "view_content":
                content_id = parts[1]
                await self._handle_view_content(content_id, user, update, context)
            
            elif action == "share_analytics":
                await self._handle_share_analytics(user, update, context)
            
            elif action == "view_leaderboard":
                await self._handle_view_leaderboard(user, update, context)
            
            elif action == "dismiss_share_suggestion":
                await update.callback_query.edit_message_text(
                    "No problem! I'll keep looking for more shareable moments. ✨"
                )
            
        except Exception as e:
            logger.error("share_callback_error", action=action, error=str(e))
    
    async def _handle_platform_share(
        self,
        content_id: str,
        platform: str,
        user: User,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle sharing to a specific platform."""
        
        content = self.db.query(ShareableContent).filter(
            ShareableContent.id == content_id
        ).first()
        
        if not content:
            await update.callback_query.edit_message_text("Content not found.")
            return
        
        # Get platform-optimized content
        try:
            platform_enum = SocialPlatform(platform)
            optimized = await self.viral_engine.optimize_content_for_platform(
                content=content,
                platform=platform_enum
            )
        except ValueError:
            await update.callback_query.edit_message_text("Invalid platform.")
            return
        
        # Generate sharing text
        share_text = f"{optimized['title']}\n\n{optimized.get('description', '')}"
        hashtags = " ".join(f"#{tag}" for tag in optimized.get('hashtags', []))
        if hashtags:
            share_text += f"\n\n{hashtags}"
        
        # Add referral link
        referral = await self.referral_tracker.generate_referral_code(
            user_id=user.id,
            shared_content_id=content.id
        )
        share_text += f"\n\n{optimized['share_url']}&ref={referral.referral_code}"
        
        platform_urls = {
            "twitter": f"https://twitter.com/intent/tweet?text={share_text}",
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={optimized['share_url']}&quote={share_text}",
            "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={optimized['share_url']}"
        }
        
        keyboard = []
        if platform in platform_urls:
            keyboard.append([
                InlineKeyboardButton(
                    f"Open {platform.title()}",
                    url=platform_urls[platform]
                )
            ])
        
        keyboard.append([
            InlineKeyboardButton("📋 Copy Text", callback_data=f"copy_share_text:{content_id}"),
            InlineKeyboardButton("🔙 Back", callback_data=f"preview_share:{content_id}")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            f"**Ready to share on {platform.title()}!**\n\n{share_text}",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def _handle_share_analytics(
        self,
        user: User,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Show user's sharing analytics."""
        
        stats = await self.referral_tracker.get_user_referral_stats(user.id)
        
        analytics_text = f"""
📊 **Your Sharing Analytics**

**Referrals**: {stats['referrals_sent']} sent, {stats['referrals_converted']} converted
**Success Rate**: {stats['conversion_rate']:.1%}
**Rank**: #{stats['rank']} on leaderboard
**Points**: {stats['total_points']}
**Credits**: ${stats['credits']}

**Recent Performance**:
• Content Shares: {stats['content_shares']}
• Achievements: {len(stats['achievements'])}

Keep sharing to climb the leaderboard! 🚀
        """.strip()
        
        keyboard = [
            [
                InlineKeyboardButton("🏆 Full Leaderboard", callback_data="view_leaderboard"),
                InlineKeyboardButton("🎯 Generate Referral", callback_data="generate_referral")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="dismiss_share_suggestion")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            analytics_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def _handle_view_leaderboard(
        self,
        user: User,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Show referral leaderboard."""
        
        leaderboard = await self.referral_tracker.get_referral_leaderboard(limit=10)
        
        leaderboard_text = "🏆 **Referral Leaderboard**\n\n"
        
        for entry in leaderboard:
            rank_emoji = {"1": "🥇", "2": "🥈", "3": "🥉"}.get(str(entry.rank), "🏅")
            
            badges_text = "".join(entry.badges) if entry.badges else ""
            leaderboard_text += f"{rank_emoji} **{entry.display_name}**\n"
            leaderboard_text += f"   {entry.conversion_count} conversions • {entry.total_points} pts {badges_text}\n\n"
        
        # Find user's position if not in top 10
        user_rank = await self.referral_tracker._get_user_rank(user.id)
        if user_rank > 10:
            stats = await self.referral_tracker.get_user_referral_stats(user.id)
            leaderboard_text += f"...\n🔹 **You**: Rank #{user_rank} • {stats['referrals_converted']} conversions • {stats['total_points']} pts"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="view_leaderboard")],
            [InlineKeyboardButton("🔙 Back", callback_data="share_analytics")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            leaderboard_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def check_referral_conversions(self):
        """
        Background task to check for referral conversions.
        
        Should be called periodically to process user activity
        and convert referrals when users become active.
        """
        
        try:
            # Find users who might have reached conversion threshold
            recent_active_users = self.db.query(User).filter(
                User.message_count >= 5,  # Activity threshold
                User.updated_at >= datetime.utcnow() - timedelta(days=1)
            ).all()
            
            for user in recent_active_users:
                await self.referral_tracker.process_referral_conversion(
                    user_id=user.id,
                    activity_threshold=5
                )
            
            logger.info(
                "referral_conversions_checked",
                users_processed=len(recent_active_users)
            )
            
        except Exception as e:
            logger.error("referral_conversion_check_error", error=str(e))
    
    async def generate_daily_viral_metrics(self):
        """
        Generate daily viral metrics for analytics.
        
        Should be called daily to aggregate viral performance data.
        """
        
        try:
            from app.models.sharing import ViralMetrics
            from sqlalchemy import func
            
            today = datetime.utcnow().date()
            
            # Check if metrics already exist for today
            existing = self.db.query(ViralMetrics).filter(
                ViralMetrics.date == today,
                ViralMetrics.period_type == "daily"
            ).first()
            
            if existing:
                logger.info("daily_metrics_already_exist", date=today)
                return
            
            # Calculate daily metrics
            yesterday = today - timedelta(days=1)
            
            # Content metrics
            daily_content = self.db.query(ShareableContent).filter(
                func.date(ShareableContent.created_at) == yesterday
            )
            
            total_content_created = daily_content.count()
            total_views = daily_content.with_entities(
                func.sum(ShareableContent.view_count)
            ).scalar() or 0
            total_shares = daily_content.with_entities(
                func.sum(ShareableContent.share_count)
            ).scalar() or 0
            
            # Referral metrics
            daily_referrals = self.db.query(UserReferral).filter(
                func.date(UserReferral.created_at) == yesterday
            )
            
            total_referrals_sent = daily_referrals.count()
            total_referrals_converted = daily_referrals.filter(
                UserReferral.status == "converted"
            ).count()
            
            # Calculate viral coefficient
            total_users = self.db.query(User).count()
            viral_coefficient = total_referrals_converted / max(total_users, 1)
            
            # Create metrics record
            metrics = ViralMetrics(
                date=yesterday,
                period_type="daily",
                total_content_created=total_content_created,
                total_content_shared=total_shares,
                total_views=total_views,
                total_shares=total_shares,
                total_referrals_sent=total_referrals_sent,
                total_referrals_converted=total_referrals_converted,
                viral_coefficient=viral_coefficient
            )
            
            self.db.add(metrics)
            self.db.commit()
            
            logger.info(
                "daily_viral_metrics_generated",
                date=yesterday,
                content_created=total_content_created,
                referrals_sent=total_referrals_sent,
                viral_coefficient=viral_coefficient
            )
            
        except Exception as e:
            logger.error("daily_metrics_generation_error", error=str(e))